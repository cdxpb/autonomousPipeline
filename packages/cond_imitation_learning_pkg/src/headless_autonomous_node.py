#!/usr/bin/env python3
import sys
import os
import json
import argparse
import time
import rospy
import cv2
import numpy as np
from PIL import Image

from duckietown_msgs.msg import Twist2DStamped, WheelsCmdStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import crop_image, INTENT_MAP

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # Not pycuda.autoinit: that binds the context to the importing (main) thread,
    # but rospy dispatches image_cb on a different thread. We manage the context
    # explicitly instead and push/pop it around each callback (see image_cb).
    cuda.init()
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# two YOLO backbones, picked via rospy param ~model_set ("smaller" default, or "full")
YOLO_SPEC = {
    "smaller": {"h": 256, "w": 320, "basename": "yolo_best_256x320", "argmax_fused": True,  "crop_before_yolo": False},
    "full":    {"h": 480, "w": 640, "basename": "yolo_model",        "argmax_fused": False, "crop_before_yolo": True},
}
# not interchangeable state_dicts between "smaller"/"full" for approach 5
PILOTNET_BASENAME = {
    (5, "smaller"): "best_model_approach5_smaller",
    (5, "full"):    "best_model_regNheadv2",
    (7, "smaller"): "best_model_approach7_smaller",
    (7, "full"):    "best_model_approach7",
}
# override crop_before_yolo per (approach, model_set) when it differs from the default above
CROP_BEFORE_YOLO_OVERRIDE = {
    (5, "full"): False,
}

class HeadlessAutonomousNode:
    def __init__(self, approach=7, output_mode=None):
        rospy.init_node('headless_autonomous_node', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'duckiexp')
        self.approach = approach
        
        if output_mode is None:
            self.output_mode = "twist" if self.approach == 3 else "wheels"
        else:
            self.output_mode = output_mode

        self.skip_segmentation = (self.approach == 0)
        self.current_intent = "stop"
        
        self.tuning = {"v_fwd": 0.30, "v_rev": -0.5, "v_bump_a": 0.1, "omega_a": 3.0, "v_bump_d": 0.1, "omega_d": 5.0}

        self.is_autonomous_active = True # Always active in headless
        
        self.device = None
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        # Models are transferred separately: ./transfer_models.sh
        MODEL_BASE = "/data/models"

        self.model_set = rospy.get_param("~model_set", "smaller")
        if self.model_set not in YOLO_SPEC:
            rospy.logwarn(f"Unknown model_set '{self.model_set}', defaulting to 'smaller'")
            self.model_set = "smaller"
        yolo_spec = YOLO_SPEC[self.model_set]
        self.yolo_input_h, self.yolo_input_w = yolo_spec["h"], yolo_spec["w"]
        self.yolo_argmax_fused = yolo_spec["argmax_fused"]
        yolo_basename = yolo_spec["basename"]

        if self.approach not in (5, 7):
            rospy.logwarn(f"Unsupported approach {self.approach} for headless mode, defaulting to 7")
            self.approach = 7
        self.crop_before_yolo = CROP_BEFORE_YOLO_OVERRIDE.get((self.approach, self.model_set), yolo_spec["crop_before_yolo"])
        self.model_path = f"{MODEL_BASE}/pilotnet/{PILOTNET_BASENAME[(self.approach, self.model_set)]}"
        self.yolo_path  = f"{MODEL_BASE}/yolo_model/{yolo_basename}.onnx"

        self.prev_out1 = 0.0
        self.prev_out2 = 0.0

        
        self.last_inference_time = time.time()
        self.last_inference_img_stamp = 0.0
        self.stale_cmd_warn_ms = rospy.get_param("~stale_cmd_warn_ms", 200.0)

        self.target_fps = rospy.get_param("~target_fps", 30)
        self.publish_telemetry = rospy.get_param("~publish_telemetry", True)
        # publish preview every Nth frame, not every frame (colormap+resize+encode ain't free)
        self.telemetry_every_n = max(1, rospy.get_param("~telemetry_every_n", 1))
        self.telemetry_frame_count = 0
        self.last_frame_time = 0.0

        # ~backend sets both stages' default; ~yolo_backend/~pilotnet_backend override
        # independently (e.g. YOLO on tensorrt, PilotNet on onnx).
        self.backend = rospy.get_param("~backend", "pytorch")
        self.yolo_backend = rospy.get_param("~yolo_backend", self.backend)
        self.pilotnet_backend = rospy.get_param("~pilotnet_backend", self.backend)

        self.pt_model = None
        self.ort_session = None

        self.cuda_ctx = None
        if self.yolo_backend.lower() == "tensorrt" or self.pilotnet_backend.lower() == "tensorrt":
            self.cuda_ctx = cuda.Device(0).make_context()

        # -- YOLO --
        if not self.skip_segmentation:
            if self.yolo_backend.lower() == "tensorrt":
                self.yolo_path = f"{MODEL_BASE}/yolo_model/{yolo_basename}.engine"
                rospy.loginfo(f"Loading YOLO TensorRT engine from {self.yolo_path} ...")
                if not os.path.exists(self.yolo_path):
                    rospy.logfatal(f"YOLO engine not found at {self.yolo_path}. Run export_trt.py --yolo --model-set {self.model_set} on the robot first.")
                    raise FileNotFoundError(self.yolo_path)
                self.yolo_engine, self.yolo_context = self._load_trt_engine(self.yolo_path)
                self.yolo_d_in = cuda.mem_alloc(1 * 3 * self.yolo_input_h * self.yolo_input_w * 4)   # float32 input
                if self.yolo_argmax_fused:
                    # int32 not uint8, this Jetson's TensorRT can't cast to uint8
                    self.yolo_d_out = cuda.mem_alloc(1 * self.yolo_input_h * self.yolo_input_w * 4)
                else:
                    # raw per-class logits [1,4,H,W] float32, needs a manual argmax (see image_cb)
                    self.yolo_d_out = cuda.mem_alloc(1 * 4 * self.yolo_input_h * self.yolo_input_w * 4)
                self.yolo_bindings = [int(self.yolo_d_in), int(self.yolo_d_out)]
            elif self.yolo_backend.lower() == "onnx":
                if not ORT_AVAILABLE:
                    rospy.logfatal("yolo_backend=onnx requires onnxruntime, which isn't installed on this box. Use --yolo-backend tensorrt or pytorch instead.")
                    raise RuntimeError("onnxruntime not available")
                self.yolo_path = f"{MODEL_BASE}/yolo_model/{yolo_basename}.onnx"
                rospy.loginfo(f"Loading YOLO ONNX from {self.yolo_path} ...")
                if not os.path.exists(self.yolo_path):
                    rospy.logfatal(f"YOLO model not found at {self.yolo_path}. Run ./transfer_models.sh --all")
                    raise FileNotFoundError(self.yolo_path)
                yolo_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.yolo_ort = ort.InferenceSession(self.yolo_path, providers=yolo_providers)
                rospy.loginfo(f"YOLO ONNX running on: {self.yolo_ort.get_providers()}")
            else:
                if not TORCH_AVAILABLE:
                    rospy.logfatal("yolo_backend=pytorch requires torch, which isn't installed on this box. Use --yolo-backend tensorrt or onnx instead.")
                    raise RuntimeError("torch not available")
                self.yolo_path = f"{MODEL_BASE}/yolo_model/{yolo_basename}.pt"
                rospy.loginfo(f"Loading YOLO PyTorch from {self.yolo_path} ...")
                if not os.path.exists(self.yolo_path):
                    rospy.logfatal(f"YOLO model not found at {self.yolo_path}. Run ./transfer_models.sh --all")
                    raise FileNotFoundError(self.yolo_path)
                from ultralytics import YOLO
                self.yolo_session = YOLO(self.yolo_path, task='semantic')

        # -- PilotNet --
        onnx_path = f"{self.model_path}.onnx"
        pt_path   = f"{self.model_path}.pt"
        engine_path = f"{self.model_path}.engine"

        if self.pilotnet_backend.lower() == "tensorrt":
            rospy.loginfo(f"Loading PilotNet TensorRT engine from {engine_path}")
            if not os.path.exists(engine_path):
                rospy.logfatal(f"PilotNet engine not found at {engine_path}. Run export_trt.py --approach {self.approach} --model-set {self.model_set} on the robot first.")
                raise FileNotFoundError(engine_path)
            self.pt_engine, self.pt_context = self._load_trt_engine(engine_path)
            self.pt_d_img = cuda.mem_alloc(1 * 1 * 112 * 224 * 4)
            self.pt_d_intent = cuda.mem_alloc(1 * 4 * 4)
            self.pt_d_out = cuda.mem_alloc(1 * 2 * 4)
            self.pt_bindings = [int(self.pt_d_img), int(self.pt_d_intent), int(self.pt_d_out)]
        elif self.pilotnet_backend.lower() == "onnx" and ORT_AVAILABLE and os.path.exists(onnx_path):
            # Use CUDA if available, otherwise CPU (Jetson onnxruntime-gpu needed for CUDA)
            ort_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            rospy.loginfo(f"Loading PilotNet ONNX from {onnx_path}")
            self.ort_session = ort.InferenceSession(onnx_path, providers=ort_providers)
            rospy.loginfo(f"PilotNet ONNX running on: {self.ort_session.get_providers()}")
        else:
            if not TORCH_AVAILABLE:
                rospy.logfatal("pilotnet_backend=pytorch requires torch, which isn't installed on this box. Use --pilotnet-backend tensorrt or onnx instead.")
                raise RuntimeError("torch not available")
            if self.pilotnet_backend.lower() == "onnx" and not ORT_AVAILABLE:
                rospy.logwarn("onnxruntime not installed, falling back to PyTorch")
            elif self.pilotnet_backend.lower() == "onnx":
                rospy.logwarn(f"ONNX not found at {onnx_path}, falling back to PyTorch")
            rospy.loginfo(f"Loading PilotNet PyTorch (Approach {self.approach}) on {self.device}")
            from models import get_pilotnet_model
            self.pt_model = get_pilotnet_model(self.approach, self.model_set, self.skip_segmentation).to(self.device)
            self.pt_model.load_state_dict(torch.load(pt_path, map_location=self.device, weights_only=False))
            self.pt_model.eval()

        # -- Pubs and Subs --
        if self.output_mode == "twist":
            self.cmd_pub = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1, tcp_nodelay=True)
        else:
            self.cmd_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1, tcp_nodelay=True)

        # TELEMETRY PUBLISHERS
        self.telemetry_img_pub = rospy.Publisher(f"/{self.veh}/telemetry/preview/compressed", CompressedImage, queue_size=1, tcp_nodelay=True)
        self.telemetry_state_pub = rospy.Publisher(f"/{self.veh}/telemetry/state", String, queue_size=1)

        self.image_sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.image_cb, queue_size=1, buff_size=2**24, tcp_nodelay=True)
        self.intent_sub = rospy.Subscriber(f"/{self.veh}/data_collector/intent", String, self.intent_cb)

        # motor commands go out on their own timer, decoupled from inference cadence,
        # so one slow frame doesn't leave the wheels driver hanging
        self.last_out1 = 0.0
        self.last_out2 = 0.0
        self.motor_pub_hz = rospy.get_param("~motor_pub_hz", 30)
        self.motor_timer = rospy.Timer(rospy.Duration(1.0 / self.motor_pub_hz), self._publish_motor_cmd)

        # EMA smoothing on published velocity, ~smoothing_alpha (default 1.0 = off)
        self.smoothing_alpha = rospy.get_param("~smoothing_alpha", 1.0)
        self.smoothing_alpha = max(1e-6, min(1.0, self.smoothing_alpha))
        self.smoothed_out1 = 0.0
        self.smoothed_out2 = 0.0

        # pop so it starts "not current anywhere", image_cb's push/pop then owns it
        if self.cuda_ctx is not None:
            self.cuda_ctx.pop()

        rospy.loginfo(f"Headless Inference Node is UP and RUNNING! (approach={self.approach}, model_set={self.model_set}, crop_before_yolo={self.crop_before_yolo})")

    def _publish_motor_cmd(self, event):
        cmd_age_ms = 1000 * (time.time() - self.last_inference_time)
        if cmd_age_ms > self.stale_cmd_warn_ms:
            rospy.logwarn_throttle(
                1.0,
                f"STALE velocity: last inference completed {cmd_age_ms:.0f}ms ago "
                f"(motor timer @ {self.motor_pub_hz}Hz keeps resending vel=({self.last_out1:.2f}, {self.last_out2:.2f})) "
                f"-- inference is falling behind, robot is driving on old data"
            )

        if self.output_mode == "twist":
            cmd_msg = Twist2DStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.v = float(self.last_out1)
            cmd_msg.omega = float(self.last_out2)
            self.cmd_pub.publish(cmd_msg)
        else:
            cmd_msg = WheelsCmdStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.vel_left = float(self.last_out1)
            cmd_msg.vel_right = float(self.last_out2)
            self.cmd_pub.publish(cmd_msg)

    def _load_trt_engine(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return engine, context

    @staticmethod
    def _letterbox(image_rgb, target_h, target_w, pad_value=114):
        # matches Ultralytics' own LetterBox preprocessing (aspect-preserving resize + gray pad)
        h0, w0 = image_rgb.shape[:2]
        scale = min(target_h / h0, target_w / w0)
        new_h, new_w = int(round(h0 * scale)), int(round(w0 * scale))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_h, pad_w = target_h - new_h, target_w - new_w
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
        return padded, scale, top, left

    @staticmethod
    def _unletterbox_mask(seg_map_padded, orig_h, orig_w, scale, pad_top, pad_left):
        new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
        content = seg_map_padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
        return cv2.resize(content, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    def intent_cb(self, msg):
        if msg.data in ["straight", "left", "right", "stop", "lane_following"]:
            self.current_intent = msg.data

    def image_cb(self, msg):
        # rospy may call this on a different thread than __init__ ran on, so push/pop
        # the CUDA context around the whole callback
        if self.cuda_ctx is not None:
            self.cuda_ctx.push()
        try:
            rospy.loginfo_throttle(5.0, "image_cb: frame received")
            current_time = time.time()
            if (current_time - self.last_frame_time) < (1.0 / self.target_fps):
                return
            self.last_frame_time = current_time

            t_start = time.time()
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None or cv_image.size == 0:
                return

            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            t_decode = time.time()

            # -- YOLO semantic segmentation --
            if self.crop_before_yolo:
                seg_input_pil = crop_image(pil_image)
                seg_input_arr = np.array(seg_input_pil)
            else:
                seg_input_pil = pil_image
                seg_input_arr = rgb_image

            orig_h, orig_w = seg_input_arr.shape[:2]
            letterboxed, lb_scale, lb_pad_top, lb_pad_left = self._letterbox(
                seg_input_arr, self.yolo_input_h, self.yolo_input_w
            )

            if self.yolo_backend.lower() == "tensorrt":
                yolo_in = letterboxed.astype(np.float32) / 255.0    # HWC float32
                yolo_in = np.ascontiguousarray(yolo_in.transpose(2, 0, 1)[np.newaxis])  # [1,3,H,W]
                cuda.memcpy_htod(self.yolo_d_in, yolo_in)
                self.yolo_context.execute_v2(bindings=self.yolo_bindings)
                if self.yolo_argmax_fused:
                    # already the argmaxed class map {0,1,2,3}; int32 not uint8, same reason as above
                    raw_out = np.empty((self.yolo_input_h, self.yolo_input_w), dtype=np.int32)
                    cuda.memcpy_dtoh(raw_out, self.yolo_d_out)
                    seg_map_padded = raw_out.astype(np.uint8)
                else:
                    # Raw per-class logits [1,4,H,W] float32, needs a manual argmax.
                    raw_out = np.empty((1, 4, self.yolo_input_h, self.yolo_input_w), dtype=np.float32)
                    cuda.memcpy_dtoh(raw_out, self.yolo_d_out)
                    seg_map_padded = np.argmax(raw_out[0], axis=0).astype(np.uint8)
                seg_map_full = self._unletterbox_mask(seg_map_padded, orig_h, orig_w, lb_scale, lb_pad_top, lb_pad_left)
            elif self.yolo_backend.lower() == "onnx":
                yolo_in = letterboxed.astype(np.float32) / 255.0    # HWC float32
                yolo_in = yolo_in.transpose(2, 0, 1)[np.newaxis]   # [1,3,H,W]
                raw_out = self.yolo_ort.run(None, {'images': yolo_in})[0]
                if self.yolo_argmax_fused:
                    seg_map_padded = raw_out[0].astype(np.uint8)  # [H,W], already argmaxed
                else:
                    seg_map_padded = np.argmax(raw_out[0], axis=0).astype(np.uint8)  # [1,4,H,W] -> [H,W]
                seg_map_full = self._unletterbox_mask(seg_map_padded, orig_h, orig_w, lb_scale, lb_pad_top, lb_pad_left)
            else:
                with torch.no_grad():
                    yolo_results = self.yolo_session(seg_input_pil, verbose=False, device=self.device.type)
                seg_map_full = yolo_results[0].semantic_mask.data.cpu().numpy().astype(np.uint8)

            # only crop now if YOLO ran on the full frame, otherwise seg_map_full is already
            # cropped and crop_image() again here would double-crop it
            seg_map = seg_map_full if self.crop_before_yolo else np.array(crop_image(Image.fromarray(seg_map_full)))
            t_yolo = time.time()

            # PIL resize not cv2's, to match how the training pipeline resizes masks
            mask_resized = np.array(Image.fromarray(seg_map).resize((224, 112), Image.NEAREST))
            # uint8 / 255.0 on raw class indices, matching training (semantic_mask.data -> to_tensor -> /255)
            img_tensor = (mask_resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis]  # [1,1,112,224]

            # telemetry preview: skippable via ~publish_telemetry:=false, throttled via ~telemetry_every_n:=N
            self.telemetry_frame_count += 1
            if self.publish_telemetry and (self.telemetry_frame_count % self.telemetry_every_n == 0):
                # reuses `letterboxed`, the exact image YOLO ran on, so the UI shows the model's real input
                raw_preview = cv2.cvtColor(letterboxed, cv2.COLOR_RGB2BGR)
                mask_preview_gray = (seg_map.astype(np.uint8) * 85)
                mask_preview_color = cv2.applyColorMap(mask_preview_gray, cv2.COLORMAP_JET)
                mask_preview_color = cv2.resize(mask_preview_color, (self.yolo_input_w, self.yolo_input_h))
                telemetry_combo = np.hstack((raw_preview, mask_preview_color))

                if self.telemetry_img_pub.get_num_connections() > 0:
                    msg_out = CompressedImage()
                    msg_out.header.stamp = rospy.Time.now()
                    msg_out.format = "jpeg"
                    msg_out.data = np.array(cv2.imencode('.jpg', telemetry_combo, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).tobytes()
                    self.telemetry_img_pub.publish(msg_out)
            t_telemetry = time.time()

            # PilotNet Inference
            out1, out2 = 0.0, 0.0
            if self.current_intent != "stop":
                # INTENT_MAP natively returns a one-hot list like [1.0, 0.0, 0.0, 0.0]
                intent_list = INTENT_MAP.get(self.current_intent, [1.0, 0.0, 0.0, 0.0])

                if self.pilotnet_backend.lower() == "tensorrt":
                    img_in = np.ascontiguousarray(img_tensor)
                    intent_in = np.array([intent_list], dtype=np.float32)
                    h_output = np.empty((1, 2), dtype=np.float32)
                    cuda.memcpy_htod(self.pt_d_img, img_in)
                    cuda.memcpy_htod(self.pt_d_intent, intent_in)
                    self.pt_context.execute_v2(bindings=self.pt_bindings)
                    cuda.memcpy_dtoh(h_output, self.pt_d_out)
                    out1, out2 = float(h_output[0][0]), float(h_output[0][1])
                elif self.pt_model is not None:
                    pt_tensor = torch.from_numpy(img_tensor).to(self.device)
                    intent_tensor = torch.tensor([intent_list], dtype=torch.float32, device=self.device)

                    output = self.pt_model(pt_tensor, intent_tensor)
                    out1, out2 = output[0][0].item(), output[0][1].item()
                else:
                    intent_val = np.array([intent_list], dtype=np.float32)

                    ort_inputs = {
                        "image_input": img_tensor,
                        "intent_input": intent_val
                    }
                    output = self.ort_session.run(None, ort_inputs)[0]
                    out1, out2 = output[0][0].item(), output[0][1].item()

            # clamp to [-1, 1], raw model output isn't bounded (telemetry still shows the
            # pre-clamp value so you can compare)
            out1_clamped = max(-1.0, min(1.0, out1))
            out2_clamped = max(-1.0, min(1.0, out2))
            if abs(out1) > 1.0 or abs(out2) > 1.0:
                rospy.logwarn_throttle(2.0, f"PilotNet output out of [-1,1] range: ({out1:.2f}, {out2:.2f}), clamped to ({out1_clamped:.2f}, {out2_clamped:.2f})")

            # EMA smoothing, no-op when smoothing_alpha=1.0 (the default)
            self.smoothed_out1 = self.smoothing_alpha * out1_clamped + (1 - self.smoothing_alpha) * self.smoothed_out1
            self.smoothed_out2 = self.smoothing_alpha * out2_clamped + (1 - self.smoothing_alpha) * self.smoothed_out2
            self.last_out1 = self.smoothed_out1
            self.last_out2 = self.smoothed_out2
            self.last_inference_time = time.time()
            self.last_inference_img_stamp = msg.header.stamp.to_sec()

            # State Telemetry (raw model output, pre-clamp/pre-smoothing, so you can compare
            # against what's actually being sent to the motors)
            state_data = {
                "out1": float(out1),
                "out2": float(out2),
                "smoothed_out1": float(self.smoothed_out1),
                "smoothed_out2": float(self.smoothed_out2),
                "intent": self.current_intent,
                "backend": f"yolo={self.yolo_backend}/pilotnet={self.pilotnet_backend}"
            }
            self.telemetry_state_pub.publish(String(json.dumps(state_data)))

            t_end = time.time()
            rospy.loginfo_throttle(
                2.0,
                f"timing(ms): decode={1000*(t_decode-t_start):.1f} "
                f"yolo={1000*(t_yolo-t_decode):.1f} "
                f"telemetry={1000*(t_telemetry-t_yolo):.1f} "
                f"pilotnet={1000*(t_end-t_telemetry):.1f} "
                f"total={1000*(t_end-t_start):.1f}"
            )

        except Exception as e:
            rospy.logerr_throttle(2.0, f"Inference Error: {e}")
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", type=int, default=7)
    parser.add_argument("--output_mode", type=str, default="wheels")
    args, _ = parser.parse_known_args(rospy.myargv()[1:])
    
    node = HeadlessAutonomousNode(approach=args.approach, output_mode=args.output_mode)
    rospy.spin()
