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

# YOLO ONNX model spec:
# Input:  images  [1, 3, 480, 640]  float32 (RGB, normalised 0-1)
# Output: output0 [1, 4, 480, 640]  float32 (4-class logits)
# argmax gives semantic class map {0,1,2,3}; matches task='semantic' training
YOLO_INPUT_H, YOLO_INPUT_W = 480, 640

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

        # Models are transferred separately to /data/models/ on the Duckiebot
        # Use: rsync -av models/ duckie@duckiexp.local:/data/models/
        MODEL_BASE = "/data/models"
        if self.approach == 5:
            self.model_path = f"{MODEL_BASE}/pilotnet/best_model_regNheadv2"
            self.yolo_path  = f"{MODEL_BASE}/yolo_model/yolo_model.onnx"
        elif self.approach == 7:
            self.model_path = f"{MODEL_BASE}/pilotnet/best_model_approach7"
            self.yolo_path  = f"{MODEL_BASE}/yolo_model/yolo_model.onnx"
        else:
            rospy.logwarn(f"Unsupported approach {self.approach} for headless mode, defaulting to 7")
            self.approach = 7
            self.model_path = f"{MODEL_BASE}/pilotnet/best_model_approach7"
            self.yolo_path  = f"{MODEL_BASE}/yolo_model/yolo_model.onnx"

        self.prev_out1 = 0.0
        self.prev_out2 = 0.0
        
        # Uncapped for now (set high enough that the throttle never actually triggers --
        # the camera's own native rate is the real ceiling). Tune via _target_fps:=N.
        self.target_fps = rospy.get_param("~target_fps", 1000)
        self.last_frame_time = 0.0

        self.backend = rospy.get_param("~backend", "pytorch")

        self.pt_model = None
        self.ort_session = None

        self.cuda_ctx = None
        if self.backend.lower() == "tensorrt":
            self.cuda_ctx = cuda.Device(0).make_context()

        # ── YOLO ─────────
        if not self.skip_segmentation:
            if self.backend.lower() == "tensorrt":
                self.yolo_path = f"{MODEL_BASE}/yolo_model/yolo_model.engine"
                rospy.loginfo(f"Loading YOLO TensorRT engine from {self.yolo_path} ...")
                if not os.path.exists(self.yolo_path):
                    rospy.logfatal(f"YOLO engine not found at {self.yolo_path}. Run export_trt.py --yolo on the robot first.")
                    raise FileNotFoundError(self.yolo_path)
                self.yolo_engine, self.yolo_context = self._load_trt_engine(self.yolo_path)
                self.yolo_d_in = cuda.mem_alloc(1 * 3 * YOLO_INPUT_H * YOLO_INPUT_W * 4)
                self.yolo_d_out = cuda.mem_alloc(1 * 4 * YOLO_INPUT_H * YOLO_INPUT_W * 4)
                self.yolo_bindings = [int(self.yolo_d_in), int(self.yolo_d_out)]
            elif self.backend.lower() == "onnx":
                self.yolo_path = f"{MODEL_BASE}/yolo_model/yolo_model.onnx"
                rospy.loginfo(f"Loading YOLO ONNX from {self.yolo_path} ...")
                if not os.path.exists(self.yolo_path):
                    rospy.logfatal(f"YOLO model not found at {self.yolo_path}. Run ./transfer_models.sh --all")
                    raise FileNotFoundError(self.yolo_path)
                yolo_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.yolo_ort = ort.InferenceSession(self.yolo_path, providers=yolo_providers)
                rospy.loginfo(f"YOLO ONNX running on: {self.yolo_ort.get_providers()}")
            else:
                if not TORCH_AVAILABLE:
                    rospy.logfatal("backend=pytorch requires torch, which isn't installed on this box. Use --backend tensorrt or onnx instead.")
                    raise RuntimeError("torch not available")
                self.yolo_path = f"{MODEL_BASE}/yolo_model/yolo_model.pt"
                rospy.loginfo(f"Loading YOLO PyTorch from {self.yolo_path} ...")
                if not os.path.exists(self.yolo_path):
                    rospy.logfatal(f"YOLO model not found at {self.yolo_path}. Run ./transfer_models.sh --all")
                    raise FileNotFoundError(self.yolo_path)
                from ultralytics import YOLO
                self.yolo_session = YOLO(self.yolo_path, task='semantic')

        # ── PilotNet ──────────────────────────────────────────────────────────
        onnx_path = f"{self.model_path}.onnx"
        pt_path   = f"{self.model_path}.pt"
        engine_path = f"{self.model_path}.engine"

        if self.backend.lower() == "tensorrt":
            rospy.loginfo(f"Loading PilotNet TensorRT engine from {engine_path}")
            if not os.path.exists(engine_path):
                rospy.logfatal(f"PilotNet engine not found at {engine_path}. Run export_trt.py --approach {self.approach} on the robot first.")
                raise FileNotFoundError(engine_path)
            self.pt_engine, self.pt_context = self._load_trt_engine(engine_path)
            self.pt_d_img = cuda.mem_alloc(1 * 1 * 112 * 224 * 4)
            self.pt_d_intent = cuda.mem_alloc(1 * 4 * 4)
            self.pt_d_out = cuda.mem_alloc(1 * 2 * 4)
            self.pt_bindings = [int(self.pt_d_img), int(self.pt_d_intent), int(self.pt_d_out)]
        elif self.backend.lower() == "onnx" and os.path.exists(onnx_path):
            # Use CUDA if available, otherwise CPU (Jetson onnxruntime-gpu needed for CUDA)
            ort_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            rospy.loginfo(f"Loading PilotNet ONNX from {onnx_path}")
            self.ort_session = ort.InferenceSession(onnx_path, providers=ort_providers)
            rospy.loginfo(f"PilotNet ONNX running on: {self.ort_session.get_providers()}")
        else:
            if not TORCH_AVAILABLE:
                rospy.logfatal("backend=pytorch requires torch, which isn't installed on this box. Use --backend tensorrt or onnx instead.")
                raise RuntimeError("torch not available")
            if self.backend.lower() == "onnx":
                rospy.logwarn(f"ONNX not found at {onnx_path}, falling back to PyTorch")
            rospy.loginfo(f"Loading PilotNet PyTorch (Approach {self.approach}) on {self.device}")
            if self.approach == 7:
                from pilotnet_FiLM import ConditionalPilotNetFiLM
                self.pt_model = ConditionalPilotNetFiLM().to(self.device)
            elif self.approach == 5:
                from pilotnet_regNheadv2 import ConditionalPilotNet
                self.pt_model = ConditionalPilotNet().to(self.device)
            self.pt_model.load_state_dict(torch.load(pt_path, map_location=self.device, weights_only=False))
            self.pt_model.eval()

        # ── Pubs and Subs ─────────────────────────────────────────────────────
        if self.output_mode == "twist":
            self.cmd_pub = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1, tcp_nodelay=True)
        else:
            self.cmd_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1, tcp_nodelay=True)
            
        # TELEMETRY PUBLISHERS
        self.telemetry_img_pub = rospy.Publisher(f"/{self.veh}/telemetry/preview/compressed", CompressedImage, queue_size=1, tcp_nodelay=True)
        self.telemetry_state_pub = rospy.Publisher(f"/{self.veh}/telemetry/state", String, queue_size=1)

        self.image_sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.image_cb, queue_size=1, buff_size=2**24, tcp_nodelay=True)
        self.intent_sub = rospy.Subscriber(f"/{self.veh}/data_collector/intent", String, self.intent_cb)
        
        # Pop the context now that init-time CUDA calls (engine load, mem_alloc) are done,
        # so it starts "not current anywhere" -- image_cb's push/pop then owns it cleanly
        # on whichever thread rospy dispatches callbacks on.
        if self.cuda_ctx is not None:
            self.cuda_ctx.pop()

        rospy.loginfo("Headless Inference Node is UP and RUNNING!")

    def _load_trt_engine(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return engine, context

    def intent_cb(self, msg):
        if msg.data in ["straight", "left", "right", "stop", "lane_following"]:
            self.current_intent = msg.data

    def image_cb(self, msg):
        # rospy may invoke this on a different thread than __init__ ran on;
        # the CUDA context must be pushed as current on whichever thread calls
        # into pycuda/tensorrt, hence the explicit push/pop around this whole callback.
        if self.cuda_ctx is not None:
            self.cuda_ctx.push()
        try:
            rospy.loginfo_throttle(5.0, "image_cb: frame received")
            current_time = time.time()
            if (current_time - self.last_frame_time) < (1.0 / self.target_fps):
                return
            self.last_frame_time = current_time

            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None or cv_image.size == 0:
                return

            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            cropped_img = crop_image(pil_image)  # PIL RGB, cropped

            # ── YOLO semantic segmentation ───────────────────────
            if self.backend.lower() == "tensorrt":
                yolo_in = cv2.resize(
                    np.array(cropped_img), (YOLO_INPUT_W, YOLO_INPUT_H)
                ).astype(np.float32) / 255.0                        # HWC float32
                yolo_in = np.ascontiguousarray(yolo_in.transpose(2, 0, 1)[np.newaxis])  # [1,3,H,W]
                logits = np.empty((1, 4, YOLO_INPUT_H, YOLO_INPUT_W), dtype=np.float32)
                cuda.memcpy_htod(self.yolo_d_in, yolo_in)
                self.yolo_context.execute_v2(bindings=self.yolo_bindings)
                cuda.memcpy_dtoh(logits, self.yolo_d_out)
                seg_map = np.argmax(logits[0], axis=0).astype(np.uint8)
            elif self.backend.lower() == "onnx":
                # Resize to YOLO input size, normalise, CHW
                yolo_in = cv2.resize(
                    np.array(cropped_img), (YOLO_INPUT_W, YOLO_INPUT_H)
                ).astype(np.float32) / 255.0                        # HWC float32
                yolo_in = yolo_in.transpose(2, 0, 1)[np.newaxis]   # [1,3,H,W]
                logits = self.yolo_ort.run(None, {'images': yolo_in})[0]  # [1,4,H,W]
                # argmax over class dim → uint8 class map {0,1,2,3}, shape [H,W]
                seg_map = np.argmax(logits[0], axis=0).astype(np.uint8)
            else:
                with torch.no_grad():
                    yolo_results = self.yolo_session(cropped_img, verbose=False, device=self.device.type)
                seg_map = yolo_results[0].semantic_mask.data.cpu().numpy().astype(np.uint8)

            # Resize to PilotNet input size [112, 224]
            mask_resized = cv2.resize(seg_map, (224, 112), interpolation=cv2.INTER_NEAREST)
            # Normalise exactly as TF.to_tensor does: uint8 / 255.0
            # This matches the training pipeline (semantic_mask.data → to_tensor → /255)
            img_tensor = (mask_resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis]  # [1,1,112,224]
            
            # Publish Telemetry Image (Low Res side-by-side)
            # We construct a 320x120 side-by-side image to send over network (extremely small payload!)
            raw_preview = cv2.resize(cv_image, (160, 120))
            mask_preview_gray = mask_resized * 85
            mask_preview_color = cv2.applyColorMap(mask_preview_gray.astype(np.uint8), cv2.COLORMAP_JET)
            mask_preview_color = cv2.resize(mask_preview_color, (160, 120))
            telemetry_combo = np.hstack((raw_preview, mask_preview_color))
            
            # Send Telemetry
            if self.telemetry_img_pub.get_num_connections() > 0:
                msg_out = CompressedImage()
                msg_out.header.stamp = rospy.Time.now()
                msg_out.format = "jpeg"
                msg_out.data = np.array(cv2.imencode('.jpg', telemetry_combo, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).tobytes()
                self.telemetry_img_pub.publish(msg_out)

            # PilotNet Inference
            out1, out2 = 0.0, 0.0
            if self.current_intent != "stop":
                # INTENT_MAP natively returns a one-hot list like [1.0, 0.0, 0.0, 0.0]
                intent_list = INTENT_MAP.get(self.current_intent, [1.0, 0.0, 0.0, 0.0])

                if self.backend.lower() == "tensorrt":
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

            # Publish Motors
            if self.output_mode == "twist":
                cmd_msg = Twist2DStamped()
                cmd_msg.header.stamp = rospy.Time.now()
                cmd_msg.v = float(out1)
                cmd_msg.omega = float(out2)
                self.cmd_pub.publish(cmd_msg)
            else:
                cmd_msg = WheelsCmdStamped()
                cmd_msg.header.stamp = rospy.Time.now()
                cmd_msg.vel_left = float(out1)
                cmd_msg.vel_right = float(out2)
                self.cmd_pub.publish(cmd_msg)

            # State Telemetry
            state_data = {
                "out1": float(out1),
                "out2": float(out2),
                "intent": self.current_intent,
                "backend": self.backend
            }
            self.telemetry_state_pub.publish(String(json.dumps(state_data)))

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
