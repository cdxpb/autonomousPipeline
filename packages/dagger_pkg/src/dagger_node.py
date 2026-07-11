#!/usr/bin/env python3
import os
import sys
import csv
import time
import argparse
import rospy
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
try:
    from ultralytics import YOLO
except ImportError:
    pass

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import String, Bool

# Import shared model/preprocessing code from cond_imitation_learning_pkg -- the canonical,
# actively-maintained copy -- rather than any local files in this package. A local fork of
# pilotnet_regNheadv2.py here previously drifted out of sync (missing the SafePool fix for
# MPS's broken AdaptiveAvgPool2d), and there's no local pilotnet_FiLM.py for approach 7 at all.
_CIL_PKG_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cond_imitation_learning_pkg", "src"))
sys.path.insert(0, _CIL_PKG_SRC)
from utils import crop_image, INTENT_MAP

# Same relative-path convention as autonomous_node.py (works from the native Mac workspace,
# relies on APFS being case-insensitive -- "autonomouspipeline" vs "autonomousPipeline").
MODEL_BASE = "../autonomouspipeline/models"
YOLO_BASENAME = "yolo_model"  # approach 5/7 both use the "full" YOLO, same as headless's model_set="full"
PILOTNET_BASENAME = {
    5: "best_model_regNheadv2",
    7: "best_model_approach7",
}

class DAggerNode:
    def __init__(self, approach=5, data_dir=None, fwd_speed=0.3, turn_bias=0.15,
                 yolo_backend="onnx", pilotnet_backend="onnx"):
        rospy.init_node('dagger_node', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'default_robot')

        if approach not in PILOTNET_BASENAME:
            raise ValueError(f"DAgger native mode only supports approach 5 or 7, got {approach}")
        self.approach = approach
        self.yolo_backend = yolo_backend.lower()
        self.pilotnet_backend = pilotnet_backend.lower()
        if (self.yolo_backend == "onnx" or self.pilotnet_backend == "onnx") and not ORT_AVAILABLE:
            raise RuntimeError("onnx backend requested but onnxruntime isn't installed in this env")

        # --- DAgger Setup ---
        run_id = time.strftime("%Y%m%d-%H%M%S")
        base_dir = data_dir or os.path.expanduser("~/Desktop/my_dataset")
        self.data_dir = os.path.join(base_dir, f"dagger_aggregate_{run_id}")
        self.img_dir = os.path.join(self.data_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)

        self.csv_path = os.path.join(self.data_dir, "dagger_log.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "image_filename", "vel_left", "vel_right", "intent", "w", "a", "s", "d"])

        rospy.loginfo(f"[DAgger] approach={self.approach} | Saving corrections to: {self.data_dir}")

        # --- AI Model Setup (approach 5/7 only, matches autonomous_node.py's model choice) ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # -- PilotNet --
        self.pt_model = None
        self.ort_session = None
        pilotnet_basename = PILOTNET_BASENAME[self.approach]
        if self.pilotnet_backend == "onnx":
            onnx_path = f"{MODEL_BASE}/pilotnet/{pilotnet_basename}.onnx"
            rospy.loginfo(f"[DAgger] Loading PilotNet ONNX from {onnx_path}")
            self.ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        else:
            pt_path = f"{MODEL_BASE}/pilotnet/{pilotnet_basename}.pt"
            rospy.loginfo(f"[DAgger] Loading PilotNet PyTorch from {pt_path}")
            if self.approach == 7:
                from pilotnet_FiLM import ConditionalPilotNetFiLM
                self.pt_model = ConditionalPilotNetFiLM().to(self.device)
            else:
                from pilotnet_regNheadv2 import ConditionalPilotNet
                self.pt_model = ConditionalPilotNet().to(self.device)
            self.pt_model.load_state_dict(torch.load(pt_path, map_location=self.device))
            self.pt_model.eval()

        # -- YOLO segmentation -- ultralytics' YOLO() loads .pt/.onnx interchangeably behind
        # the same call interface, so switching backend is just a matter of which file we hand it.
        yolo_ext = "onnx" if self.yolo_backend == "onnx" else "pt"
        yolo_path = f"{MODEL_BASE}/yolo_model/{YOLO_BASENAME}.{yolo_ext}"
        rospy.loginfo(f"[DAgger] Loading YOLO ({self.yolo_backend}) from {yolo_path} ...")
        self.yolo_session = YOLO(yolo_path, task='semantic')

        # --- State Variables ---
        self.current_intent = "straight"
        self.human_keys = [0, 0, 0, 0]  # W, A, S, D
        self.is_human_intervening = False
        self.is_recording = False
        self.is_model_running = False

        # WASD -> vel_left/vel_right directly, same [-1, 1] scale PilotNet outputs -- NOT v/omega.
        self.fwd_speed = fwd_speed
        self.turn_bias = turn_bias

        # --- Pubs & Subs ---
        # Publish straight to wheels_driver_node, matching approach 5/7's "wheels" output_mode
        # used everywhere else in this codebase (not car_cmd_switch_node/twist). This also means
        # what we log below is exactly what we commanded, keeping dagger_log.csv's vel_left/
        # vel_right consistent with the base dataset's log.csv (which reads back the real
        # wheels_cmd topic -- see data_collector_pkg/logger_node.py).
        self.cmd_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1, tcp_nodelay=True)

        rospy.Subscriber(f"/{self.veh}/data_collector/intent", String, self.intent_cb)
        rospy.Subscriber(f"/{self.veh}/data_collector/keys", String, self.keys_cb)
        rospy.Subscriber(f"/{self.veh}/data_collector/is_recording", Bool, self.recording_cb)
        rospy.Subscriber(f"/{self.veh}/data_collector/is_model_running", Bool, self.model_running_cb)

        self.image_sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.image_cb, queue_size=1, buff_size=2**24, tcp_nodelay=True)

    def intent_cb(self, msg):
        self.current_intent = msg.data

    def keys_cb(self, msg):
        try:
            self.human_keys = [int(x) for x in msg.data.split(',')]
            self.is_human_intervening = sum(self.human_keys) > 0
        except Exception as e:
            rospy.logerr(f"[DAgger] Error parsing keys: {e}. Received data: {msg.data}")

    def recording_cb(self, msg):
        self.is_recording = msg.data

    def model_running_cb(self, msg):
        self.is_model_running = msg.data

    def calculate_human_commands(self):
        vel_left, vel_right = 0.0, 0.0
        if self.human_keys[0]:  # W
            vel_left += self.fwd_speed
            vel_right += self.fwd_speed
        if self.human_keys[2]:  # S
            vel_left -= self.fwd_speed
            vel_right -= self.fwd_speed
        if self.human_keys[1]:  # A: turn left
            vel_left -= self.turn_bias
            vel_right += self.turn_bias
        if self.human_keys[3]:  # D: turn right
            vel_left += self.turn_bias
            vel_right -= self.turn_bias
        return max(-1.0, min(1.0, vel_left)), max(-1.0, min(1.0, vel_right))

    def image_cb(self, msg):
        timestamp = msg.header.stamp.to_sec()

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None or cv_image.size == 0:
            return
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        cropped_img = crop_image(pil_image)

        vel_left, vel_right = 0.0, 0.0

        if self.is_human_intervening:
            # --- DAgger INTERVENTION LOGIC ---
            vel_left, vel_right = self.calculate_human_commands()
        elif self.is_model_running:
            # --- AUTONOMOUS LOGIC ---
            # device= only means something for the pytorch backend; onnx runs on whatever
            # provider the session was built with (CPUExecutionProvider here).
            yolo_kwargs = {"device": self.device.type} if self.yolo_backend == "pytorch" else {}
            with torch.no_grad():
                yolo_results = self.yolo_session(cropped_img, verbose=False, **yolo_kwargs)

            mask = yolo_results[0].semantic_mask.data.cpu()
            mask_pil = Image.fromarray(mask.numpy())

            # Mask is already cropped because YOLO input was cropped
            resized_mask_pil = mask_pil.resize((224, 112), Image.NEAREST)
            intent_list = INTENT_MAP.get(self.current_intent, [1.0, 0.0, 0.0, 0.0])

            if self.pilotnet_backend == "onnx":
                img_np = TF.to_tensor(resized_mask_pil).unsqueeze(0).numpy().astype(np.float32)
                intent_np = np.array([intent_list], dtype=np.float32)
                ort_inputs = {"image_input": img_np, "intent_input": intent_np}
                output = self.ort_session.run(None, ort_inputs)[0]
                vel_left = max(-1.0, min(1.0, float(output[0][0])))
                vel_right = max(-1.0, min(1.0, float(output[0][1])))
            else:
                img_tensor = TF.to_tensor(resized_mask_pil).unsqueeze(0).to(self.device)
                intent_tensor = torch.tensor([intent_list], dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    output = self.pt_model(img_tensor, intent_tensor)
                vel_left = max(-1.0, min(1.0, output[0][0].item()))
                vel_right = max(-1.0, min(1.0, output[0][1].item()))

        # Record if requested and human is intervening (that's the whole point of DAgger:
        # only the human's corrections get aggregated back into the training set)
        if self.is_recording and self.is_human_intervening:
            img_filename = f"{timestamp:.4f}.jpg"
            img_filepath = os.path.join(self.img_dir, img_filename)
            cv2.imwrite(img_filepath, cv_image)

            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [timestamp, img_filename, vel_left, vel_right, self.current_intent] + self.human_keys
                writer.writerow(row)

            rospy.loginfo_throttle(1.0, "[DAgger] RECORDING correction")

        cmd_msg = WheelsCmdStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.vel_left = float(vel_left)
        cmd_msg.vel_right = float(vel_right)
        self.cmd_pub.publish(cmd_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", type=int, choices=[5, 7], default=5)
    parser.add_argument("--data-dir", type=str, default=None, help="Base dir for dagger_aggregate_* runs (default: ~/Desktop/my_dataset)")
    parser.add_argument("--fwd-speed", type=float, default=0.3)
    parser.add_argument("--turn-bias", type=float, default=0.15)
    parser.add_argument("--backend", type=str, choices=["pytorch", "onnx"], default="onnx",
                         help="Default backend for both YOLO and PilotNet")
    parser.add_argument("--yolo-backend", type=str, choices=["pytorch", "onnx"], default=None,
                         help="Overrides --backend for YOLO only")
    parser.add_argument("--pilotnet-backend", type=str, choices=["pytorch", "onnx"], default=None,
                         help="Overrides --backend for PilotNet only")
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    node = DAggerNode(
        approach=args.approach,
        data_dir=args.data_dir,
        fwd_speed=args.fwd_speed,
        turn_bias=args.turn_bias,
        yolo_backend=args.yolo_backend or args.backend,
        pilotnet_backend=args.pilotnet_backend or args.backend,
    )
    rospy.spin()
