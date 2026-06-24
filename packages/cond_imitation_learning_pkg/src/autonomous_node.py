#!/usr/bin/env python3
import sys
import os
import rospy
import cv2
import numpy as np
import argparse
import torch
import torchvision.transforms.functional as TF
try:
    from ultralytics import YOLO
except ImportError:
    pass
from PIL import Image

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import String

# DRY Imports from our package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import crop_image, get_eval_transforms, INTENT_MAP, CROP_TOP_ROWS

# ---------------------------------------------------------
# DUAL-BACKEND DETECTION (TensorRT vs ONNX Runtime)
# ---------------------------------------------------------
"""
# COMMENTED OUT AUTOMATIC GPU DETECTION FOR NOW
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    USE_TENSORRT = True
    rospy.loginfo("NVIDIA GPU DETECTED! Using TensorRT.")
except ImportError:
    import onnxruntime as ort
    USE_TENSORRT = False
    rospy.loginfo("No NVIDIA GPU. Falling back to ONNX Runtime (CPU).")
"""

# HARDCODE ONNX FOR TESTING
import onnxruntime as ort
USE_TENSORRT = False
rospy.logwarn("MANUAL OVERRIDE: Forcing ONNX Runtime on Physical Robot!")

class AutonomousDriver:
    def __init__(self, skip_segmentation=False):
        rospy.init_node('autonomous_driver_node', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'default_robot')
        self.skip_segmentation = skip_segmentation

        # Load Models
        if self.skip_segmentation:
            self.model_path = "/models/pilotnet/best_model" # Omit extension, logic decides
        else:
            self.model_path = "/models/pilotnet/segPilot"
            self.yolo_path = "models/yolo_model/yolo_model.onnx"
            
        self.setup_inference_engine()

        self.transform = get_eval_transforms()
        self.current_intent = "straight"

        # Publishers & Subscribers
        self.cmd_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1, tcp_nodelay=True)
        self.image_sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.image_cb, queue_size=1, buff_size=2**24, tcp_nodelay=True)
        self.intent_sub = rospy.Subscriber(f"/{self.veh}/data_collector/intent", String, self.intent_cb)

    def setup_inference_engine(self):
        if not self.skip_segmentation:
            rospy.loginfo("Loading YOLO Semantic Segmentation model...")
            self.yolo_session = YOLO(self.yolo_path, task="semantic")

        rospy.loginfo(f"Loading PilotNet model from {self.model_path}...")
        if USE_TENSORRT:
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(f"{self.model_path}.engine", "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            
            # Allocate memory for inputs/outputs
            channels = 3 if self.skip_segmentation else 1
            self.d_img_in = cuda.mem_alloc(1 * channels * 112 * 224 * 4) # Float32 size
            self.d_intent_in = cuda.mem_alloc(1 * 4 * 4) 
            self.d_output = cuda.mem_alloc(1 * 2 * 4)
            self.bindings = [int(self.d_img_in), int(self.d_intent_in), int(self.d_output)]
        else:
            self.ort_session = ort.InferenceSession(f"{self.model_path}.onnx")

    def intent_cb(self, msg):
        self.current_intent = msg.data

    def image_cb(self, msg):
        try:
            # Decode image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if self.skip_segmentation:
                # Original Pipeline
                pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                cropped_img = crop_image(pil_image)
                img_tensor = self.transform(cropped_img).unsqueeze(0).numpy() # Shape: (1, 3, 112, 224)
            else:
                # YOLO Segmentation Pipeline
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                yolo_input = rgb_image.astype(np.float32) / 255.0
                yolo_input = np.transpose(yolo_input, (2, 0, 1)) # (3, 480, 640)
                yolo_input = np.expand_dims(yolo_input, axis=0)  # (1, 3, 480, 640)

                mask = self.yolo_session(yolo_input)[0].semantic_mask.data # (480, 640)
                mask = mask.unsqueeze(0) # (1, 480, 640)

                _, height, width = mask.shape
                cropped_img = TF.crop(mask, top=CROP_TOP_ROWS, left=0, height=height - CROP_TOP_ROWS, width=width)
                resized_mask = TF.resize(cropped_img, (112, 224))

                img_tensor = resized_mask.to(torch.float32).unsqueeze(0).cpu().numpy() # (1, 1, 112, 224)

            # Prep Intent
            intent_tensor = np.array([INTENT_MAP.get(self.current_intent, [1.0, 0.0, 0.0, 0.0])], dtype=np.float32) # Shape: (1, 4)

            # Inference
            if USE_TENSORRT:
                cuda.memcpy_htod(self.d_img_in, img_tensor)
                cuda.memcpy_htod(self.d_intent_in, intent_tensor)
                self.context.execute_v2(bindings=self.bindings)
                h_output = np.empty((1, 2), dtype=np.float32)
                cuda.memcpy_dtoh(h_output, self.d_output)
                vel_left, vel_right = h_output[0][0], h_output[0][1]
            else:
                ort_inputs = {
                    self.ort_session.get_inputs()[0].name: img_tensor,
                    self.ort_session.get_inputs()[1].name: intent_tensor
                }
                ort_outs = self.ort_session.run(None, ort_inputs)
                vel_left, vel_right = ort_outs[0][0][0], ort_outs[0][0][1]

            # Publish
            cmd_msg = WheelsCmdStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.vel_left = float(vel_left)
            cmd_msg.vel_right = float(vel_right)
            self.cmd_pub.publish(cmd_msg)

        except Exception as e:
            rospy.logerr(f"Inference crash: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autonomous Driver Node")
    parser.add_argument("--skip_segmentation", action="store_true", help="Skip YOLO segmentation and use original PilotNet")
    args, unknown = parser.parse_known_args(rospy.myargv()[1:])
    
    node = AutonomousDriver(skip_segmentation=args.skip_segmentation)
    rospy.spin()