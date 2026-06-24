#!/usr/bin/env python3
import sys
import os
import rospy
import cv2
import numpy as np
from PIL import Image

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import String

# DRY Imports from our package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import crop_image, get_eval_transforms, INTENT_MAP

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
    def __init__(self):
        rospy.init_node('autonomous_driver_node', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'default_robot')

        # Load Model
        self.model_path = "/models/pilotnet/best_model" # Omit extension, logic decides
        self.setup_inference_engine()

        self.transform = get_eval_transforms()
        self.current_intent = "straight"

        # Publishers & Subscribers
        self.cmd_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1, tcp_nodelay=True)
        self.image_sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.image_cb, queue_size=1, buff_size=2**24, tcp_nodelay=True)
        self.intent_sub = rospy.Subscriber(f"/{self.veh}/data_collector/intent", String, self.intent_cb)

    def setup_inference_engine(self):
        if USE_TENSORRT:
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(f"{self.model_path}.engine", "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            
            # Allocate memory for inputs/outputs
            self.d_img_in = cuda.mem_alloc(1 * 3 * 112 * 224 * 4) # Float32 size
            self.d_intent_in = cuda.mem_alloc(1 * 4 * 4) 
            self.d_output = cuda.mem_alloc(1 * 2 * 4)
            self.bindings = [int(self.d_img_in), int(self.d_intent_in), int(self.d_output)]
        else:
            self.ort_session = ort.InferenceSession(f"{self.model_path}.onnx")

    def intent_cb(self, msg):
        self.current_intent = msg.data

    def image_cb(self, msg):
        try:
            # Decode & Crop
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            cropped_img = crop_image(pil_image)

            # Prep Inputs
            img_tensor = self.transform(cropped_img).unsqueeze(0).numpy() # Shape: (1, 3, 112, 224)
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
    node = AutonomousDriver()
    rospy.spin()