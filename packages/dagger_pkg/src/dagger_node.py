#!/usr/bin/env python3
import os
import csv
import time
import rospy
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import String

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import crop_image, get_eval_transforms, INTENT_MAP

class DAggerNode:
    def __init__(self):
        rospy.init_node('dagger_node', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'default_robot')
        
        # --- DAgger Setup ---
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.data_dir = f"/dataset/dagger_aggregate_{run_id}"
        self.img_dir = os.path.join(self.data_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.data_dir, "dagger_log.csv")
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "image_filename", "vel_left", "vel_right", "intent", "w", "a", "s", "d"])

        rospy.loginfo(f"[DAgger] Initialized. Saving corrections to: {self.data_dir}")

        # --- AI Model Setup ---
        self.model_path = "/models/pilotnet/best_model.onnx"
        self.ort_session = ort.InferenceSession(self.model_path)
        self.transform = get_eval_transforms()

        # --- State Variables ---
        self.current_intent = "straight"
        self.human_keys = [0, 0, 0, 0] # W, A, S, D
        self.is_human_intervening = False

        # Tuning params for mapping keys to wheel speeds
        self.v_fwd = 0.5
        self.omega_turn = 5.0

        # --- Pubs & Subs ---
        self.cmd_pub = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1, tcp_nodelay=True)
        
        # Listen to UI node for intent and interventions
        rospy.Subscriber(f"/{self.veh}/data_collector/intent", String, self.intent_cb)
        rospy.Subscriber(f"/{self.veh}/data_collector/keys", String, self.keys_cb)
        
        # Listen to camera
        rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.image_cb, queue_size=1, buff_size=2**24, tcp_nodelay=True)

    def intent_cb(self, msg):
        self.current_intent = msg.data

    def keys_cb(self, msg):
        try:
            self.human_keys = [int(x) for x in msg.data.split(',')]
            # If any key is pressed, human is intervening
            self.is_human_intervening = sum(self.human_keys) > 0
        except Exception as e:
            rospy.logerr(f"[DAgger] Error parsing keys: {e}. Received data: {msg.data}")

    def calculate_human_commands(self):
        v = 0.0
        omega = 0.0
        if self.human_keys[0]: v += self.v_fwd           # W
        if self.human_keys[2]: v -= self.v_fwd           # S
        if self.human_keys[1]: omega += self.omega_turn  # A
        if self.human_keys[3]: omega -= self.omega_turn  # D
        return v, omega

    def image_cb(self, msg):
        timestamp = msg.header.stamp.to_sec()

        # Decode image
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        cropped_img = crop_image(pil_image)

        # Control routing
        if self.is_human_intervening:
            # --- DAgger INTERVENTION LOGIC ---
            v, omega = self.calculate_human_commands()
            
            # Save frame
            img_filename = f"{timestamp:.4f}.jpg"
            img_filepath = os.path.join(self.img_dir, img_filename)
            cv2.imwrite(img_filepath, cv_image)

            # Log correction (CSV headers need to match v and omega if retraining)
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [timestamp, img_filename, v, omega, self.current_intent] + self.human_keys
                writer.writerow(row)
            
            rospy.loginfo_throttle(1.0, "[DAgger] HUMAN OVERRIDE - Logging Correction!")

        else:
            # --- AUTONOMOUS LOGIC ---
            img_tensor = self.transform(cropped_img).unsqueeze(0).numpy()
            intent_tensor = np.array([INTENT_MAP.get(self.current_intent, [1.0, 0.0, 0.0, 0.0])], dtype=np.float32)

            ort_inputs = {
                self.ort_session.get_inputs()[0].name: img_tensor,
                self.ort_session.get_inputs()[1].name: intent_tensor
            }
            ort_outs = self.ort_session.run(None, ort_inputs)
            # Interpret network outputs directly as high-level commands
            v, omega = ort_outs[0][0][0], ort_outs[0][0][1]

        # Publish high-level Twist command to utilize Duckiebot calibration
        cmd_msg = Twist2DStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.v = float(v)
        cmd_msg.omega = float(omega)
        self.cmd_pub.publish(cmd_msg)

if __name__ == '__main__':
    node = DAggerNode()
    rospy.spin()