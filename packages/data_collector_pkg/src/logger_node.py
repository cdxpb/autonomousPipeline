#!/usr/bin/env python3
import os
import csv
import time
import rospy
import cv2
import numpy as np
import message_filters

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Bool, String

class DataLoggerNode:
    def __init__(self):
        rospy.init_node('data_logger_node', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'default_robot')
        
        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.data_dir = f"/dataset/cil_dataset_{run_id}"
        self.img_dir = os.path.join(self.data_dir, "images")
        
        os.makedirs(self.img_dir, exist_ok=True)
        self.csv_path = os.path.join(self.data_dir, "log.csv")
        
        # Added w, a, s, d to the CSV headers
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "image_filename", "vel_left", "vel_right", "intent", "w", "a", "s", "d"])

        rospy.loginfo(f"Logger initialized. Saving data to: {self.data_dir}")

        # State Variables
        self.is_recording = False
        self.current_intent = "straight"
        self.current_keys = [0, 0, 0, 0] # Default state for W,A,S,D

        # Standard Subscribers for Headerless state tracking
        rospy.Subscriber(f"/{self.veh}/data_collector/is_recording", Bool, self.recording_cb)
        rospy.Subscriber(f"/{self.veh}/data_collector/intent", String, self.intent_cb)
        rospy.Subscriber(f"/{self.veh}/data_collector/keys", String, self.keys_cb)

        # Synchronized Subscribers for Image + Wheel telemetry
        img_sub = message_filters.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage)
        wheel_sub = message_filters.Subscriber(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped)

        self.ts = message_filters.ApproximateTimeSynchronizer([img_sub, wheel_sub], queue_size=50, slop=0.5)
        self.ts.registerCallback(self.synced_callback)

    def recording_cb(self, msg):
        self.is_recording = msg.data

    def intent_cb(self, msg):
        self.current_intent = msg.data

    def keys_cb(self, msg):
        try:
            # Safely converts the incoming "1,0,0,0" string into a list of integers
            self.current_keys = [int(x) for x in msg.data.split(',')]
        except Exception as e:
            pass

    def synced_callback(self, img_msg, wheel_msg):
        if not self.is_recording:
            return

        timestamp = img_msg.header.stamp.to_sec()
        
        # Save image
        img_filename = f"{timestamp:.4f}.jpg"
        img_filepath = os.path.join(self.img_dir, img_filename)
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(img_filepath, cv_image)

        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Unpack the list directly into the row using list concatenation
            row = [timestamp, img_filename, wheel_msg.vel_left, wheel_msg.vel_right, self.current_intent] + self.current_keys
            writer.writerow(row)
            
        rospy.loginfo(f"SUCCESS: Saved synchronized frame {img_filename}")

if __name__ == '__main__':
    node = DataLoggerNode()
    rospy.spin()