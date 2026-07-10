#!/usr/bin/env python3
"""
Plain CIL dashboard. Subscribes to headless_autonomous_node.py's telemetry and shows the
live/mask preview, motor output, and manual intent override. No VLM planner here, that's
vlm_dashboard_node.py.
"""
import sys
import os
import json
import rospy
import cv2
import numpy as np

# PyQt5
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QGroupBox, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MacDashboardUI(QMainWindow):
    # Signals to safely update UI from ROS thread
    telemetry_img_signal = pyqtSignal(np.ndarray, np.ndarray)
    telemetry_state_signal = pyqtSignal(float, float, str, str)

    def __init__(self):
        super().__init__()
        rospy.init_node('mac_dashboard_ui', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'golduck')
        self.current_intent = "straight"

        self.telemetry_img_signal.connect(self.update_images)
        self.telemetry_state_signal.connect(self.update_state)

        self.init_ui()

        # Telemetry Subscribers
        self.img_sub = rospy.Subscriber(f"/{self.veh}/telemetry/preview/compressed", CompressedImage, self.img_cb, queue_size=1)
        self.state_sub = rospy.Subscriber(f"/{self.veh}/telemetry/state", String, self.state_cb, queue_size=1)
        self.intent_pub = rospy.Publisher(f"/{self.veh}/data_collector/intent", String, queue_size=1)

    def init_ui(self):
        self.setWindowTitle(f"Mac Dashboard (Connected to {self.veh})")
        self.setGeometry(100, 100, 900, 700)

        main_widget = QWidget(self)
        layout = QVBoxLayout()

        # --- CAMERA DISPLAYS ---
        img_layout = QHBoxLayout()
        self.live_label = QLabel(self)
        self.live_label.setAlignment(Qt.AlignCenter)
        self.live_label.setText("Waiting for Telemetry...")
        self.live_label.setFixedSize(400, 300)
        self.live_label.setStyleSheet("background-color: #121212; color: #aaaaaa; border: 2px solid #333;")

        self.model_label = QLabel(self)
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setText("Waiting for Segmentation...")
        self.model_label.setFixedSize(400, 300)
        self.model_label.setStyleSheet("background-color: #121212; color: #aaaaaa; border: 2px solid #333;")

        img_layout.addWidget(self.live_label)
        img_layout.addWidget(self.model_label)
        layout.addLayout(img_layout)

        # --- TELEMETRY GRAPHICS ---
        telemetry_group = QGroupBox("Network Predictions & State (From Jetson)")
        telemetry_layout = QHBoxLayout()

        self.motor_text = QLabel("Left Motor: 0.00  |  Right Motor: 0.00 (IDLE)")
        self.motor_text.setStyleSheet("font-family: monospace; font-size: 18px; font-weight: bold; color: #2b2b2b;")
        self.motor_text.setAlignment(Qt.AlignCenter)
        telemetry_layout.addWidget(self.motor_text)

        telemetry_group.setLayout(telemetry_layout)
        layout.addWidget(telemetry_group)

        # --- MANUAL INTENT OVERRIDE ---
        intent_group = QGroupBox("Manual Intent Override (Sends to Jetson)")
        intent_layout = QHBoxLayout()

        self.btn_straight = QPushButton("Straight [I]")
        self.btn_left = QPushButton("Left [J]")
        self.btn_right = QPushButton("Right [L]")
        self.btn_lane_following = QPushButton("Lane Follow [U]")
        self.btn_stop_intent = QPushButton("Stop [K]")

        self.btn_straight.clicked.connect(lambda: self.publish_intent("straight"))
        self.btn_left.clicked.connect(lambda: self.publish_intent("left"))
        self.btn_right.clicked.connect(lambda: self.publish_intent("right"))
        self.btn_lane_following.clicked.connect(lambda: self.publish_intent("lane_following"))
        self.btn_stop_intent.clicked.connect(lambda: self.publish_intent("stop"))

        intent_layout.addWidget(self.btn_straight)
        intent_layout.addWidget(self.btn_left)
        intent_layout.addWidget(self.btn_right)
        intent_layout.addWidget(self.btn_lane_following)
        intent_layout.addWidget(self.btn_stop_intent)
        intent_group.setLayout(intent_layout)
        layout.addWidget(intent_group)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.update_intent_button_styles()
        self.setFocusPolicy(Qt.StrongFocus)

    def publish_intent(self, intent_str):
        self.current_intent = intent_str
        self.intent_pub.publish(String(intent_str))
        self.update_intent_button_styles()

    def update_intent_button_styles(self):
        buttons = {"straight": self.btn_straight, "left": self.btn_left,
                   "right": self.btn_right, "stop": self.btn_stop_intent, "lane_following": self.btn_lane_following}
        for name, btn in buttons.items():
            if name == self.current_intent:
                color = "#dc3545" if name == "stop" else "#007bff"
                btn.setStyleSheet(f"background-color: {color}; color: white; font-weight: bold; padding: 8px;")
            else:
                btn.setStyleSheet("background-color: #f8f9fa; color: black; padding: 8px;")

    def img_cb(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None: return

            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # The telemetry image is a side-by-side (120, 320, 3)
            # Left half is raw preview, right half is mask preview
            h, w, c = rgb_image.shape
            mid = w // 2
            raw_preview = rgb_image[:, :mid]
            mask_preview = rgb_image[:, mid:]

            self.telemetry_img_signal.emit(raw_preview, mask_preview)
            self.on_raw_frame(raw_preview)

        except Exception as e:
            rospy.logerr(f"Dashboard Telemetry Image Error: {e}")

    def on_raw_frame(self, raw_preview):
        """Hook for subclasses (e.g. the VLM dashboard) that want the raw preview
        frame as it arrives, without duplicating img_cb's decode/split logic."""
        pass

    def state_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.telemetry_state_signal.emit(data["out1"], data["out2"], data["intent"], data["backend"])
        except Exception as e:
            rospy.logerr(f"Dashboard State Error: {e}")

    def update_images(self, raw_img, model_img):
        h, w, ch = raw_img.shape
        qt_raw = QImage(raw_img.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self.live_label.setPixmap(QPixmap.fromImage(qt_raw).scaled(self.live_label.width(), self.live_label.height(), Qt.KeepAspectRatio))

        h, w, ch = model_img.shape
        qt_model = QImage(model_img.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self.model_label.setPixmap(QPixmap.fromImage(qt_model).scaled(self.model_label.width(), self.model_label.height(), Qt.KeepAspectRatio))

    def update_state(self, out1, out2, intent, backend):
        self.motor_text.setText(f"L: {out1:+.3f} | R: {out2:+.3f} | Backend: {backend}")
        if self.current_intent != intent:
            self.current_intent = intent
            self.update_intent_button_styles()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_I: self.publish_intent("straight")
        elif event.key() == Qt.Key_J: self.publish_intent("left")
        elif event.key() == Qt.Key_L: self.publish_intent("right")
        elif event.key() == Qt.Key_K: self.publish_intent("stop")
        elif event.key() == Qt.Key_U: self.publish_intent("lane_following")

    def mousePressEvent(self, event):
        self.setFocus()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    driver_ui = MacDashboardUI()
    driver_ui.show()
    sys.exit(app.exec_())
