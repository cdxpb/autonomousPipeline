#!/usr/bin/env python3
import sys
import os
import json
import rospy
import cv2
import numpy as np

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QDoubleSpinBox, QFormLayout, QGroupBox, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String
from duckietown_msgs.msg import Twist2DStamped, WheelsCmdStamped

class ROSBridge(QObject):
    new_image_signal = pyqtSignal(np.ndarray)

class DataCollectorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        rospy.init_node('data_collector_ui', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'default_robot')
        
        # Load Tuning Config
        self.config_path = "/dataset/tuning_config.json"
        self.tuning = self.load_tuning_config()

        # --- LIVE DRIVING VARIABLES ---
        self.live_v_fwd = self.tuning["v_fwd"]
        self.live_v_rev = self.tuning["v_rev"]
        self.live_v_bump_a = self.tuning["v_bump_a"]
        self.live_omega_a = self.tuning["omega_a"]
        self.live_v_bump_d = self.tuning["v_bump_d"]
        self.live_omega_d = self.tuning["omega_d"]

        # Publishers & Subscribers
        self.cmd_pub = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1, tcp_nodelay=True)
        self.rec_pub = rospy.Publisher(f"/{self.veh}/data_collector/is_recording", Bool, queue_size=1)
        self.intent_pub = rospy.Publisher(f"/{self.veh}/data_collector/intent", String, queue_size=1)
        self.keys_pub = rospy.Publisher(f"/{self.veh}/data_collector/keys", String, queue_size=1)
        
        self.image_sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.image_cb, queue_size=1, tcp_nodelay=True)
        self.wheels_sub = rospy.Subscriber(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, self.wheels_cb)
        
        self.bridge = ROSBridge()
        self.bridge.new_image_signal.connect(self.update_image_display)

        # State Variables
        self.is_recording = False
        self.current_intent = "straight"
        self.current_vel_left = 0.0
        self.current_vel_right = 0.0
        self.keys_pressed = set()

        self.init_ui()

        # Control Loop (10 Hz)
        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.publish_commands)
        self.control_timer.start(100)

    def load_tuning_config(self):
        default_config = {
            "v_fwd": 0.5, "v_rev": -0.5, 
            "v_bump_a": 0.1, "omega_a": 5.0,
            "v_bump_d": 0.1, "omega_d": 5.0
        }
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    # Update defaults with saved values in case new keys were added
                    saved = json.load(f)
                    default_config.update(saved)
        except Exception:
            pass
        return default_config

    def apply_and_save_tuning(self):
        # Hot-swap live variables
        self.live_v_fwd = self.w_spin.value()
        self.live_v_rev = self.s_spin.value()
        self.live_v_bump_a = self.a_v_spin.value()
        self.live_omega_a = self.a_omega_spin.value()
        self.live_v_bump_d = self.d_v_spin.value()
        self.live_omega_d = self.d_omega_spin.value()
        
        # Update Dictionary
        self.tuning = {
            "v_fwd": self.live_v_fwd,
            "v_rev": self.live_v_rev,
            "v_bump_a": self.live_v_bump_a,
            "omega_a": self.live_omega_a,
            "v_bump_d": self.live_v_bump_d,
            "omega_d": self.live_omega_d
        }
        
        # Save to Disk
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.tuning, f, indent=4)
        except Exception as e:
            rospy.logerr(f"Failed to save config: {e}")

        self.status_label.setText(f"Recording: {'ON' if self.is_recording else 'OFF'} | Intent: {self.current_intent} | (TUNING APPLIED!)")
        self.setFocus()

    def init_ui(self):
        self.setWindowTitle("CIL Data Collector & Tuner")
        self.setGeometry(100, 100, 800, 750)

        main_widget = QWidget(self)
        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Waiting for Camera Feed...")
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.image_label)

        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.motor_label = QLabel(self)
        self.motor_label.setAlignment(Qt.AlignCenter)
        self.motor_label.setStyleSheet("color: black; font-family: monospace; font-size: 14px;")
        layout.addWidget(self.motor_label)

        # --- TUNING DASHBOARD ---
        tuning_group = QGroupBox("Live Tuning Parameters")
        tuning_layout = QHBoxLayout()

        # W/S (Straight) Column
        form_w_s = QFormLayout()
        self.w_spin = QDoubleSpinBox()
        self.w_spin.setRange(0.0, 1.0); self.w_spin.setSingleStep(0.05); self.w_spin.setValue(self.live_v_fwd)
        
        self.s_spin = QDoubleSpinBox()
        self.s_spin.setRange(-1.0, 0.0); self.s_spin.setSingleStep(0.05); self.s_spin.setValue(self.live_v_rev)
        
        form_w_s.addRow("W (Fwd V):", self.w_spin)
        form_w_s.addRow("S (Rev V):", self.s_spin)
        tuning_layout.addLayout(form_w_s)

        # A (Left Turn) Column
        form_a = QFormLayout()
        self.a_v_spin = QDoubleSpinBox()
        self.a_v_spin.setRange(-1.0, 1.0); self.a_v_spin.setSingleStep(0.05); self.a_v_spin.setValue(self.live_v_bump_a)
        
        self.a_omega_spin = QDoubleSpinBox()
        self.a_omega_spin.setRange(0.0, 15.0); self.a_omega_spin.setSingleStep(0.5); self.a_omega_spin.setValue(self.live_omega_a)
        
        form_a.addRow("A (Fwd V):", self.a_v_spin)
        form_a.addRow("A (+Omega):", self.a_omega_spin)
        tuning_layout.addLayout(form_a)

        # D (Right Turn) Column
        form_d = QFormLayout()
        self.d_v_spin = QDoubleSpinBox()
        self.d_v_spin.setRange(-1.0, 1.0); self.d_v_spin.setSingleStep(0.05); self.d_v_spin.setValue(self.live_v_bump_d)
        
        self.d_omega_spin = QDoubleSpinBox()
        # Note: We keep this positive in the UI, and subtract it in the logic block
        self.d_omega_spin.setRange(0.0, 15.0); self.d_omega_spin.setSingleStep(0.5); self.d_omega_spin.setValue(self.live_omega_d)
        
        form_d.addRow("D (Fwd V):", self.d_v_spin)
        form_d.addRow("D (-Omega):", self.d_omega_spin)
        tuning_layout.addLayout(form_d)

        # Apply Button
        self.apply_btn = QPushButton("Apply\n&\nSave")
        self.apply_btn.clicked.connect(self.apply_and_save_tuning)
        self.apply_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px;")
        tuning_layout.addWidget(self.apply_btn)

        tuning_group.setLayout(tuning_layout)
        layout.addWidget(tuning_group)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.update_status_ui()
        
        self.setFocusPolicy(Qt.StrongFocus)

    def mousePressEvent(self, event):
        self.setFocus()

    def update_status_ui(self):
        color = "green" if self.is_recording else "red"
        state = "ON" if self.is_recording else "OFF"
        self.status_label.setText(f"Recording: {state} | Intent: {self.current_intent}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 16px;")

    def image_cb(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.bridge.new_image_signal.emit(cv_image)
        except RuntimeError:
            pass

    def wheels_cb(self, msg):
        self.current_vel_left = msg.vel_left
        self.current_vel_right = msg.vel_right

    def update_image_display(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))
        self.motor_label.setText(f"Motors -> Left: {self.current_vel_left:.2f} | Right: {self.current_vel_right:.2f}")

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            self.keys_pressed.add(event.key())

        if event.key() == Qt.Key_I:
            self.current_intent = "straight"
            self.update_status_ui()
        elif event.key() == Qt.Key_J:
            self.current_intent = "left"
            self.update_status_ui()
        elif event.key() == Qt.Key_L:
            self.current_intent = "right"
            self.update_status_ui()
        elif event.key() == Qt.Key_K:
            self.current_intent = "lane_following"
            self.update_status_ui()
        elif event.key() == Qt.Key_R:
            self.is_recording = not self.is_recording
            self.update_status_ui()

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat() and event.key() in self.keys_pressed:
            self.keys_pressed.remove(event.key())

    def publish_commands(self):
        v = 0.0
        omega = 0.0
        
        # Drive using the strictly independent LIVE variables
        if Qt.Key_W in self.keys_pressed: v += self.live_v_fwd
        if Qt.Key_S in self.keys_pressed: v += self.live_v_rev
        if Qt.Key_A in self.keys_pressed: 
            omega += self.live_omega_a
            v += self.live_v_bump_a
        if Qt.Key_D in self.keys_pressed: 
            # Note: We subtract here so the UI user can just enter a positive magnitude for D
            omega -= self.live_omega_d
            v += self.live_v_bump_d

        # Log Keys
        w_state = 1 if Qt.Key_W in self.keys_pressed else 0
        a_state = 1 if Qt.Key_A in self.keys_pressed else 0
        s_state = 1 if Qt.Key_S in self.keys_pressed else 0
        d_state = 1 if Qt.Key_D in self.keys_pressed else 0

        cmd_msg = Twist2DStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.v = v
        cmd_msg.omega = omega
        self.cmd_pub.publish(cmd_msg)

        rec_msg = Bool()
        rec_msg.data = self.is_recording
        self.rec_pub.publish(rec_msg)

        intent_msg = String()
        intent_msg.data = self.current_intent
        self.intent_pub.publish(intent_msg)

        keys_msg = String()
        keys_msg.data = f"{w_state},{a_state},{s_state},{d_state}"
        self.keys_pub.publish(keys_msg)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DataCollectorUI()
    ex.show()
    sys.exit(app.exec_())