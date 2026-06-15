#!/usr/bin/env python3
import sys
import os
import rospy
import cv2
import numpy as np

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
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
        
        # Publishers & Subscribers (tcp_nodelay=True added to kill network lag)
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
        
        # Track multiple key presses
        self.keys_pressed = set()

        self.init_ui()

        # Control Loop
        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.publish_commands)
        self.control_timer.start(100) # 10 Hz

    def init_ui(self):
        self.setWindowTitle("CIL Data Collector")
        self.setGeometry(100, 100, 640, 580)

        widget = QWidget(self)
        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Waiting for Camera Feed...")
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label)

        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.motor_label = QLabel(self)
        self.motor_label.setAlignment(Qt.AlignCenter)
        self.motor_label.setStyleSheet("color: black; font-family: monospace; font-size: 14px;")
        layout.addWidget(self.motor_label)

        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.update_status_ui()

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
            self.current_intent = "stop"
            self.update_status_ui()
        elif event.key() == Qt.Key_R:
            self.is_recording = not self.is_recording
            self.update_status_ui()

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat() and event.key() in self.keys_pressed:
            self.keys_pressed.remove(event.key())

    def publish_commands(self):
        # Calculate velocity based on ALL currently held keys
        v = 0.0
        omega = 0.0
        if Qt.Key_W in self.keys_pressed: v += 0.5
        if Qt.Key_S in self.keys_pressed: v -= 0.5
        if Qt.Key_A in self.keys_pressed: 
            omega += 7.0
            v += 0.1 # Slight forward bump to overcome physical floor friction
        if Qt.Key_D in self.keys_pressed: 
            omega -= 15.0
            v += 0.15

        # Extract specific WASD binary states for logging
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