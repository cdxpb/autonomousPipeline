#!/usr/bin/env python3
import sys
import os
import time
import json
import argparse
import rospy
import cv2
import requests
import numpy as np
from PIL import Image

# PyQt5
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QGroupBox, QPushButton, QLineEdit, QCheckBox, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# Import VLM Planner
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_planner import NavFSM, Config, parse_instruction, VLMOracle

class VLMWorker(QThread):
    update_signal = pyqtSignal(str, str, str, float, int, str)

    def __init__(self, instruction: str, model_id: str, server_url: str = None):
        super().__init__()
        self.instruction = instruction
        self.running = True
        self.latest_image = None
        self.server_url = server_url
        
        rospy.loginfo(f"VLMWorker: Initializing with instruction '{instruction}'")
        
        cfg = Config(model_id=model_id, precision="fp32")
        plan = parse_instruction(instruction)
        self.fsm = NavFSM(plan, cfg)
        
        if not self.server_url:
            rospy.loginfo("Loading VLMOracle LOCALLY (Warning: Slow on CPU)")
            self.vlm = VLMOracle(cfg)
        else:
            rospy.loginfo(f"VLMWorker: Using remote server at {self.server_url}")
            self.vlm = None
        
    def set_image(self, rgb_image):
        self.latest_image = rgb_image
        
    def stop(self):
        self.running = False

    def run(self):
        self.update_signal.emit(str(self.fsm.plan), self.fsm.state.name, "no", 0.0, self.fsm.count, "lane_following")
        
        rate = rospy.Rate(4) # 4 FPS for VLM
        while self.running and not rospy.is_shutdown():
            if self.latest_image is not None and not self.fsm.done:
                current_target = self.fsm._current[0] if not self.fsm.done else "straight"
                if current_target == "stop":
                    current_target = "straight"
                    
                if self.server_url:
                    try:
                        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(self.latest_image, cv2.COLOR_RGB2BGR))
                        if is_success:
                            response = requests.post(
                                f"{self.server_url}/predict/vlm", 
                                data={"direction": current_target},
                                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                                timeout=15.0
                            )
                            if response.status_code == 200:
                                data = response.json()
                                ans = data.get("ans", "no")
                                conf = data.get("confidence", 0.0)
                            else:
                                rospy.logwarn_throttle(2.0, f"Server returned {response.status_code}")
                                rate.sleep()
                                continue
                    except requests.exceptions.RequestException as e:
                        rospy.logwarn_throttle(2.0, f"VLM Server connection failed: {e}")
                        rate.sleep()
                        continue
                else:
                    pil_image = Image.fromarray(self.latest_image)
                    ans, conf = self.vlm.at_intersection(pil_image, direction=current_target)
                
                intent = self.fsm.step(ans)
                self.update_signal.emit(str(self.fsm.plan), self.fsm.state.name, ans, conf, self.fsm.count, intent)
                
            rate.sleep()

class MacDashboardUI(QMainWindow):
    # Signals to safely update UI from ROS thread
    telemetry_img_signal = pyqtSignal(np.ndarray, np.ndarray)
    telemetry_state_signal = pyqtSignal(float, float, str, str)

    def __init__(self):
        super().__init__()
        rospy.init_node('mac_dashboard_ui', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'golduck')
        self.current_intent = "straight"
        
        self.vlm_worker = None
        self.model_id = "HuggingFaceTB/SmolVLM2-256M-Instruct"

        self.telemetry_img_signal.connect(self.update_images)
        self.telemetry_state_signal.connect(self.update_state)

        self.init_ui()

        # Telemetry Subscribers
        self.img_sub = rospy.Subscriber(f"/{self.veh}/telemetry/preview/compressed", CompressedImage, self.img_cb, queue_size=1)
        self.state_sub = rospy.Subscriber(f"/{self.veh}/telemetry/state", String, self.state_cb, queue_size=1)
        self.intent_pub = rospy.Publisher(f"/{self.veh}/data_collector/intent", String, queue_size=1)

    def init_ui(self):
        self.setWindowTitle(f"Mac Dashboard (Connected to {self.veh})")
        self.setGeometry(100, 100, 900, 850)

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

        # --- VLM PLANNER PANEL ---
        vlm_group = QGroupBox("VLM High-Level Planner")
        vlm_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        vlm_layout = QVBoxLayout()
        
        server_layout = QHBoxLayout()
        self.chk_offload = QCheckBox("Offload VLM to Native API Server")
        self.chk_offload.setChecked(True)
        self.chk_offload.stateChanged.connect(self.toggle_server_input)
        
        self.server_input = QLineEdit()
        self.server_input.setText("http://127.0.0.1:8000")
        
        server_layout.addWidget(self.chk_offload)
        server_layout.addWidget(QLabel("Server URL:"))
        server_layout.addWidget(self.server_input)
        vlm_layout.addLayout(server_layout)

        input_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("e.g. take the first left, second right and stop")
        self.prompt_input.setText("take the first left, second right and stop")
        
        self.btn_plan = QPushButton("Start VLM Planner")
        self.btn_plan.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold; padding: 8px;")
        self.btn_plan.clicked.connect(self.start_vlm_planner)
        
        self.btn_stop_vlm = QPushButton("Stop VLM")
        self.btn_stop_vlm.setStyleSheet("background-color: #6c757d; color: white; padding: 8px;")
        self.btn_stop_vlm.setEnabled(False)
        self.btn_stop_vlm.clicked.connect(self.stop_vlm_planner)
        
        input_layout.addWidget(QLabel("Instruction:"))
        input_layout.addWidget(self.prompt_input)
        input_layout.addWidget(self.btn_plan)
        input_layout.addWidget(self.btn_stop_vlm)
        vlm_layout.addLayout(input_layout)
        
        status_layout = QHBoxLayout()
        self.lbl_plan = QLabel("Plan: []")
        self.lbl_fsm = QLabel("FSM: IDLE")
        self.lbl_vlm_ans = QLabel("VLM: N/A")
        self.lbl_memory = QLabel("Mem Count: 0")
        
        for lbl in [self.lbl_plan, self.lbl_fsm, self.lbl_vlm_ans, self.lbl_memory]:
            lbl.setStyleSheet("font-family: monospace; font-size: 14px; background-color: #f8f9fa; padding: 5px; border: 1px solid #ccc;")
            status_layout.addWidget(lbl)
            
        vlm_layout.addLayout(status_layout)
        vlm_group.setLayout(vlm_layout)
        layout.addWidget(vlm_group)

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
            
            # Send low-res raw preview to VLM Planner!
            if self.vlm_worker and self.vlm_worker.running:
                self.vlm_worker.set_image(raw_preview.copy())

        except Exception as e:
            rospy.logerr(f"Dashboard Telemetry Image Error: {e}")

    def state_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.telemetry_state_signal.emit(data["out1"], data["out2"], data["intent"], data["backend"])
        except Exception as e:
            rospy.logerr(f"Dashboard State Error: {e}")

    def update_images(self, raw_img, model_img):
        h, w, ch = raw_img.shape
        qt_raw = QImage(raw_img.data, w, h, ch * w, QImage.Format_RGB888)
        self.live_label.setPixmap(QPixmap.fromImage(qt_raw).scaled(self.live_label.width(), self.live_label.height(), Qt.KeepAspectRatio))

        h, w, ch = model_img.shape
        qt_model = QImage(model_img.data, w, h, ch * w, QImage.Format_RGB888)
        self.model_label.setPixmap(QPixmap.fromImage(qt_model).scaled(self.model_label.width(), self.model_label.height(), Qt.KeepAspectRatio))

    def update_state(self, out1, out2, intent, backend):
        self.motor_text.setText(f"L: {out1:+.3f} | R: {out2:+.3f} | Backend: {backend}")
        if self.current_intent != intent:
            self.current_intent = intent
            self.update_intent_button_styles()

    def toggle_server_input(self):
        self.server_input.setEnabled(self.chk_offload.isChecked())

    def start_vlm_planner(self):
        instruction = self.prompt_input.text().strip()
        if not instruction: return
            
        self.btn_plan.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.btn_stop_vlm.setEnabled(True)
        self.chk_offload.setEnabled(False)
        self.server_input.setEnabled(False)
        
        server_url = self.server_input.text().strip() if self.chk_offload.isChecked() else None
        
        self.vlm_worker = VLMWorker(instruction, self.model_id, server_url=server_url)
        self.vlm_worker.update_signal.connect(self.on_vlm_update)
        self.vlm_worker.start()

    def stop_vlm_planner(self):
        if self.vlm_worker:
            self.vlm_worker.stop()
            self.vlm_worker.wait()
            self.vlm_worker = None
            
        self.btn_plan.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.btn_stop_vlm.setEnabled(False)
        self.chk_offload.setEnabled(True)
        self.server_input.setEnabled(self.chk_offload.isChecked())
        
        self.lbl_fsm.setText("FSM: STOPPED")
        
    def on_vlm_update(self, plan_str, fsm_state, ans, confidence, memory_count, intent):
        self.lbl_plan.setText(f"Plan: {plan_str}")
        self.lbl_fsm.setText(f"FSM: {fsm_state}")
        
        color = "#28a745" if ans == "yes" else ("#ffc107" if ans == "pass" else "#dc3545")
        self.lbl_vlm_ans.setText(f"VLM: {ans.upper()} ({confidence:.2f})")
        self.lbl_vlm_ans.setStyleSheet(f"font-family: monospace; font-size: 14px; color: {color}; font-weight: bold; background-color: #f8f9fa; padding: 5px; border: 1px solid #ccc;")
        
        self.lbl_memory.setText(f"Mem Count: {memory_count}")
        self.publish_intent(intent)

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
