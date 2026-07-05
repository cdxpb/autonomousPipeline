#!/usr/bin/env python3
import sys
import os
import time
import argparse
import rospy
import cv2
import requests
import numpy as np
from PIL import Image

# PyQt5
from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QHBoxLayout, 
                             QWidget, QGroupBox, QPushButton, QLineEdit, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal, QThread

# Import from existing nodes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autonomous_node import AutonomousDriverUI

# Import VLM Planner
from vlm_planner import NavFSM, Config, parse_instruction, VLMOracle

class VLMWorker(QThread):
    # Signals for updating UI
    update_signal = pyqtSignal(str, str, bool, float, int, str)

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
        # 4 FPS rate limiting
        rate = rospy.Rate(4)
        while self.running and not rospy.is_shutdown():
            if self.latest_image is not None and not self.fsm.done:
                # Local or Remote Inference
                if self.server_url:
                    try:
                        # Compress to JPEG for faster network transfer
                        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(self.latest_image, cv2.COLOR_RGB2BGR))
                        if is_success:
                            response = requests.post(
                                f"{self.server_url}/predict/vlm", 
                                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                                timeout=2.0
                            )
                            if response.status_code == 200:
                                data = response.json()
                                at_inter = data["at_intersection"]
                                conf = data["confidence"]
                            else:
                                rospy.logwarn_throttle(2.0, f"Server returned {response.status_code}")
                                rate.sleep()
                                continue
                    except requests.exceptions.RequestException as e:
                        rospy.logwarn_throttle(2.0, f"VLM Server connection failed: {e}")
                        rate.sleep()
                        continue
                else:
                    # Run purely locally (CPU heavy)
                    pil_image = Image.fromarray(self.latest_image)
                    at_inter, conf = self.vlm.at_intersection(pil_image)
                
                # Step the navigator
                intent = self.fsm.step(at_inter)
                plan_str = str(self.fsm.plan)
                
                # Emit signal to update UI and PilotNet intent
                self.update_signal.emit(
                    plan_str,
                    self.fsm.state.name,
                    at_inter,
                    conf,
                    self.fsm.count,
                    intent
                )
                
            rate.sleep()


class VLMAutonomousDriverUI(AutonomousDriverUI):
    def __init__(self):
        super().__init__(approach=5, output_mode="wheels")
        self.vlm_worker = None
        self.model_id = "HuggingFaceTB/SmolVLM2-256M-Instruct"

    def init_ui(self):
        # Call base class UI setup
        super().init_ui()
        
        main_layout = self.centralWidget().layout()
        
        # --- VLM PLANNER PANEL ---
        vlm_group = QGroupBox("VLM High-Level Planner")
        vlm_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        vlm_layout = QVBoxLayout()
        
        # Server / Network Row
        server_layout = QHBoxLayout()
        self.chk_offload = QCheckBox("Offload VLM to Native API Server")
        self.chk_offload.setChecked(True) # Default to offloading
        self.chk_offload.stateChanged.connect(self.toggle_server_input)
        
        self.server_input = QLineEdit()
        self.server_input.setText("http://127.0.0.1:8000")
        
        server_layout.addWidget(self.chk_offload)
        server_layout.addWidget(QLabel("Server URL:"))
        server_layout.addWidget(self.server_input)
        vlm_layout.addLayout(server_layout)

        # Input row
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
        
        # Status row
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
        
        # Insert above the intent panel
        count = main_layout.count()
        main_layout.insertWidget(count - 1, vlm_group)
        
        self.resize(900, 850)

    def toggle_server_input(self):
        self.server_input.setEnabled(self.chk_offload.isChecked())

    def start_vlm_planner(self):
        instruction = self.prompt_input.text().strip()
        if not instruction:
            rospy.logwarn("Instruction is empty!")
            return
            
        self.btn_plan.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.btn_stop_vlm.setEnabled(True)
        self.chk_offload.setEnabled(False)
        self.server_input.setEnabled(False)
        
        # Check offloading status
        server_url = self.server_input.text().strip() if self.chk_offload.isChecked() else None
        
        rospy.loginfo(f"Starting VLM thread. Instruction: '{instruction}' | Server: {server_url}")
        
        # Initialize and start thread
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
        
    def on_vlm_update(self, plan_str, fsm_state, at_intersection, confidence, memory_count, intent):
        self.lbl_plan.setText(f"Plan: {plan_str}")
        self.lbl_fsm.setText(f"FSM: {fsm_state}")
        
        ans_text = "YES" if at_intersection else "NO"
        color = "#28a745" if at_intersection else "#dc3545"
        self.lbl_vlm_ans.setText(f"VLM: {ans_text} ({confidence:.2f})")
        self.lbl_vlm_ans.setStyleSheet(f"font-family: monospace; font-size: 14px; color: {color}; font-weight: bold; background-color: #f8f9fa; padding: 5px; border: 1px solid #ccc;")
        
        self.lbl_memory.setText(f"Mem Count: {memory_count}")
        
        # Crucial: Override current intent so PilotNet uses it
        self.set_intent(intent)

    def update_ui(self, raw_img, model_img, out1, out2, intent):
        super().update_ui(raw_img, model_img, out1, out2, intent)
        
        if self.vlm_worker and self.vlm_worker.running:
            # For approach 2-5, model_img is the semantic segmentation mask (0, 85, 170, 255 for classes).
            # Convert it back to a 3-channel image for the VLM.
            if len(model_img.shape) == 2:
                model_rgb = cv2.cvtColor(model_img, cv2.COLOR_GRAY2RGB)
            else:
                model_rgb = model_img.copy()
            self.vlm_worker.set_image(model_rgb)

    def closeEvent(self, event):
        self.stop_vlm_planner()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    driver_ui = VLMAutonomousDriverUI()
    driver_ui.show()
    
    sys.exit(app.exec_())
