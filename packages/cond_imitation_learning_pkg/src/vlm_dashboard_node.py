#!/usr/bin/env python3
"""
VLM-enabled CIL dashboard. Adds a VLM planner panel on top of MacDashboardUI that
drives the intent from a natural-language instruction. Launched via run_dashboard_vlm.sh.
"""
import sys
import os
import rospy
import cv2
import requests
from PIL import Image

# PyQt5
from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLineEdit, QCheckBox)
from PyQt5.QtCore import pyqtSignal, QThread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mac_dashboard_node import MacDashboardUI
from vlm_planner import NavFSM, Config, parse_instruction, VLMOracle


class VLMWorker(QThread):
    # plan_str, fsm_state, vlm_query, vlm_answer, confidence, memory_count, intent
    update_signal = pyqtSignal(str, str, str, str, float, int, str)

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

    def _query_remote(self, image_rgb, query: str, direction: str):
        is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not is_success:
            return None, 0.0
        response = requests.post(
            f"{self.server_url}/predict/vlm",
            data={"direction": direction, "mode": query},
            files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=15.0
        )
        if response.status_code != 200:
            rospy.logwarn_throttle(2.0, f"Server returned {response.status_code}")
            return None, 0.0
        data = response.json()
        return data.get("ans"), data.get("confidence", 0.0)

    def run(self):
        self.update_signal.emit(str(self.fsm.plan), self.fsm.state.name, "-", "-", 0.0, self.fsm.count, "lane_following")

        rate = rospy.Rate(4)  # 4 FPS for VLM
        while self.running and not rospy.is_shutdown():
            if self.latest_image is None or self.fsm.done:
                rate.sleep()
                continue

            # needs() is None during cooldown/DONE, still step() to decrement the
            # cooldown and emit lane_following without spending a VLM call on it
            query = self.fsm.needs()
            if query is None:
                intent = self.fsm.step(None)
                self.update_signal.emit(str(self.fsm.plan), self.fsm.state.name, "-", "-", 0.0, self.fsm.count, intent)
                rate.sleep()
                continue

            direction = self.fsm._current[0]
            if direction == "stop":
                direction = "straight"

            try:
                if self.server_url:
                    ans, conf = self._query_remote(self.latest_image, query, direction)
                    if ans is None:
                        rate.sleep()
                        continue
                elif query == "exit":
                    ans, conf = self.vlm.classify_exit(Image.fromarray(self.latest_image))
                else:
                    ans, conf = self.vlm.classify_approach(Image.fromarray(self.latest_image), direction=direction)
            except requests.exceptions.RequestException as e:
                rospy.logwarn_throttle(2.0, f"VLM Server connection failed: {e}")
                rate.sleep()
                continue

            intent = self.fsm.step(ans)
            self.update_signal.emit(str(self.fsm.plan), self.fsm.state.name, query, ans, conf, self.fsm.count, intent)

            rate.sleep()


class VLMMacDashboardUI(MacDashboardUI):
    def __init__(self):
        super().__init__()
        self.vlm_worker = None
        self.model_id = "HuggingFaceTB/SmolVLM2-256M-Instruct"

    def init_ui(self):
        # Call base class UI setup (telemetry previews + manual intent override)
        super().init_ui()

        main_layout = self.centralWidget().layout()

        # --- VLM PLANNER PANEL ---
        vlm_group = QGroupBox("VLM High-Level Planner")
        vlm_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        vlm_layout = QVBoxLayout()

        # Server / Network Row. Defaults come from run_dashboard_vlm.sh's env vars
        # (VLM_LOCAL/VLM_SERVER_URL) so --local doesn't also require unchecking this
        # by hand; still overridable in the UI either way.
        local_default = os.environ.get("VLM_LOCAL", "false").lower() == "true"
        server_default = os.environ.get("VLM_SERVER_URL", "http://127.0.0.1:8000")

        server_layout = QHBoxLayout()
        self.chk_offload = QCheckBox("Offload VLM to Native API Server")
        self.chk_offload.setChecked(not local_default)
        self.chk_offload.stateChanged.connect(self.toggle_server_input)

        self.server_input = QLineEdit()
        self.server_input.setText(server_default)
        self.server_input.setEnabled(not local_default)

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

        # Insert above the manual-intent-override group (the last widget the base
        # class added).
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

        server_url = self.server_input.text().strip() if self.chk_offload.isChecked() else None

        rospy.loginfo(f"Starting VLM thread. Instruction: '{instruction}' | Server: {server_url}")

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

    def on_vlm_update(self, plan_str, fsm_state, vlm_query, vlm_answer, confidence, memory_count, intent):
        self.lbl_plan.setText(f"Plan: {plan_str}")
        self.lbl_fsm.setText(f"FSM: {fsm_state}")

        color = "#28a745" if vlm_answer == "yes" else ("#ffc107" if vlm_answer == "pass" else "#dc3545")
        query_tag = f"[{vlm_query}] " if vlm_query != "-" else ""
        self.lbl_vlm_ans.setText(f"VLM: {query_tag}{vlm_answer.upper()} ({confidence:.2f})")
        self.lbl_vlm_ans.setStyleSheet(f"font-family: monospace; font-size: 14px; color: {color}; font-weight: bold; background-color: #f8f9fa; padding: 5px; border: 1px solid #ccc;")

        self.lbl_memory.setText(f"Mem Count: {memory_count}")
        self.publish_intent(intent)

    def on_raw_frame(self, raw_preview):
        # raw (non-mask) half of the telemetry preview, feed it to the VLM worker
        if self.vlm_worker and self.vlm_worker.running:
            self.vlm_worker.set_image(raw_preview.copy())

    def closeEvent(self, event):
        self.stop_vlm_planner()
        super().closeEvent(event)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    driver_ui = VLMMacDashboardUI()
    driver_ui.show()
    sys.exit(app.exec_())
