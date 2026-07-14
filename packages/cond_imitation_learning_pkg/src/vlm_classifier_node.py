#!/usr/bin/env python3

import sys
import os
import argparse
import rospy
from PIL import Image as PILImage

from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QHBoxLayout, QGroupBox,
                              QPushButton, QLineEdit, QComboBox, QSpinBox, QApplication)
from PyQt5.QtCore import pyqtSignal, QThread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autonomous_node import AutonomousDriverUI

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.insert(0, os.path.join(REPO_ROOT, "vlmplanner"))
from vlm_planner import NavFSM, Config, parse_instruction  # noqa: E402
from live_oracle import TrainedOracle  # noqa: E402
from segmentation import crop_image  # noqa: E402

CKPT_DIR = os.path.join(REPO_ROOT, "dataset", "checkpoints")


CHECKPOINTS = {
    "SmolVLM2-256M (crop, best)": (os.path.join(CKPT_DIR, "head_256m_crop", "checkpoint.pt"), "crop"),
    "SmolVLM2-256M (segmented)": (os.path.join(CKPT_DIR, "head_256m_spatial3", "checkpoint.pt"), "cropseg"),
    "SmolVLM2-500M (segmented)": (os.path.join(CKPT_DIR, "head_500m_spatial3", "checkpoint.pt"), "cropseg"),
}


class VLMClassifierWorker(QThread):
    # plan_str, fsm_state, vlm_query, vlm_answer, confidence, memory_count, intent
    update_signal = pyqtSignal(str, str, str, str, float, int, str)
    status_signal = pyqtSignal(str)

    def __init__(self, instruction: str, checkpoint_path: str, at_window: int, exit_window: int, target_fps: int):
        super().__init__()
        self.instruction = instruction
        self.checkpoint_path = checkpoint_path
        self.target_fps = target_fps
        self.running = True
        self.latest_image = None  # PIL.Image, already cropped+segmented+colorized

        cfg = Config(at_intersection_window=at_window, exit_window=exit_window)
        plan = parse_instruction(instruction)
        self.fsm = NavFSM(plan, cfg)
        self.oracle = None

    def set_image(self, pil_image):
        self.latest_image = pil_image

    def stop(self):
        self.running = False

    def run(self):
        self.status_signal.emit(f"loading {os.path.basename(os.path.dirname(self.checkpoint_path))} ...")
        try:
            self.oracle = TrainedOracle(self.checkpoint_path)
        except Exception as e:
            self.status_signal.emit(f"failed to load checkpoint: {e}")
            return
        self.status_signal.emit("ready")
        self.update_signal.emit(str(self.fsm.plan), self.fsm.state.name, "-", "-", 0.0, self.fsm.count, "lane_following")

        rate = rospy.Rate(self.target_fps)
        while self.running and not rospy.is_shutdown():
            if self.latest_image is None or self.fsm.done:
                rate.sleep()
                continue

            query = self.fsm.needs()
            if query is None:
                intent = self.fsm.step(None)
                self.update_signal.emit(str(self.fsm.plan), self.fsm.state.name, "-", "-", 0.0, self.fsm.count, intent)
                rate.sleep()
                continue

            direction = self.fsm._current[0]
            if direction == "stop":
                direction = "straight"

            image = self.latest_image
            if query == "exit":
                ans, conf = self.oracle.classify_exit(image)
            else:
                ans, conf = self.oracle.classify_approach(image, direction=direction)

            intent = self.fsm.step(ans)
            self.update_signal.emit(str(self.fsm.plan), self.fsm.state.name, query, ans, conf, self.fsm.count, intent)
            rate.sleep()


class VLMClassifierUI(AutonomousDriverUI):
    def __init__(self, approach=5, output_mode=None):
        self.vlm_worker = None
        self.vlm_input_mode = "crop"  # matches CHECKPOINTS' default (first) entry
        super().__init__(approach=approach, output_mode=output_mode)

    def init_ui(self):
        super().init_ui()

        # onnx is the default here, no TensorRT on Mac
        idx = self.backend_combo.findText("ONNX Runtime")
        if idx >= 0:
            self.backend_combo.setCurrentIndex(idx)

        main_layout = self.centralWidget().layout()

        vlm_group = QGroupBox("VLM Planner (trained classifier head)")
        vlm_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        vlm_layout = QVBoxLayout()

        cfg_layout = QHBoxLayout()
        self.checkpoint_combo = QComboBox()
        for name, (path, input_mode) in CHECKPOINTS.items():
            label = name if os.path.exists(path) else f"{name} (not trained yet)"
            self.checkpoint_combo.addItem(label, (path, input_mode))

        self.at_window_spin = QSpinBox()
        self.at_window_spin.setRange(1, 10)
        self.at_window_spin.setValue(3)
        self.exit_window_spin = QSpinBox()
        self.exit_window_spin.setRange(1, 10)
        self.exit_window_spin.setValue(3)
        self.vlm_fps_combo = QComboBox()
        self.vlm_fps_combo.addItems(["Max", "10", "5", "4", "2", "1"])
        self.vlm_fps_combo.setCurrentText("4")

        cfg_layout.addWidget(QLabel("Checkpoint:"))
        cfg_layout.addWidget(self.checkpoint_combo)
        cfg_layout.addWidget(QLabel("Approach-confirm frames:"))
        cfg_layout.addWidget(self.at_window_spin)
        cfg_layout.addWidget(QLabel("Exit-confirm frames:"))
        cfg_layout.addWidget(self.exit_window_spin)
        cfg_layout.addWidget(QLabel("VLM FPS:"))
        cfg_layout.addWidget(self.vlm_fps_combo)
        vlm_layout.addLayout(cfg_layout)

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

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setStyleSheet("background-color: #fd7e14; color: white; font-weight: bold; padding: 8px;")
        self.btn_reset.clicked.connect(self.reset_all)

        input_layout.addWidget(QLabel("Instruction:"))
        input_layout.addWidget(self.prompt_input)
        input_layout.addWidget(self.btn_plan)
        input_layout.addWidget(self.btn_stop_vlm)
        input_layout.addWidget(self.btn_reset)
        vlm_layout.addLayout(input_layout)

        status_layout = QHBoxLayout()
        self.lbl_plan = QLabel("Plan: []")
        self.lbl_fsm = QLabel("FSM: IDLE")
        self.lbl_vlm_ans = QLabel("VLM: N/A")
        self.lbl_memory = QLabel("Count: 0")
        for lbl in [self.lbl_plan, self.lbl_fsm, self.lbl_vlm_ans, self.lbl_memory]:
            lbl.setStyleSheet("font-family: monospace; font-size: 14px; background-color: #f8f9fa; "
                              "padding: 5px; border: 1px solid #ccc;")
            status_layout.addWidget(lbl)
        vlm_layout.addLayout(status_layout)

        vlm_group.setLayout(vlm_layout)
        count = main_layout.count()
        main_layout.insertWidget(count - 1, vlm_group)
        self.resize(950, 900)

    def start_vlm_planner(self):
        instruction = self.prompt_input.text().strip()
        if not instruction:
            rospy.logwarn("Instruction is empty!")
            return
        checkpoint_path, input_mode = self.checkpoint_combo.currentData()
        if not os.path.exists(checkpoint_path):
            rospy.logwarn(f"Checkpoint not found: {checkpoint_path}")
            self.lbl_vlm_ans.setText(f"VLM: checkpoint not found ({checkpoint_path})")
            return
        self.vlm_input_mode = input_mode

        self.btn_plan.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.checkpoint_combo.setEnabled(False)
        self.at_window_spin.setEnabled(False)
        self.exit_window_spin.setEnabled(False)
        self.vlm_fps_combo.setEnabled(False)
        self.btn_stop_vlm.setEnabled(True)

        fps_text = self.vlm_fps_combo.currentText()
        target_fps = 1000 if fps_text == "Max" else int(fps_text)

        rospy.loginfo(f"Starting VLM classifier thread. Instruction: '{instruction}' | checkpoint: {checkpoint_path}")
        self.vlm_worker = VLMClassifierWorker(
            instruction, checkpoint_path,
            self.at_window_spin.value(), self.exit_window_spin.value(), target_fps,
        )
        self.vlm_worker.update_signal.connect(self.on_vlm_update)
        self.vlm_worker.status_signal.connect(self.on_vlm_status)
        self.vlm_worker.start()

    def stop_vlm_planner(self):
        if self.vlm_worker:
            self.vlm_worker.stop()
            self.vlm_worker.wait()
            self.vlm_worker = None

        self.btn_plan.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.checkpoint_combo.setEnabled(True)
        self.at_window_spin.setEnabled(True)
        self.exit_window_spin.setEnabled(True)
        self.vlm_fps_combo.setEnabled(True)
        self.btn_stop_vlm.setEnabled(False)
        self.lbl_fsm.setText("FSM: STOPPED")

    def reset_all(self):

        self.stop_vlm_planner()

        if self.is_autonomous_active:
            self.stop_autonomous()

        self.set_intent("lane_following")

        self.lbl_plan.setText("Plan: []")
        self.lbl_fsm.setText("FSM: IDLE")
        self.lbl_vlm_ans.setText("VLM: N/A")
        self.lbl_vlm_ans.setStyleSheet("font-family: monospace; font-size: 14px; background-color: #f8f9fa; "
                                       "padding: 5px; border: 1px solid #ccc;")
        self.lbl_memory.setText("Count: 0")

        rospy.loginfo("Reset: VLM planner/FSM cleared, PilotNet stopped, intent reset to lane_following.")

    def on_vlm_status(self, text):
        self.lbl_vlm_ans.setText(f"VLM: {text}")

    def on_vlm_update(self, plan_str, fsm_state, vlm_query, vlm_answer, confidence, memory_count, intent):
        self.lbl_plan.setText(f"Plan: {plan_str}")
        self.lbl_fsm.setText(f"FSM: {fsm_state}")

        color = "#28a745" if vlm_answer == "yes" else ("#ffc107" if vlm_answer == "pass" else "#dc3545")
        query_tag = f"[{vlm_query}] " if vlm_query != "-" else ""
        self.lbl_vlm_ans.setText(f"VLM: {query_tag}{vlm_answer.upper()} ({confidence:.2f})")
        self.lbl_vlm_ans.setStyleSheet(f"font-family: monospace; font-size: 14px; color: {color}; "
                                       "font-weight: bold; background-color: #f8f9fa; padding: 5px; border: 1px solid #ccc;")
        self.lbl_memory.setText(f"Count: {memory_count}")
        self.set_intent(intent)

    def update_ui(self, raw_img, model_img, out1, out2, intent):
        super().update_ui(raw_img, model_img, out1, out2, intent)
        if self.vlm_worker is not None and self.vlm_worker.running:
            if getattr(self, "vlm_input_mode", "crop") == "crop":
                vlm_img = crop_image(PILImage.fromarray(raw_img))
            else:
                vlm_img = PILImage.fromarray(model_img)
            self.vlm_worker.set_image(vlm_img.copy())

    def closeEvent(self, event):
        self.stop_vlm_planner()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Native Mac CIL + trained VLM classifier dashboard")
    parser.add_argument("--approach", type=int, choices=[5, 7], default=5,
                        help="5: regNheadv2, 7: FiLM -- the two approaches the VLM classifier is trained against")
    parser.add_argument("--output_mode", type=str, choices=["twist", "wheels"], default=None)
    args, unknown = parser.parse_known_args(rospy.myargv()[1:])

    app = QApplication(sys.argv)
    driver_ui = VLMClassifierUI(approach=args.approach, output_mode=args.output_mode)
    driver_ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
