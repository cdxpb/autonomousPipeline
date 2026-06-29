#!/usr/bin/env python3
import sys
import os
import argparse
import rospy
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    pass

# ROS Messages
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import String

# PyQt5 UI Components
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, 
                             QHBoxLayout, QWidget, QGroupBox, QPushButton, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QObject

# DRY Imports from your package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import crop_image, get_eval_transforms, INTENT_MAP, CROP_TOP_ROWS

# ---------------------------------------------------------
# BACKEND LIBRARIES
# ---------------------------------------------------------
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class TelemetryBridge(QObject):
    # Sends: raw_frame, model_frame, vel_left, vel_right, current_intent
    telemetry_signal = pyqtSignal(np.ndarray, np.ndarray, float, float, str)

class AutonomousDriverUI(QMainWindow):
    def __init__(self, skip_segmentation=False):
        super().__init__()
        rospy.init_node('autonomous_driver_ui_node', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'golduck')
        self.skip_segmentation = skip_segmentation
        self.current_intent = "straight"
        
        # State Flags
        self.is_autonomous_active = False
        self.active_backend = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model Paths
        if self.skip_segmentation:
            self.model_path = "../autonomouspipeline/models/pilotnet/best_model"
        else:
            self.model_path = "../autonomouspipeline/models/pilotnet/segPilot_approach2"
            self.yolo_path = "../autonomouspipeline/models/yolo_model/yolo_model.onnx"

        self.transform = get_eval_transforms()

        # Thread-safe communication bridge
        self.bridge = TelemetryBridge()
        self.bridge.telemetry_signal.connect(self.update_ui)

        # UI Setup
        self.init_ui()

        # Always load YOLO if not skipping, so we can preview the mask even when stopped
        if not self.skip_segmentation:
            rospy.loginfo("UI: Loading YOLO Semantic Segmentation model for previews...")
            self.yolo_session = YOLO(self.yolo_path, task='semantic')

        # ROS Publishers & Subscribers``
        self.cmd_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1, tcp_nodelay=True)
        self.image_sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.image_cb, queue_size=1, buff_size=2**24, tcp_nodelay=True)
        self.intent_sub = rospy.Subscriber(f"/{self.veh}/data_collector/intent", String, self.intent_cb)

    def init_ui(self):
        self.setWindowTitle(f"Autonomous Dashboard — {self.veh}")
        self.setGeometry(100, 100, 900, 700)

        main_widget = QWidget(self)
        layout = QVBoxLayout()

        # --- TOP CONTROL BAR ---
        control_group = QGroupBox("Execution Control")
        control_layout = QHBoxLayout()
        
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("PyTorch")
        if ORT_AVAILABLE: self.backend_combo.addItem("ONNX Runtime")
        if TRT_AVAILABLE: self.backend_combo.addItem("TensorRT")
        
        self.btn_start = QPushButton("START AUTONOMOUS")
        self.btn_start.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.start_autonomous)

        self.btn_stop = QPushButton("EMERGENCY STOP")
        self.btn_stop.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 10px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_autonomous)

        control_layout.addWidget(QLabel("Inference Backend:"))
        control_layout.addWidget(self.backend_combo)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # --- CAMERA DISPLAYS ---
        img_layout = QHBoxLayout()
        
        self.live_label = QLabel(self)
        self.live_label.setAlignment(Qt.AlignCenter)
        self.live_label.setText("Waiting for Raw Feed...")
        self.live_label.setFixedSize(400, 300)
        self.live_label.setStyleSheet("background-color: #121212; color: #aaaaaa; border: 2px solid #333;")
        
        self.model_label = QLabel(self)
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setText("Waiting for Network Preprocessor...")
        self.model_label.setFixedSize(400, 300)
        self.model_label.setStyleSheet("background-color: #121212; color: #aaaaaa; border: 2px solid #333;")

        img_layout.addWidget(self.live_label)
        img_layout.addWidget(self.model_label)
        layout.addLayout(img_layout)

        # --- TELEMETRY GRAPHICS ---
        telemetry_group = QGroupBox("Network Predictions & State")
        telemetry_layout = QHBoxLayout()

        self.motor_text = QLabel("Left Motor: 0.00  |  Right Motor: 0.00 (IDLE)")
        self.motor_text.setStyleSheet("font-family: monospace; font-size: 18px; font-weight: bold; color: #2b2b2b;")
        self.motor_text.setAlignment(Qt.AlignCenter)
        telemetry_layout.addWidget(self.motor_text)

        telemetry_group.setLayout(telemetry_layout)
        layout.addWidget(telemetry_group)

        # --- INTENT INJECTION PANEL ---
        intent_group = QGroupBox("CIL Executive Intent Control")
        intent_layout = QHBoxLayout()

        self.btn_straight = QPushButton("Straight [I]")
        self.btn_left = QPushButton("Left [J]")
        self.btn_right = QPushButton("Right [L]")
        self.btn_stop_intent = QPushButton("Stop [K]")

        self.btn_straight.clicked.connect(lambda: self.set_intent("straight"))
        self.btn_left.clicked.connect(lambda: self.set_intent("left"))
        self.btn_right.clicked.connect(lambda: self.set_intent("right"))
        self.btn_stop_intent.clicked.connect(lambda: self.set_intent("stop"))

        intent_layout.addWidget(self.btn_straight)
        intent_layout.addWidget(self.btn_left)
        intent_layout.addWidget(self.btn_right)
        intent_layout.addWidget(self.btn_stop_intent)
        intent_group.setLayout(intent_layout)
        layout.addWidget(intent_group)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.update_intent_button_styles()
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------------------------------------------------------
    # STATE CONTROL
    # ---------------------------------------------------------
    def start_autonomous(self):
        backend = self.backend_combo.currentText()
        rospy.loginfo(f"Loading Model for Backend: {backend}...")

        try:
            if backend == "PyTorch":
                # Assumes you have a TorchScript exported model (.pt) or you can drop your PilotNet class here
                from pilotnet import ConditionalPilotNet
                self.pt_model = ConditionalPilotNet(in_channels=3 if self.skip_segmentation else 1).to(self.device)
                self.pt_model.load_state_dict(torch.load(f"{self.model_path}.pt", map_location=self.device))
                self.pt_model.eval()

            elif backend == "ONNX Runtime":
                self.ort_session = ort.InferenceSession(f"{self.model_path}.onnx")

            elif backend == "TensorRT":
                self.logger = trt.Logger(trt.Logger.WARNING)
                with open(f"{self.model_path}.engine", "rb") as f, trt.Runtime(self.logger) as runtime:
                    self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()
                channels = 3 if self.skip_segmentation else 1
                self.d_img_in = cuda.mem_alloc(1 * channels * 112 * 224 * 4)
                self.d_intent_in = cuda.mem_alloc(1 * 4 * 4) 
                self.d_output = cuda.mem_alloc(1 * 2 * 4)
                self.bindings = [int(self.d_img_in), int(self.d_intent_in), int(self.d_output)]

            # Switch UI State
            self.active_backend = backend
            self.is_autonomous_active = True
            self.btn_start.setEnabled(False)
            self.backend_combo.setEnabled(False)
            self.btn_stop.setEnabled(True)
            rospy.loginfo(f"Autonomous Mode STARTED with {backend}")

        except Exception as e:
            rospy.logerr(f"Failed to load {backend} model: {e}")

    def stop_autonomous(self):
        self.is_autonomous_active = False
        self.btn_start.setEnabled(True)
        self.backend_combo.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        # Instantly kill motors
        cmd_msg = WheelsCmdStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.vel_left = 0.0
        cmd_msg.vel_right = 0.0
        self.cmd_pub.publish(cmd_msg)
        
        self.motor_text.setText("Left Motor: 0.00  |  Right Motor: 0.00 (STOPPED)")
        rospy.logwarn("Autonomous Mode STOPPED. Zero velocities sent.")

    # ---------------------------------------------------------
    # INTENT HANDLING
    # ---------------------------------------------------------
    def intent_cb(self, msg):
        self.set_intent(msg.data)

    def set_intent(self, intent_str):
        if intent_str in ["straight", "left", "right", "stop"]:
            self.current_intent = intent_str
            self.update_intent_button_styles()

    def update_intent_button_styles(self):
        buttons = {"straight": self.btn_straight, "left": self.btn_left, 
                   "right": self.btn_right, "stop": self.btn_stop_intent}
        for name, btn in buttons.items():
            if name == self.current_intent:
                color = "#dc3545" if name == "stop" else "#007bff"
                btn.setStyleSheet(f"background-color: {color}; color: white; font-weight: bold; padding: 8px;")
            else:
                btn.setStyleSheet("background-color: #f8f9fa; color: black; padding: 8px;")

    # ---------------------------------------------------------
    # MAIN ROS INFERENCE LOOP
    # ---------------------------------------------------------
    def setup_inference_engine(self):
        if not self.skip_segmentation:
            rospy.loginfo("UI: Loading YOLO Semantic Segmentation model for previews...")
            # Added verbose=False to stop the terminal spam!
            self.yolo_session = YOLO(self.yolo_path, task="semantic")

        rospy.loginfo(f"UI: Loading PilotNet model from {self.model_path}...")
        if USE_TENSORRT:
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(f"{self.model_path}.engine", "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            
            channels = 3 if self.skip_segmentation else 1
            self.d_img_in = cuda.mem_alloc(1 * channels * 112 * 224 * 4)
            self.d_intent_in = cuda.mem_alloc(1 * 4 * 4) 
            self.d_output = cuda.mem_alloc(1 * 2 * 4)
            self.bindings = [int(self.d_img_in), int(self.d_intent_in), int(self.d_output)]
        else:
            self.ort_session = ort.InferenceSession(f"{self.model_path}.onnx")


    def image_cb(self, msg):
        try:
            # 1. Decode Image Safely
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None or cv_image.size == 0:
                rospy.logwarn_throttle(2.0, "Received empty or corrupted image frame.")
                return

            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            ui_model_view = None

            # 2. Preprocessing
            if self.skip_segmentation:
                pil_image = Image.fromarray(rgb_image)
                cropped_img = crop_image(pil_image)
                img_tensor = self.transform(cropped_img).unsqueeze(0).numpy()
                
                # Safe OpenCV Resize for UI preview
                if cropped_img.size[0] > 0 and cropped_img.size[1] > 0:
                    ui_model_view = cv2.resize(np.array(cropped_img), (224, 112))
                else:
                    ui_model_view = np.zeros((112, 224, 3), dtype=np.uint8)
            else:
                pil_image = Image.fromarray(rgb_image)

                with torch.no_grad():
                    yolo_results = self.yolo_session(pil_image, verbose=False)

                mask = yolo_results[0].semantic_mask.data.cpu()
                mask_pil = Image.fromarray(mask.numpy())

                cropped_mask_pil = crop_image(mask_pil)
                resized_mask_pil = cropped_mask_pil.resize((224, 112), Image.NEAREST)

                img_tensor = TF.to_tensor(resized_mask_pil).unsqueeze(0).numpy()
                ui_model_view = (np.array(resized_mask_pil) * 85).astype(np.uint8)

            vel_left, vel_right = 0.0, 0.0

            # 3. Inference Gate (Only runs if "START" is active)
            if self.is_autonomous_active:
                intent_tensor = np.array([INTENT_MAP.get(self.current_intent, [1.0, 0.0, 0.0, 0.0])], dtype=np.float32)

                if self.active_backend == "PyTorch":
                    img_t = torch.from_numpy(img_tensor).to(self.device)
                    intent_t = torch.from_numpy(intent_tensor).to(self.device)
                    with torch.no_grad():
                        output = self.pt_model(img_t, intent_t)
                    vel_left, vel_right = output[0][0].item(), output[0][1].item()

                elif self.active_backend == "TensorRT":
                    cuda.memcpy_htod(self.d_img_in, img_tensor)
                    cuda.memcpy_htod(self.d_intent_in, intent_tensor)
                    self.context.execute_v2(bindings=self.bindings)
                    h_output = np.empty((1, 2), dtype=np.float32)
                    cuda.memcpy_dtoh(h_output, self.d_output)
                    vel_left, vel_right = h_output[0][0], h_output[0][1]

                elif self.active_backend == "ONNX Runtime":
                    ort_inputs = {}
                    
                    # Ensure intent is strictly a 2D array: Shape (1, 4)
                    raw_intent = INTENT_MAP.get(self.current_intent, [1.0, 0.0, 0.0, 0.0])
                    # Catch if the map accidentally returns a nested list
                    if isinstance(raw_intent, list) and isinstance(raw_intent[0], list):
                        raw_intent = raw_intent[0]
                    intent_tensor = np.array([raw_intent], dtype=np.float32)

                    for ort_in in self.ort_session.get_inputs():
                        expected_shape = ort_in.shape
                        in_name = ort_in.name
                        in_type = ort_in.type
                        
                        # 1. Identify Intent (Usually 1D or 2D, or named 'intent'/'cmd')
                        if (expected_shape and len(expected_shape) <= 2) or any(k in in_name.lower() for k in ['intent', 'cmd', 'command']):
                            # Match the type ONNX expects (Fallback to float32)
                            if 'int64' in in_type:
                                ort_inputs[in_name] = intent_tensor.astype(np.int64)
                            else:
                                ort_inputs[in_name] = intent_tensor.astype(np.float32)
                                
                        # 2. Identify Image (Usually 4D, or named 'input'/'img'/'x')
                        else:
                            if 'float64' in in_type:
                                ort_inputs[in_name] = img_tensor.astype(np.float64)
                            else:
                                ort_inputs[in_name] = img_tensor.astype(np.float32)

                    try:
                        ort_outs = self.ort_session.run(None, ort_inputs)
                        vel_left = float(ort_outs[0][0][0]) 
                        vel_right = float(ort_outs[0][0][1])
                    except Exception as e:
                        # Log the exact dictionary mapping we attempted vs what ONNX wanted
                        mapping_debug = {k: v.shape for k, v in ort_inputs.items()}
                        expected_debug = {i.name: i.shape for i in self.ort_session.get_inputs()}
                        rospy.logerr(f"CRITICAL ONNX MISMATCH! We sent: {mapping_debug} | ONNX Expected: {expected_debug}")
                        raise e # Re-raise to trigger the throttle block below

                # Publish Motor Commands
                cmd_msg = WheelsCmdStamped()
                cmd_msg.header.stamp = rospy.Time.now()
                cmd_msg.vel_left = float(vel_left)
                cmd_msg.vel_right = float(vel_right)
                self.cmd_pub.publish(cmd_msg)

            # Safely relay everything to the Qt thread for live rendering
            if ui_model_view is not None:
                self.bridge.telemetry_signal.emit(rgb_image.copy(), ui_model_view.copy(), float(vel_left), float(vel_right), self.current_intent)

        except Exception as e:
            rospy.logerr_throttle(2.0, f"Image Processing Error: {e}")
    def update_ui(self, raw_img, model_img, vel_left, vel_right, intent):
        # 1. Render Raw Camera View
        h, w, ch = raw_img.shape
        qt_raw = QImage(raw_img.data, w, h, ch * w, QImage.Format_RGB888)
        self.live_label.setPixmap(QPixmap.fromImage(qt_raw).scaled(self.live_label.width(), self.live_label.height(), Qt.KeepAspectRatio))

        # 2. Render Network Input Image
        if len(model_img.shape) == 2:  # Grayscale Mask
            mh, mw = model_img.shape
            qt_model = QImage(model_img.data, mw, mh, mw, QImage.Format_Indexed8)
        else:  # RGB Cropped Image
            mh, mw, mch = model_img.shape
            qt_model = QImage(model_img.data, mw, mh, mch * mw, QImage.Format_RGB888)
            
        self.model_label.setPixmap(QPixmap.fromImage(qt_model).scaled(self.model_label.width(), self.model_label.height(), Qt.KeepAspectRatio))

        # 3. Update Text Metrics
        if self.is_autonomous_active:
            self.motor_text.setText(f"Left Motor: {vel_left:+.3f}  |  Right Motor: {vel_right:+.3f} [RUNNING: {self.active_backend}]")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_I: self.set_intent("straight")
        elif event.key() == Qt.Key_J: self.set_intent("left")
        elif event.key() == Qt.Key_L: self.set_intent("right")
        elif event.key() == Qt.Key_K: self.set_intent("stop")

    def mousePressEvent(self, event):
        self.setFocus()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autonomous Driver Dashboard Node")
    parser.add_argument("--skip_segmentation", action="store_true", help="Skip YOLO segmentation and view PilotNet cropping directly")
    args, unknown = parser.parse_known_args(rospy.myargv()[1:])
    
    app = QApplication(sys.argv)
    driver_ui = AutonomousDriverUI(skip_segmentation=args.skip_segmentation)
    driver_ui.show()
    
    sys.exit(app.exec_())