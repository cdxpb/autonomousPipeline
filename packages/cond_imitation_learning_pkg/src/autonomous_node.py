import sys
import os
import json
import argparse
import time
import requests
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
from duckietown_msgs.msg import Twist2DStamped, WheelsCmdStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

# PyQt5 UI Components
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, 
                             QHBoxLayout, QWidget, QGroupBox, QPushButton, QComboBox,
                             QCheckBox, QLineEdit)
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
    def __init__(self, approach=2, output_mode=None):
        super().__init__()
        rospy.init_node('autonomous_driver_ui_node', anonymous=False)
        self.veh = os.environ.get('VEHICLE_NAME', 'golduck')
        self.approach = approach
        
        if output_mode is None:
            self.output_mode = "twist" if self.approach == 3 else "wheels"
        else:
            self.output_mode = output_mode

        self.skip_segmentation = (self.approach == 0)
        self.current_intent = "straight"
        
        self.config_path = "/dataset/tuning_config.json"
        self.tuning = self.load_tuning_config()

        # State Flags
        self.is_autonomous_active = False
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Model Paths
        if self.approach == 0:
            self.model_path = "../autonomouspipeline/models/pilotnet/best_model"
        elif self.approach == 2:
            self.model_path = "../autonomouspipeline/models/pilotnet/segPilot_approach2"
            self.yolo_path = "../autonomouspipeline/models/yolo_model/yolo_v1_best.pt"
        elif self.approach == 3:
            self.model_path = "../autonomouspipeline/models/pilotnet/best_model_regNhead"
            self.yolo_path = "../autonomouspipeline/models/yolo_model/yolo_v1_best.pt"
        elif self.approach == 4:
            self.model_path = "../autonomouspipeline/models/pilotnet/segRegNHeadsTemporal"
            self.yolo_path = "../autonomouspipeline/models/yolo_model/yolo_v1_best.pt"
        elif self.approach == 5:
            self.model_path = "../autonomouspipeline/models/pilotnet/best_model_regNheadv2"
            self.yolo_path = "../autonomouspipeline/models/yolo_model/yolo_model.pt"
        elif self.approach == 6:
            self.model_path = "../autonomouspipeline/models/pilotnet/best_model_approach6"
            self.yolo_path = "../autonomouspipeline/models/yolo_model/yolo_v1_best.pt"
        elif self.approach == 7:
            self.model_path = "../autonomouspipeline/models/pilotnet/best_model_approach7"
            self.yolo_path = "../autonomouspipeline/models/yolo_model/yolo_model.pt"

        self.transform = get_eval_transforms()
        self.frame_buffer = []
        self.prev_out1 = 0.0
        self.prev_out2 = 0.0
        
        # Load management
        self.target_fps = 12
        self.last_frame_time = 0.0

        # Thread-safe communication bridge
        self.bridge = TelemetryBridge()
        self.bridge.telemetry_signal.connect(self.update_ui)

        # UI Setup
        self.init_ui()

        # Load YOLO if we are not skipping segmentation and not offloading by default
        self.yolo_session = None
        if not self.skip_segmentation and not self.chk_seg_offload.isChecked():
            self._load_local_yolo()

        # ROS Publishers & Subscribers``
        if self.output_mode == "twist":
            self.cmd_pub = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1, tcp_nodelay=True)
        else:
            self.cmd_pub = rospy.Publisher(f"/{self.veh}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1, tcp_nodelay=True)
        self.image_sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed", CompressedImage, self.image_cb, queue_size=1, buff_size=2**24, tcp_nodelay=True)
        self.intent_sub = rospy.Subscriber(f"/{self.veh}/data_collector/intent", String, self.intent_cb)

    def _load_local_yolo(self):
        if self.yolo_session is None:
            rospy.loginfo("UI: Loading YOLO Semantic Segmentation model locally...")
            self.yolo_session = YOLO(self.yolo_path, task='semantic')

    def load_tuning_config(self):
        default_config = {
            "v_fwd": 0.30, "v_rev": -0.5, 
            "v_bump_a": 0.1, "omega_a": 3.0,
            "v_bump_d": 0.1, "omega_d": 5.0
        }
        # try:
        #     if os.path.exists(self.config_path):
        #         with open(self.config_path, 'r') as f:
        #             saved = json.load(f)
        #             default_config.update(saved)
        # except Exception:
        #     pass
        return default_config

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

        # --- SEGMENTATION & PERFORMANCE BAR ---
        perf_group = QGroupBox("Segmentation Offload & Performance")
        perf_layout = QHBoxLayout()

        self.chk_seg_offload = QCheckBox("Offload Segmentation (Mac Server)")
        self.chk_seg_offload.setChecked(True)
        self.chk_seg_offload.stateChanged.connect(self.toggle_seg_offload)

        self.seg_server_input = QLineEdit()
        self.seg_server_input.setText("http://192.168.1.100:8000") # Default local network IP
        
        self.fps_input = QComboBox()
        self.fps_input.addItems(["Max", "30", "15", "12", "10", "5"])
        self.fps_input.setCurrentText("12")
        self.fps_input.currentTextChanged.connect(self.update_fps_limit)

        perf_layout.addWidget(self.chk_seg_offload)
        perf_layout.addWidget(QLabel("Server URL:"))
        perf_layout.addWidget(self.seg_server_input)
        perf_layout.addWidget(QLabel("Target FPS:"))
        perf_layout.addWidget(self.fps_input)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)

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

        if self.output_mode == "twist":
            self.motor_text = QLabel("v: 0.00  |  omega: 0.00 (IDLE)")
        else:
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
        self.btn_lane_following = QPushButton("Lane Follow [U]")
        self.btn_stop_intent = QPushButton("Stop [K]")

        self.btn_straight.clicked.connect(lambda: self.set_intent("straight"))
        self.btn_left.clicked.connect(lambda: self.set_intent("left"))
        self.btn_right.clicked.connect(lambda: self.set_intent("right"))
        self.btn_lane_following.clicked.connect(lambda: self.set_intent("lane_following"))
        self.btn_stop_intent.clicked.connect(lambda: self.set_intent("stop"))

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

    def toggle_seg_offload(self):
        is_offload = self.chk_seg_offload.isChecked()
        self.seg_server_input.setEnabled(is_offload)
        if not is_offload and not self.skip_segmentation:
            self._load_local_yolo()

    def update_fps_limit(self, text):
        if text == "Max":
            self.target_fps = 1000
        else:
            try:
                self.target_fps = int(text)
            except ValueError:
                self.target_fps = 12

    def action_to_vel(self, action_idx):
        # 0: straight, 1: left, 2: right, 3: stop
        mapping = {
            0: (0.15, 0.2),
            1: (0.15, 0.2),
            2: (0.2, 0.15),
            3: (0.0, 0.0)
        }
        return mapping.get(action_idx, (0.0, 0.0))

    def action_to_twist(self, action_idx):
        # 0: straight, 1: left, 2: right, 3: stop
        if action_idx == 0:
            return self.tuning["v_fwd"], 0.0
        elif action_idx == 1:
            return self.tuning["v_fwd"] + self.tuning["v_bump_a"], self.tuning["omega_a"]
        elif action_idx == 2:
            return self.tuning["v_fwd"] + self.tuning["v_bump_d"], -self.tuning["omega_d"]
        else: # 3: stop
            return 0.0, 0.0

    # ---------------------------------------------------------
    # STATE CONTROL
    # ---------------------------------------------------------
    def start_autonomous(self):
        backend = self.backend_combo.currentText()
        rospy.loginfo(f"Loading Model for Backend: {backend}...")

        try:
            if backend == "PyTorch":
                # Assumes you have a TorchScript exported model (.pt) or you can drop your PilotNet class here
                if self.approach == 3:
                    from pilotnet_regNhead import ConditionalPilotNet
                    self.pt_model = ConditionalPilotNet().to(self.device)
                elif self.approach == 4:
                    from pilotnet_regNCIL_temporal import ConditionalPilotNet
                    self.pt_model = ConditionalPilotNet(num_frames=3).to(self.device)
                elif self.approach == 5:
                    from pilotnet_regNheadv2 import ConditionalPilotNet
                    self.pt_model = ConditionalPilotNet().to(self.device)
                elif self.approach == 6:
                    from pilotnet_classNhead import ConditionalPilotNet
                    self.pt_model = ConditionalPilotNet().to(self.device)
                elif self.approach == 7:
                    from pilotnet_FiLM import ConditionalPilotNetFiLM
                    self.pt_model = ConditionalPilotNetFiLM().to(self.device)
                else:
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
                channels = 3 if self.skip_segmentation else (3 if self.approach == 4 else 1)
                self.d_img_in = cuda.mem_alloc(1 * channels * 112 * 224 * 4)
                self.d_intent_in = cuda.mem_alloc(1 * 4 * 4) 
                
                if self.approach == 4:
                    self.d_vel_in = cuda.mem_alloc(1 * 2 * 4)
                    self.d_output = cuda.mem_alloc(1 * 2 * 4)
                    self.bindings = [int(self.d_img_in), int(self.d_vel_in), int(self.d_intent_in), int(self.d_output)]
                else:
                    out_size = 4 if self.approach == 3 else (3 if self.approach == 6 else 2)
                    self.d_output = cuda.mem_alloc(1 * out_size * 4)
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
        if self.output_mode == "twist":
            cmd_msg = Twist2DStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.v = 0.0
            cmd_msg.omega = 0.0
            self.cmd_pub.publish(cmd_msg)
            self.motor_text.setText("v: 0.00  |  omega: 0.00 (STOPPED)")
        else:
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
        if intent_str in ["straight", "left", "right", "stop", "lane_following"]:
            self.current_intent = intent_str
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

    # ---------------------------------------------------------
    # MAIN ROS INFERENCE LOOP
    # ---------------------------------------------------------
    def setup_inference_engine(self):
        if not self.skip_segmentation and not self.chk_seg_offload.isChecked():
            self._load_local_yolo()

        rospy.loginfo(f"UI: Loading PilotNet model from {self.model_path}...")
        if TRT_AVAILABLE:
            self.logger = trt.Logger(trt.Logger.WARNING)
            with open(f"{self.model_path}.engine", "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            
            channels = 3 if self.skip_segmentation else (3 if self.approach == 4 else 1)
            self.d_img_in = cuda.mem_alloc(1 * channels * 112 * 224 * 4)
            self.d_intent_in = cuda.mem_alloc(1 * 4 * 4) 
            
            if self.approach == 4:
                self.d_vel_in = cuda.mem_alloc(1 * 2 * 4)
                self.d_output = cuda.mem_alloc(1 * 2 * 4)
                self.bindings = [int(self.d_img_in), int(self.d_vel_in), int(self.d_intent_in), int(self.d_output)]
            else:
                out_size = 4 if self.approach == 3 else (3 if self.approach == 6 else 2)
                self.d_output = cuda.mem_alloc(1 * out_size * 4)
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
                cropped_img = crop_image(pil_image)
                
                # Frame Rate Limiter
                current_time = time.time()
                if (current_time - self.last_frame_time) < (1.0 / self.target_fps):
                    return
                self.last_frame_time = current_time

                if self.chk_seg_offload.isChecked():
                    # Offload to external server
                    server_url = self.seg_server_input.text().strip()
                    img_to_send = cropped_img if self.approach in [5, 6, 7] else pil_image
                    
                    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(np.array(img_to_send), cv2.COLOR_RGB2BGR))
                    if is_success:
                        try:
                            # Send synchronous request
                            response = requests.post(
                                f"{server_url}/predict/segmentation",
                                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                                timeout=2.0
                            )
                            if response.status_code == 200:
                                mask_bytes = response.content
                                mask_np_arr = np.frombuffer(mask_bytes, np.uint8)
                                mask_cv = cv2.imdecode(mask_np_arr, cv2.IMREAD_GRAYSCALE)
                                if mask_cv is not None:
                                    mask_pil = Image.fromarray(mask_cv)
                                else:
                                    raise ValueError("Failed to decode received mask")
                            else:
                                rospy.logwarn_throttle(2.0, f"Server returned {response.status_code}")
                                return
                        except requests.exceptions.RequestException as e:
                            rospy.logwarn_throttle(2.0, f"Segmentation Server connection failed: {e}")
                            return
                else:
                    # Local Inference
                    with torch.no_grad():
                        if self.approach in [5, 6, 7]:
                            yolo_results = self.yolo_session(cropped_img, verbose=False, device=self.device.type)
                        else:
                            yolo_results = self.yolo_session(pil_image, verbose=False, device=self.device.type)

                    mask = yolo_results[0].semantic_mask.data.cpu()
                    mask_pil = Image.fromarray(mask.numpy())

                if self.approach not in [5, 6, 7]:
                    mask_pil = crop_image(mask_pil)

                resized_mask_pil = mask_pil.resize((224, 112), Image.NEAREST)

                if self.approach == 4:
                    img_tensor = np.array(resized_mask_pil, dtype=np.float32)[np.newaxis, np.newaxis, ...]
                else:
                    img_tensor = TF.to_tensor(resized_mask_pil).unsqueeze(0).numpy()
                    
                ui_model_view = (np.array(resized_mask_pil) * 85).astype(np.uint8)

            if self.approach == 4:
                if len(self.frame_buffer) == 0:
                    self.frame_buffer = [img_tensor] * 3
                else:
                    self.frame_buffer.pop(0)
                    self.frame_buffer.append(img_tensor)
                img_tensor = np.concatenate(self.frame_buffer, axis=1)

            out1, out2 = 0.0, 0.0

            # 3. Inference Gate (Only runs if "START" is active)
            if self.is_autonomous_active:
                intent_tensor = np.array([INTENT_MAP.get(self.current_intent, [1.0, 0.0, 0.0, 0.0])], dtype=np.float32)

                if self.active_backend == "PyTorch":
                    img_t = torch.from_numpy(img_tensor).to(self.device)
                    intent_t = torch.from_numpy(intent_tensor).to(self.device)
                    
                    if self.approach == 4:
                        vel_t = torch.tensor([[self.prev_out1, self.prev_out2]], dtype=torch.float32).to(self.device)
                        with torch.no_grad():
                            output = self.pt_model(img_t, vel_t, intent_t)
                        out1, out2 = output[0][0].item(), output[0][1].item()
                    else:
                        with torch.no_grad():
                            output = self.pt_model(img_t, intent_t)
                        if self.approach == 3:
                            out1, out2 = output[0][0].item(), output[0][1].item()
                            # pred_action = torch.argmax(output, dim=1).cpu().item()
                            # if self.output_mode == "twist":
                            #     out1, out2 = self.action_to_twist(pred_action)
                            # else:
                            #     out1, out2 = self.action_to_vel(pred_action)
                        elif self.approach == 6:
                            pred_action = torch.argmax(output, dim=1).cpu().item()
                            if self.output_mode == "twist":
                                out1, out2 = self.action_to_twist(pred_action)
                            else:
                                out1, out2 = self.action_to_vel(pred_action)
                        else:
                            out1, out2 = output[0][0].item(), output[0][1].item()

                elif self.active_backend == "TensorRT":
                    cuda.memcpy_htod(self.d_img_in, img_tensor)
                    cuda.memcpy_htod(self.d_intent_in, intent_tensor)
                    if self.approach == 4:
                        vel_tensor = np.array([[self.prev_out1, self.prev_out2]], dtype=np.float32)
                        cuda.memcpy_htod(self.d_vel_in, vel_tensor)
                    self.context.execute_v2(bindings=self.bindings)
                    h_output = np.empty((1, 4 if self.approach == 3 else (3 if self.approach == 6 else 2)), dtype=np.float32)
                    cuda.memcpy_dtoh(h_output, self.d_output)
                    if self.approach in [3, 6]:
                        pred_action = np.argmax(h_output[0])
                        if self.output_mode == "twist":
                            out1, out2 = self.action_to_twist(pred_action)
                        else:
                            out1, out2 = self.action_to_vel(pred_action)
                    else:
                        out1, out2 = h_output[0][0], h_output[0][1]

                elif self.active_backend == "ONNX Runtime":
                    ort_inputs = {}
                    
                    # Ensure intent is strictly a 2D array: Shape (1, 4)
                    raw_intent = INTENT_MAP.get(self.current_intent, [1.0, 0.0, 0.0, 0.0])
                    # Catch if the map accidentally returns a nested list
                    if isinstance(raw_intent, list) and isinstance(raw_intent[0], list):
                        raw_intent = raw_intent[0]
                    intent_tensor = np.array([raw_intent], dtype=np.float32)
                    
                    vel_tensor = None
                    if self.approach == 4:
                        vel_tensor = np.array([[self.prev_out1, self.prev_out2]], dtype=np.float32)

                    for ort_in in self.ort_session.get_inputs():
                        expected_shape = ort_in.shape
                        in_name = ort_in.name
                        in_type = ort_in.type
                        
                        # 1. Identify Intent (Usually 1D or 2D, or named 'intent'/'cmd')
                        if (expected_shape and len(expected_shape) <= 2 and expected_shape[-1] == 4) or any(k in in_name.lower() for k in ['intent', 'cmd', 'command']):
                            # Match the type ONNX expects (Fallback to float32)
                            if 'int64' in in_type:
                                ort_inputs[in_name] = intent_tensor.astype(np.int64)
                            else:
                                ort_inputs[in_name] = intent_tensor.astype(np.float32)
                                
                        # 2. Identify Velocities (Approach 4)
                        elif self.approach == 4 and ((expected_shape and len(expected_shape) == 2 and expected_shape[-1] == 2) or any(k in in_name.lower() for k in ['vel', 'state'])):
                            if 'int64' in in_type:
                                ort_inputs[in_name] = vel_tensor.astype(np.int64)
                            else:
                                ort_inputs[in_name] = vel_tensor.astype(np.float32)
                                
                        # 3. Identify Image (Usually 4D, or named 'input'/'img'/'x')
                        else:
                            if 'float64' in in_type:
                                ort_inputs[in_name] = img_tensor.astype(np.float64)
                            else:
                                ort_inputs[in_name] = img_tensor.astype(np.float32)

                    try:
                        ort_outs = self.ort_session.run(None, ort_inputs)
                        if self.approach in [3, 6]:
                            pred_action = np.argmax(ort_outs[0][0])
                            if self.output_mode == "twist":
                                out1, out2 = self.action_to_twist(pred_action)
                            else:
                                out1, out2 = self.action_to_vel(pred_action)
                        else:
                            out1 = float(ort_outs[0][0][0]) 
                            out2 = float(ort_outs[0][0][1])
                    except Exception as e:
                        # Log the exact dictionary mapping we attempted vs what ONNX wanted
                        mapping_debug = {k: v.shape for k, v in ort_inputs.items()}
                        expected_debug = {i.name: i.shape for i in self.ort_session.get_inputs()}
                        rospy.logerr(f"CRITICAL ONNX MISMATCH! We sent: {mapping_debug} | ONNX Expected: {expected_debug}")
                        raise e # Re-raise to trigger the throttle block below

                if self.approach == 4:
                    self.prev_out1 = out1
                    self.prev_out2 = out2

                # Publish Motor Commands
                if self.output_mode == "twist":
                    cmd_msg = Twist2DStamped()
                    cmd_msg.header.stamp = rospy.Time.now()
                    cmd_msg.v = float(out1)
                    cmd_msg.omega = float(out2)
                    self.cmd_pub.publish(cmd_msg)
                else:
                    cmd_msg = WheelsCmdStamped()
                    cmd_msg.header.stamp = rospy.Time.now()
                    cmd_msg.vel_left = float(out1)
                    cmd_msg.vel_right = float(out2)
                    self.cmd_pub.publish(cmd_msg)

            # Safely relay everything to the Qt thread for live rendering
            if ui_model_view is not None:
                self.bridge.telemetry_signal.emit(rgb_image.copy(), ui_model_view.copy(), float(out1), float(out2), self.current_intent)

        except Exception as e:
            rospy.logerr_throttle(2.0, f"Image Processing Error: {e}")
    def update_ui(self, raw_img, model_img, out1, out2, intent):
        # 1. Render Raw Camera View
        h, w, ch = raw_img.shape
        qt_raw = QImage(raw_img.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self.live_label.setPixmap(QPixmap.fromImage(qt_raw).scaled(self.live_label.width(), self.live_label.height(), Qt.KeepAspectRatio))

        mh, mw = model_img.shape[:2]
        if len(model_img.shape) == 2:
            # Grayscale Mask
            qt_model = QImage(model_img.tobytes(), mw, mh, mw, QImage.Format_Indexed8)
        else:
            mch = model_img.shape[2]
            qt_model = QImage(model_img.tobytes(), mw, mh, mch * mw, QImage.Format_RGB888)
            
        self.model_label.setPixmap(QPixmap.fromImage(qt_model).scaled(self.model_label.width(), self.model_label.height(), Qt.KeepAspectRatio))

        # 3. Update Text Metrics
        if self.is_autonomous_active:
            if self.output_mode == "twist":
                self.motor_text.setText(f"v: {out1:+.3f}  |  omega: {out2:+.3f} [RUNNING: {self.active_backend}]")
            else:
                self.motor_text.setText(f"Left Motor: {out1:+.3f}  |  Right Motor: {out2:+.3f} [RUNNING: {self.active_backend}]")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_I: self.set_intent("straight")
        elif event.key() == Qt.Key_J: self.set_intent("left")
        elif event.key() == Qt.Key_L: self.set_intent("right")
        elif event.key() == Qt.Key_K: self.set_intent("stop")
        elif event.key() == Qt.Key_U: self.set_intent("lane_following")

    def mousePressEvent(self, event):
        self.setFocus()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autonomous Driver Dashboard Node")
    parser.add_argument("--approach", type=int, choices=[0, 2, 3, 4, 5, 6, 7], default=2, help="0: skip segmentation, 2: segmentation + regression, 3: segmentation + classification, 4: segmentation + regression temporal, 5: segmentation + regression 4 heads (regNheadv2), 6: segmentation + classification 4 heads (classNhead), 7: segmentation + regression FiLM (FiLM)")
    parser.add_argument("--skip_segmentation", action="store_true", help="Deprecated. Use --approach 0 instead.")
    parser.add_argument("--output_mode", type=str, choices=["twist", "wheels"], default=None, help="Output mode for driving commands. Defaults to wheels for approach 0/2, twist for approach 3.")
    args, unknown = parser.parse_known_args(rospy.myargv()[1:])
    
    approach = args.approach
    if args.skip_segmentation:
        approach = 0
    
    app = QApplication(sys.argv)
    driver_ui = AutonomousDriverUI(approach=approach, output_mode=args.output_mode)
    driver_ui.show()
    
    sys.exit(app.exec_())