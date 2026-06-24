import os
import torch
from pilotnet import ConditionalPilotNet

def compile_to_onnx(data_path):
    """Loads saved PyTorch weights and exports them to ONNX format."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalPilotNet().to(device)
    model.eval() # Ensure dropout layers are disabled for export
    
    # Define exact dummy inputs expected by the model
    dummy_img = torch.randn(1, 3, 112, 224).to(device)
    dummy_intent = torch.randn(1, 4).to(device)

    # Compile Best Model
    best_pth = os.path.join(data_path, "best_model.pth")
    if os.path.exists(best_pth):
        model.load_state_dict(torch.load(best_pth, map_location=device))
        torch.onnx.export(
            model, (dummy_img, dummy_intent), os.path.join(data_path, "best_model.onnx"),
            export_params=True, opset_version=11,          
            input_names=['image_input', 'intent_input'], output_names=['wheel_velocities']    
        )
        print("SUCCESS: Exported best_model.onnx")

    # Compile Last Model
    last_pth = os.path.join(data_path, "last_model.pth")
    if os.path.exists(last_pth):
        model.load_state_dict(torch.load(last_pth, map_location=device))
        torch.onnx.export(
            model, (dummy_img, dummy_intent), os.path.join(data_path, "last_model.onnx"),
            export_params=True, opset_version=11,          
            input_names=['image_input', 'intent_input'], output_names=['wheel_velocities']    
        )
        print("SUCCESS: Exported last_model.onnx")