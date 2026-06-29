import torch
from ultralytics import YOLO
import os
from PIL import Image


def crop_image(pil_image):
    width, height = pil_image.size
    # Ensure cropping does not result in negative height
    crop_height = height - CROP_TOP_ROWS
    if crop_height <= 0:
        # If the image is too short, just return it or handle as an error
        print(f"Warning: Image height {height} is too small for CROP_TOP_ROWS {CROP_TOP_ROWS}. Not cropping.")
        return pil_image
    return pil_image.crop((0, CROP_TOP_ROWS, width, height))


def load_yolo(model_folder: str):
    """
    Loads the YOLO segmentation model from the specified folder for inference.
    Includes fallback logic for common YOLO model names (best.pt, last.pt).
    """
    seg_model_filename = 'segModel.pt'
    seg_model_path = os.path.join(model_folder, seg_model_filename)

    if not os.path.exists(seg_model_path):
        fallback_paths = [
            os.path.join(model_folder, 'best.pt'),
            os.path.join(model_folder, 'last.pt'),
        ]
        found_fallback = False
        for fp in fallback_paths:
            if os.path.exists(fp):
                seg_model_path = fp
                seg_model_filename = os.path.basename(fp)
                print(f"Using fallback YOLO model: {seg_model_path}")
                found_fallback = True
                break

        if not found_fallback:
            raise FileNotFoundError(f"No YOLO segmentation model found at {seg_model_path} or common fallback paths in {model_folder}.")

    print(f"Loading YOLO segmentation model from: {seg_model_path}")
    model = YOLO(seg_model_path) # Load as YOLO object
    model.eval() # Set to evaluation mode to freeze weights and disable dropout/batchnorm updates
    return model

def load_pilotnet(model_path: str, device: torch.device):
    """
    Loads the ConditionalPilotNet model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ConditionalPilotNet model not found at {model_path}.")

    print(f"Loading ConditionalPilotNet from {model_path}")
    model = ConditionalPilotNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode
    return model


def run_inference_pipeline(raw_image: Image.Image, intent_str: str, seg_model, pilotnet_model) -> int:
    """
    Runs the full inference pipeline using the loaded segmentation model and PilotNet model.

    Args:
        raw_image (PIL.Image.Image): The input image to process.
        intent_str (str): The intent as a string ('straight', 'left', 'right', 'stop').
        seg_model: The loaded YOLO segmentation model.
        pilotnet_model: The loaded ConditionalPilotNet model.

    Returns:
        int: The predicted action (0: straight, 1: left, 2: right, 3: other).
    """
    global device, INTENT_MAP, crop_image, get_eval_mask_transforms, INPUT_SHAPE, CROP_TOP_ROWS

    # Ensure models are in evaluation mode
    seg_model.eval()
    pilotnet_model.eval()

    with torch.no_grad():
        # 1. Get Segmentation Mask from YOLO
        yolo_results = seg_model(raw_image, verbose=False)

        if not yolo_results or not hasattr(yolo_results[0], 'semantic_mask'):
            print(f"Warning: YOLO failed for image. Creating a dummy mask.")
            # Fallback to a black dummy mask if YOLO fails
            dummy_mask = torch.zeros(INPUT_SHAPE[0] + CROP_TOP_ROWS, INPUT_SHAPE[1], dtype=torch.uint8) # Original image size to match crop_image expected input
            segmentation_mask_tensor = dummy_mask
        else:
            segmentation_mask_tensor = yolo_results[0].semantic_mask.data.cpu() # [H, W], uint8

        mask_pil = Image.fromarray(segmentation_mask_tensor.numpy())

        # 2. Crop the Mask
        cropped_mask_pil = crop_image(mask_pil)

        # 3. Apply Final Transforms to the Mask
        processed_mask_tensor = get_eval_mask_transforms()(cropped_mask_pil)
        image_input = processed_mask_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

        # 4. Prepare Intent Tensor
        intent_tensor = torch.tensor(INTENT_MAP.get(intent_str, [1.0, 0.0, 0.0, 0.0]), dtype=torch.float32)
        intent_input = intent_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

        # 5. Make Prediction with ConditionalPilotNet
        output_logits = pilotnet_model(image_input, intent_input)
        predicted_action = torch.argmax(output_logits, dim=1).cpu().item()

    return output_logits, predicted_action