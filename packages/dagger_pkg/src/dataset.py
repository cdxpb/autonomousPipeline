import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import crop_image, get_train_transforms, INTENT_MAP

class DuckieTownDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        try:
            from ultralytics import YOLO
            # Assuming model paths inside the docker container
            self.yolo = YOLO("/models/yolo_model/yolo_v1_best.onnx", task="semantic")
        except ImportError:
            self.yolo = None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]['image_filename'])
        image = Image.open(img_name).convert('RGB')
        image = crop_image(image)
        
        if self.yolo:
            import torch
            with torch.no_grad():
                results = self.yolo(image, verbose=False)
            mask = results[0].semantic_mask.data.cpu().numpy()
            image = Image.fromarray(mask).resize((224, 112), Image.NEAREST)

        vel_left = self.dataframe.iloc[idx]['vel_left']
        vel_right = self.dataframe.iloc[idx]['vel_right']
        intent_str = self.dataframe.iloc[idx]['intent']

        # Horizontal Flip Trap Prevention
        if self.transform == get_train_transforms() and random.random() > 0.5:
            import torchvision.transforms.functional as TF
            image = TF.hflip(image)
            target_vel = torch.tensor([vel_right, vel_left], dtype=torch.float32)
            
            # If flipping horizontally, a left turn becomes a right turn!
            if intent_str == "left": intent_str = "right"
            elif intent_str == "right": intent_str = "left"
        else:
            target_vel = torch.tensor([vel_left, vel_right], dtype=torch.float32)

        # One-hot encode the intent
        intent_tensor = torch.tensor(INTENT_MAP.get(intent_str, [1.0, 0.0, 0.0, 0.0]), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, intent_tensor, target_vel