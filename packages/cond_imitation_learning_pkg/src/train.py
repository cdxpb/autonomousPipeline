import os
import time
import torch
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

from pilotnet import ConditionalPilotNet
from dataset import DuckieTownDataset
from utils import get_train_transforms, get_eval_transforms

def train_model(data_path, epochs=10, batch_size=64, learning_rate=0.001):
    """Handles Data Loading, Training, Validation, and Saving PyTorch weights."""
    images_folder = os.path.join(data_path, 'images')
    logs_filepath = os.path.join(data_path, 'log.csv')
    
    print("Loading dataset...")
    logs_df = pd.read_csv(logs_filepath)
    train_df, val_df = train_test_split(logs_df, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)} | Validation set: {len(val_df)}")

    train_dataset = DuckieTownDataset(train_df, images_folder, transform=get_train_transforms())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = DuckieTownDataset(val_df, images_folder, transform=get_eval_transforms())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalPilotNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print(f"Training Started on device: {device}...")
    best_val_loss = float('inf')
    best_pth_path = os.path.join(data_path, "best_model.pth")
    last_pth_path = os.path.join(data_path, "last_model.pth")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        running_train_loss = 0.0
        for images, intents, targets in train_loader:
            images, intents, targets = images.to(device), intents.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, intents)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            
        avg_train_loss = running_train_loss / len(train_loader.dataset)

        # --- VALIDATE ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, intents, targets in val_loader:
                images, intents, targets = images.to(device), intents.to(device), targets.to(device)
                outputs = model(images, intents)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * images.size(0)
                
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_duration:.2f}s")

        # --- SAVE BEST ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_pth_path)
            print(f"   [*] New best validation loss! Saved best_model.pth")

    # --- SAVE LAST ---
    torch.save(model.state_dict(), last_pth_path)
    print(f"\nTraining Complete. Saved last_model.pth")