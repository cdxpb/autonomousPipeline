#!/usr/bin/env python3
import os
import shutil
import pandas as pd
import torch
import torch.optim as optim

from pilotnet import ConditionalPilotNet
from dataset import DuckieTownDataset
from utils import get_train_transforms

def aggregate_and_train(base_dataset_dir, dagger_run_dir, model_save_dir):
    print("=== DAgger Aggregation Phase ===")
    
    # Paths
    base_csv = os.path.join(base_dataset_dir, "log.csv")
    dagger_csv = os.path.join(dagger_run_dir, "dagger_log.csv")
    
    base_img_dir = os.path.join(base_dataset_dir, "images")
    dagger_img_dir = os.path.join(dagger_run_dir, "images")

    # Append CSV data
    df_base = pd.read_csv(base_csv)
    df_dagger = pd.read_csv(dagger_csv)
    
    df_combined = pd.concat([df_base, df_dagger], ignore_index=True)
    df_combined.to_csv(base_csv, index=False)
    
    # Copy images to main directory
    print(f"Aggregating {len(df_dagger)} new correction frames...")
    for filename in df_dagger['image_filename']:
        src = os.path.join(dagger_img_dir, filename)
        dst = os.path.join(base_img_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    print("Aggregation complete. Starting fine-tuning...")

    # Fine-tune model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalPilotNet().to(device)
    
    # Load previous best weights to fine-tune
    previous_weights = os.path.join(model_save_dir, "best_model.pth")
    if os.path.exists(previous_weights):
        model.load_state_dict(torch.load(previous_weights, map_location=device))
        print("Loaded previous model weights for fine-tuning.")
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Lower learning rate for fine-tuning
    criterion = torch.nn.MSELoss()
    
    dataset = DuckieTownDataset(df_combined, base_img_dir, transform=get_train_transforms())
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    epochs = 3 # Only fine-tuning
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, intents, targets in loader:
            images, intents, targets = images.to(device), intents.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, intents)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Fine-Tuning Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader):.4f}")

    # Save newly tuned model
    torch.save(model.state_dict(), previous_weights)
    print("DAgger Fine-Tuning Complete! Model updated.")

if __name__ == '__main__':
    # Dynamic paths based on system setup
    BASE_DIR = "/dataset/cil_dataset_master"
    LATEST_DAGGER_DIR = "/dataset/dagger_aggregate_20231024-120000"
    MODEL_DIR = "/models/pilotnet"
    
    aggregate_and_train(BASE_DIR, LATEST_DAGGER_DIR, MODEL_DIR)