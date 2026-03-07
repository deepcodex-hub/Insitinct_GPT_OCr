import os
import sys
# Force CPU and disable dynamo upfront
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
os.environ["TRITON_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ag_module.decimal_detector import DecimalCNN

# Model is already imported above via explicit sys path

class SyntheticDecimalDataset(Dataset):
    """
    Since we don't have a labeled patch-level dataset for decimals,
    we'll generate a synthetic one using OpenCV to draw digits and decimals 
    with various noise, blur, and contrast profiles to mimic LCD artifacts.
    """
    def __init__(self, num_samples=5000, patch_size=(32, 32)):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        data = []
        labels = []
        for _ in range(self.num_samples):
            # Base background (gray LCD color)
            base_bg = random.randint(150, 220)
            img = np.full((self.patch_size[1], self.patch_size[0]), base_bg, dtype=np.uint8)
            
            has_decimal = random.choice([0, 1])
            labels.append(has_decimal)
            
            # Add some random digits
            num_digits = random.randint(1, 3)
            for _ in range(num_digits):
                digit = str(random.randint(0, 9))
                x = random.randint(0, self.patch_size[0] - 10)
                y = random.randint(15, self.patch_size[1] - 5)
                cv2.putText(img, digit, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                            random.uniform(0.3, 0.6), (0, 0, 0), 1)

            if has_decimal:
                # Draw decimal point
                cx = random.randint(5, self.patch_size[0] - 5)
                cy = random.randint(self.patch_size[1] - 10, self.patch_size[1] - 2)
                radius = random.randint(1, 2)
                cv2.circle(img, (cx, cy), radius, (0, 0, 0), -1)

            # Add noise and blur to simulate bad LCD crops
            noise = np.random.normal(0, random.uniform(5, 15), img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
            
            if random.random() > 0.5:
                img = cv2.GaussianBlur(img, (3, 3), 0)

            # Normalize to tensor
            tensor = torch.from_numpy(img).float() / 255.0
            tensor = tensor.unsqueeze(0) # Add channel dim
            
            data.append(tensor)
            
        return data, labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


def train_decimal_detector():
    print("Initializing Synthetic Dataset...")
    train_dataset = SyntheticDecimalDataset(num_samples=8000)
    val_dataset = SyntheticDecimalDataset(num_samples=2000)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    model = DecimalCNN()
    
    # We must explicitly move the model to CPU since we disabled CUDA
    device = torch.device('cpu')
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    print(f"Starting Training for {epochs} epochs...")
    
    best_loss = float('inf')
    output_dir = Path(r"D:\GPT_instinct\models\weights")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
        val_loss /= len(val_loader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = output_dir / "decimal_cnn_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model to {save_path}")

if __name__ == "__main__":
    train_decimal_detector()
