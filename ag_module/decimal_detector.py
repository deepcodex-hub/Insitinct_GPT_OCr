import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class DecimalCNN(nn.Module):
    """Small CNN classifier that checks a local patch for a decimal point."""
    def __init__(self):
        super(DecimalCNN, self).__init__()
        # Input: 1 channel (grayscale), e.g., 32x32 patch
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 1) # Binary classification (logits)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

class DecimalDetectorConfig:
    def __init__(self, model_path=None):
        self.model = DecimalCNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.patch_size = 32

        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Could not load decimal detector weights: {e}")

    def detect(self, patch: np.ndarray) -> float:
        """Returns the probability of a decimal existing in the given image patch."""
        # Preprocess patch
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
        patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0) / 255.0
        patch_tensor = patch_tensor.to(self.device)

        with torch.no_grad():
            prob = self.model(patch_tensor).item()
        
        return prob
