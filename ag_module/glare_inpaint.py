import cv2
import numpy as np
import torch
import torch.nn as nn

class UNetInpainter(nn.Module):
    """Simple U-Net architecture for inpainting (stubbed for MVP)."""
    def __init__(self):
        super().__init__()
        # Encoder/Decoder definition goes here
        self.conv = nn.Conv2d(4, 3, kernel_size=3, padding=1) # 3 channels + 1 mask

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=1)
        return torch.sigmoid(self.conv(inp))

class GlareInpainter:
    """Detects specular highlights (glare) and inpaints them."""
    def __init__(self, use_gpu=True, unet_weights=None):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = UNetInpainter().to(self.device)
        if unet_weights:
            self.model.load_state_dict(torch.load(unet_weights, map_location=self.device))
        self.model.eval()

    def create_glare_mask(self, image: np.ndarray, thresh=220):
        """Detect overexposed regions."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        # Expand mask slightly to catch borders of glare
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask

    def inpaint(self, image: np.ndarray):
        """Inpaint glare. Fast fallback uses cv2.inpaint, otherwise U-Net."""
        mask = self.create_glare_mask(image)
        if cv2.countNonZero(mask) == 0:
            return image
            
        # Fallback to Telea inpainting if U-Net not fully trained/configured
        # This is fast and robust for small specular regions
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return inpainted
