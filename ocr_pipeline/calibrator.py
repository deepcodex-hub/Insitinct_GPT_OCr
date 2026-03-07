import numpy as np
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import pickle
import os

class ModelCalibrator:
    """Pipenine for calibrating OCR confidences using Temperature Scaling + Isotonic Regression."""
    def __init__(self, model_dir="data/calibration"):
        self.model_dir = model_dir
        self.iso_reg = IsotonicRegression(out_of_bounds='clip')
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Default temp
        
        # Load fitted isotonic model if exists
        self.iso_path = os.path.join(model_dir, "isotonic_reg.pkl")
        if os.path.exists(self.iso_path):
            try:
                with open(self.iso_path, 'rb') as f:
                    self.iso_reg = pickle.load(f)
                self.is_fitted = True
            except Exception:
                self.is_fitted = False
        else:
            self.is_fitted = False

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Perform temperature scaling on logits."""
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def calibrate(self, raw_confidence: float) -> float:
        """Applies nonparametric calibration (Isotonic Regression) to a raw [0,1] confidence score."""
        if not self.is_fitted:
            # Fallback if not calibrated: just return the raw confidence
            # In a real deployed scenario, this would have been fitted during Stage 9.
            return raw_confidence
            
        # Isotonic regression expects a 1D array of shape (n_samples,)
        # It maps the uncalibrated score to a strictly calibrated empirical probability
        calibrated_prob = self.iso_reg.transform([raw_confidence])[0]
        return float(calibrated_prob)

    def fit(self, val_confidences: np.ndarray, val_labels: np.ndarray):
        """Fits the isotonic regressor. Expected to run during the Stage 9 calibration phase."""
        os.makedirs(self.model_dir, exist_ok=True)
        self.iso_reg.fit(val_confidences, val_labels)
        self.is_fitted = True
        with open(self.iso_path, 'wb') as f:
            pickle.dump(self.iso_reg, f)
