import cv2
import numpy as np

class AGMEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.2):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.gamma = gamma
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Applies CLAHE to a BGR image."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def apply_gamma(self, image: np.ndarray, gamma: float = None) -> np.ndarray:
        """Applies gamma correction."""
        g = gamma if gamma is not None else self.gamma
        inv_gamma = 1.0 / g
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def enhance(self, image: np.ndarray) -> dict:
        """Full enhancement pipeline for a single image."""
        enhanced = self.apply_clahe(image)
        enhanced = self.apply_gamma(enhanced)
        return {
            "enhanced_image": enhanced,
            "debug_artifacts": {
                "clahe_applied": True,
                "gamma": self.gamma
            }
        }
