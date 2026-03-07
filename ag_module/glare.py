import cv2
import numpy as np

class GlareProcessor:
    def __init__(self, threshold=240):
        self.threshold = threshold

    def detect_glare(self, image: np.ndarray):
        """Detects high-intensity regions likely to be glare."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def inpaint_glare(self, image: np.ndarray, mask: np.ndarray):
        """Inpaints glare regions using Telea algorithm."""
        if np.sum(mask) == 0:
            return image
        
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return inpainted

    def process(self, image: np.ndarray):
        mask = self.detect_glare(image)
        inpainted = self.inpaint_glare(image, mask)
        return {
            "inpainted_image": inpainted,
            "glare_mask": mask
        }
