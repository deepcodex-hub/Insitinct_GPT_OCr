import numpy as np
import cv2

class TemporalFuser:
    """Robust temporal median fusion with confidence masking."""
    def __init__(self):
        pass
        
    def fuse(self, aligned_frames: list):
        """
        Fuses multiple aligned frames.
        Computes temporal median and an optional confidence/variance map.
        """
        if not aligned_frames:
            return None
        if len(aligned_frames) == 1:
            return aligned_frames[0]
            
        stack = np.stack(aligned_frames, axis=0)
        
        # Temporal median is robust to transient glares and noise
        fused = np.median(stack, axis=0).astype(np.uint8)
        
        # Calculate per-pixel variance (lower variance = higher confidence)
        # variance_map = np.var(stack, axis=0)
        # confidence_map = np.exp(-variance_map / 255.0)  # normalized confidence
        
        return fused
