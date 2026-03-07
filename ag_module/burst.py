import cv2
import numpy as np

class MultiFrameProcessor:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align_frames(self, target_frame: np.ndarray, reference_frame: np.ndarray):
        """Aligns target_frame to reference_frame using global homography."""
        kp1, des1 = self.orb.detectAndCompute(reference_frame, None)
        kp2, des2 = self.orb.detectAndCompute(target_frame, None)

        if des1 is None or des2 is None:
            return target_frame

        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:50]
        if len(good_matches) < 4:
            return target_frame

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        h, w = reference_frame.shape[:2]
        aligned = cv2.warpPerspective(target_frame, M, (w, h))
        return aligned

    def fuse_frames(self, frames: list):
        """Fuses multiple aligned frames using median filtering to reduce noise/glare."""
        if not frames:
            return None
        
        # Stack frames
        stack = np.stack(frames, axis=0)
        # Compute median across frames
        fused = np.median(stack, axis=0).astype(np.uint8)
        return fused

    def process_burst(self, frames: list):
        """Full burst processing pipeline."""
        if not frames:
            return None
        
        reference = frames[0]
        aligned_frames = [reference]
        
        for i in range(1, len(frames)):
            aligned = self.align_frames(frames[i], reference)
            aligned_frames.append(aligned)
            
        fused = self.fuse_frames(aligned_frames)
        return fused
