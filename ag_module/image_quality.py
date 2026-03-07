import cv2
import numpy as np

def detect_blur(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Detects blur using the Variance of Laplacian method."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def detect_glare(image: np.ndarray, threshold_ratio: float = 0.05) -> bool:
    """Detects if a significant portion of the image is glaring (overexposed specular)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Threshold for very bright pixels
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare_ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])
    return glare_ratio > threshold_ratio

def detect_tilt(image: np.ndarray, threshold_deg: float = 10.0) -> float:
    """Estimates the tilt of the main text/LCD block. Returns angle in degrees."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return 0.0
        
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep angles that represent roughly horizontal lines
        if -45 < angle < 45:
            angles.append(angle)
            
    if not angles:
        return 0.0
        
    median_angle = np.median(angles)
    return float(median_angle)

def analyze_image_quality(image: np.ndarray) -> dict:
    """Runs a full suite of image quality heuristics."""
    blur_flag = detect_blur(image)
    glare_flag = detect_glare(image)
    tilt_deg = detect_tilt(image)
    
    not_legible = blur_flag or (abs(tilt_deg) > 40.0) # extreme heuristics
    
    return {
        "blur": blur_flag,
        "glare": glare_flag,
        "tilt_deg": tilt_deg,
        "not_legible": not_legible,
        "not_legible_status": "not_legible_to_human" if not_legible else None
    }
