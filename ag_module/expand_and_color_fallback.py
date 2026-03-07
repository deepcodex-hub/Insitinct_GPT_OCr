import cv2
import numpy as np

def expand_bbox(bbox, img_w, img_h, scale=0.15):
    """
    Expands the bounding box by a given scale factor.
    bbox format: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # Calculate expansion amounts
    dx = int(w * scale)
    dy = int(h * scale)
    
    # Expand and clamp to image boundaries
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(img_w, x2 + dx)
    ny2 = min(img_h, y2 + dy)
    
    return [nx1, ny1, nx2, ny2]

def get_color_fallback_crop(image, lower_hsv=None, upper_hsv=None):
    """
    HSV green-screen detection to find the largest green/LCD rectangle
    and use it as the crop.
    Returns:
        crop: The cropped image (or None if not found)
        mask: The color fallback mask for debugging
        bbox: The [x1, y1, x2, y2] bounding box
    """
    # Default HSV thresholds for standard green/LCD colored displays
    if lower_hsv is None:
        lower_hsv = np.array([35, 20, 20])
    if upper_hsv is None:
        upper_hsv = np.array([85, 255, 255])
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask, None
        
    # Find largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 500: # Minimum size threshold
        return None, mask, None
        
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Expand it slightly to be safe
    img_h, img_w = image.shape[:2]
    bbox = expand_bbox([x, y, x + w, y + h], img_w, img_h, scale=0.10)
    
    nx1, ny1, nx2, ny2 = bbox
    crop = image[ny1:ny2, nx1:nx2]
    
    return crop, mask, bbox
