import cv2
import numpy as np

class DewarpProcessor:
    def __init__(self):
        pass

    def robust_corner_detection(self, image: np.ndarray):
        """Detects the four corners of the meter panel with robust fallbacks."""
        # 1. Edge-based contour search
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged_dilated = cv2.dilate(edged, kernel, iterations=1)

        cnts, _ = cv2.findContours(edged_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        # 2. Fallback heuristic: MinAreaRect (if quadrilateral not explicitly found)
        if cnts:
            c = cnts[0]
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            return np.intp(box)
            
        # 3. Ultimate Fallback: Image bounds
        h, w = image.shape[:2]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

    def order_points(self, pts):
        """Orders points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def apply_dewarp(self, image: np.ndarray):
        """Applies bird's eye view perspective transform."""
        pts = self.robust_corner_detection(image)
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # Avoid 0-width/height transforms
        if max_width == 0 or max_height == 0:
            return image

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        return warped

