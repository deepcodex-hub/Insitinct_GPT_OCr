import numpy as np

class PaddleAdapter:
    """Adapter for PaddleOCR."""
    def __init__(self, use_gpu=True):
        self.mock_mode = True

    def recognize(self, image: np.ndarray) -> dict:
        if self.mock_mode:
            return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}
        
        results = self.ocr.ocr(image, cls=True)
        # Paddle returns structured nested lists
        if not results or not results[0]:
             return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}
             
        # Just grab the text from the first detected line for this isolated meter field
        line = results[0][0] 
        text = line[1][0]
        conf = line[1][1]
        
        return {
            "text": text,
            "confidence": conf,
            "tokens": list(text),
            "token_scores": [conf] * len(text) # Paddle doesn't give per-char by default
        }
