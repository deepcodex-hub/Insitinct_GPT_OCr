import numpy as np

class EasyOCRAdapter:
    """Adapter for EasyOCR."""
    def __init__(self, use_gpu=True):
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=use_gpu)
            self.mock_mode = False
        except Exception as e:
            print(f"Warning: Could not load EasyOCR. Running in mock mode. Error: {e}")
            self.mock_mode = True

    def recognize(self, image: np.ndarray) -> dict:
        if self.mock_mode:
            return {"text": "34567.2", "confidence": 0.88, "tokens": ["3","4","5","6","7",".","2"], "token_scores": [0.88]*7}
            
        results = self.reader.readtext(image)
        if not results:
             return {"text": "", "confidence": 0.0, "tokens": [], "token_scores": []}
             
        # Extract the highest confidence result or combine
        best_res = max(results, key=lambda x: x[2])
        text = best_res[1]
        conf = best_res[2]
        
        return {
            "text": text,
            "confidence": conf,
            "tokens": list(text),
            "token_scores": [conf] * len(text)
        }
