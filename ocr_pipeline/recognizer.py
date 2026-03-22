import cv2
import numpy as np
import warnings

class OCRRecognizer:
    def __init__(self, use_gpu=False):
        self.mock_mode = False
        self.use_gpu = use_gpu
        
        # Try importing engines
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            model_id = "microsoft/trocr-base-printed"
            self.trocr_processor = TrOCRProcessor.from_pretrained(model_id)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device)
        except ImportError:
            warnings.warn("Transformers/TrOCR not found. Running in mock mode.")
            self.mock_mode = True

        try:
            from paddleocr import PaddleOCR
            import easyocr
            self.paddle = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu)
            self.easy = easyocr.Reader(['en'], gpu=use_gpu)
        except (ImportError, Exception):
            warnings.warn("PaddleOCR/EasyOCR not found. Ensemble will omit them.")
            self.paddle = None
            self.easy = None

    def recognize_trocr(self, image: np.ndarray):
        if self.mock_mode:
            return [{"text": "12345", "confidence": 0.90}]
            
        # Convert BGR (cv2 array) to RGB (PIL Image)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(rgb_img)
        
        import torch
        pixel_values = self.trocr_processor(images=pil_img, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.trocr_model.generate(pixel_values)
            # Simplification: HuggingFace generate doesn't return raw logits directly easily without output_scores=True
            # For MVP, we assign a high pseudo-confidence to TrOCR as primary driver if it decodes
        
        preds = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return [{"text": preds.strip(), "confidence": 0.90}] # Stub conf

    def recognize_paddle(self, image: np.ndarray):
        if self.paddle is None:
            return []
        result = self.paddle.ocr(image, cls=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                texts.append({"text": line[1][0], "confidence": float(line[1][1])})
        return texts

    def recognize_easy(self, image: np.ndarray):
        if self.easy is None:
            return []
        result = self.easy.readtext(image)
        texts = []
        for _, text, conf in result:
            texts.append({"text": text, "confidence": float(conf)})
        return texts

    def ensemble_vote(self, results_list: list):
        """Uses ROVER-style token alignment for ensembling."""
        from ocr_pipeline.ensemble_rover import DecimalAwareRover
        
        ocr_results = []
        for engine_res in results_list:
            if not engine_res: continue
            best = max(engine_res, key=lambda x: x['confidence'])
            ocr_results.append({"text": best['text'], "confidence": best['confidence']})
            
        rover = DecimalAwareRover()
        return rover.align_and_vote(ocr_results)

