import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import argparse
import cv2
from ag_module.detector import MeterDetector
from ag_module.enhancer import AGMEnhancer
from ocr_pipeline.recognizer import OCRRecognizer

def calculate_cer(gt, pred):
    """Character Error Rate."""
    if not gt: return len(pred)
    if not pred: return len(gt)
    # Simple edit distance
    d = np.zeros((len(gt)+1, len(pred)+1))
    for i in range(len(gt)+1): d[i][0] = i
    for j in range(len(pred)+1): d[0][j] = j
    for i in range(1, len(gt)+1):
        for j in range(1, len(pred)+1):
            if gt[i-1] == pred[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+1)
    return d[len(gt)][len(pred)] / len(gt)

class Evaluator:
    def __init__(self):
        self.detector = MeterDetector()
        self.enhancer = AGMEnhancer()
        self.recognizer = OCRRecognizer()

    def evaluate(self, testset_path):
        df = pd.read_csv(testset_path)
        results = []
        
        for index, row in df.iterrows():
            img_path = row['image_path']
            gt_text = str(row['text'])
            
            image = cv2.imread(img_path)
            if image is None: continue
            
            start_time = time.time()
            
            # Full Pipeline
            detections = self.detector.detect(image)
            field = self.detector.crop_detected_fields(image, detections)[0] if detections else image
            enhanced = self.enhancer.enhance(field)['enhanced_image']
            
            paddle_res = self.recognizer.recognize_paddle(enhanced)
            final = self.recognizer.ensemble_vote([paddle_res]) # Simplified for MVP
            
            latency = (time.time() - start_time) * 1000
            pred_text = final['text'] if final else ""
            
            cer = calculate_cer(gt_text, pred_text)
            field_acc = 1.0 if gt_text == pred_text else 0.0
            
            results.append({
                "gt": gt_text,
                "pred": pred_text,
                "cer": cer,
                "field_acc": field_acc,
                "latency": latency
            })
            
        results_df = pd.DataFrame(results)
        print(f"Digit Accuracy (1-CER): {1 - results_df['cer'].mean():.4f}")
        print(f"Field-level Accuracy: {results_df['field_acc'].mean():.4f}")
        print(f"Average Latency: {results_df['latency'].mean():.2f} ms")
        
        return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", required=True)
    args = parser.parse_args()
    
    evaluator = Evaluator()
    evaluator.evaluate(args.testset)
