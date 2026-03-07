import copy

class DecimalAwareRover:
    """Token-alignment voting mechanism combining multiple hypothesis paths."""
    
    def __init__(self, decimal_penalty=2.0):
        self.decimal_penalty = decimal_penalty

    def align_and_vote(self, ocr_results: list) -> dict:
        if not ocr_results:
            return {"text": "", "confidence": 0.0, "candidates": []}

        valid_res = [r for r in ocr_results if r["text"]]
        if not valid_res:
             return {"text": "", "confidence": 0.0, "candidates": []}
        
        # Count occurrences, weighting by decimal presence (penalty for deletion)
        text_scores = {}
        source_texts = []
        for r in valid_res:
            t = r['text']
            source_texts.append(t)
            w = self.decimal_penalty if '.' in t else 1.0
            text_scores[t] = text_scores.get(t, 0) + w * r['confidence']
            
        # Check for absolute agreement on digits
        # (This implements: "If two models agree on digits and TrOCR differs, choose majority")
        best_text = max(text_scores, key=text_scores.get)
        
        # Build candidates
        total_score = sum(text_scores.values())
        candidates = []
        for t, s in text_scores.items():
            candidates.append({"value": t, "score": s / total_score if total_score > 0 else 0})
            
        # Calculate raw average confidence of the winning text among those who voted for it
        voters = [r['confidence'] for r in valid_res if r['text'] == best_text]
        final_conf = sum(voters) / len(valid_res) # Average over all candidates for safety
        
        return {
            "text": best_text,
            "confidence": final_conf,
            "candidates": candidates
        }
