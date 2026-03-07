class DecimalValidator:
    """
    If OCR string lacks a decimal, generates candidate placements. 
    Scores candidates using OCR confidences + decimal_detector output + domain rules.
    """
    def __init__(self, field_type="kwh"):
        self.field_type = field_type
        # Domain prior: kWh usually has 1 decimal place. kVAh: 1, kW: 2.
        self.expected_decimals = {
            "kwh": 1,
            "kvah": 1,
            "md_kw": 2,
            "demand_kva": 2
        }.get(field_type.lower(), 1)

    def validate_and_correct(self, raw_ocr_text: str, ocr_conf: float, decimal_detector_conf: float, initial_candidates: list = None) -> dict:
        """Generates the structured candidate payload requested."""
        if initial_candidates is None:
            initial_candidates = [{"value": raw_ocr_text, "score": ocr_conf}]
            
        candidates = []
        seen = set()
        
        # Process all incoming candidates
        for cand in initial_candidates:
            val = cand["value"]
            score = cand["score"]
            if not val.replace(".", "").isdigit():
                candidates.append(cand)
                seen.add(val)
                continue
                
            # Direct candidate
            if val not in seen:
                candidates.append({"value": val, "score": score * (1.2 if val.count(".") == 1 else 0.8)})
                seen.add(val)
                
            base_digits = val.replace(".", "")
            
            # If the string doesn't have a decimal, but the decimal detector fired high
            if "." not in val and decimal_detector_conf > 0.5:
                if len(base_digits) > self.expected_decimals:
                    idx = len(base_digits) - self.expected_decimals
                    domain_candidate = base_digits[:idx] + "." + base_digits[idx:]
                    if domain_candidate not in seen:
                        # Blended probability combining OCR and Decimal Classifier
                        candidates.append({
                            "value": domain_candidate,
                            "score": score * 0.5 + decimal_detector_conf * 0.5 
                        })
                        seen.add(domain_candidate)

        # Sort candidates
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        best_candidate = candidates[0] if candidates else {"value": raw_ocr_text, "score": ocr_conf}
        
        # Update and return structure
        return {
            "value": best_candidate["value"],
            "probability": best_candidate["score"],
            "decimals": best_candidate["value"].count("."),
            "candidates": candidates
        }
