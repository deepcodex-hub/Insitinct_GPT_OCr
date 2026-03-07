import re
import Levenshtein

class LMCorrector:
    def __init__(self, valid_patterns=None):
        self.valid_patterns = valid_patterns or [
            r'^\d{5,8}$',      # Generic meter reading
            r'^\d{4,6}\.\d{1}$' # Reading with decimal
        ]

    def correct_by_regex(self, text: str):
        """Basic regex correction."""
        # Remove non-alphanumeric except dots
        cleaned = re.sub(r'[^a-zA-Z0-9.]', '', text)
        
        # If it looks almost like a number, fix common mistakes
        # (e.g., 'O' -> '0', 'l' -> '1')
        corrections = {
            'O': '0', 'o': '0',
            'I': '1', 'l': '1', '|': '1',
            'S': '5', 's': '5',
            'B': '8',
            'Z': '2', 'z': '2'
        }
        
        for k, v in corrections.items():
            cleaned = cleaned.replace(k, v)
        
        return cleaned

    def validate(self, text: str):
        """Checks if text matches any valid pattern."""
        for pattern in self.valid_patterns:
            if re.match(pattern, text):
                return True
        return False

    def correct_ensemble_result(self, ensemble_result: dict):
        """Applies LM corrections to the ensemble result."""
        if not ensemble_result:
            return None
            
        original_text = ensemble_result['text']
        corrected_text = self.correct_by_regex(original_text)
        
        ensemble_result['text'] = corrected_text
        ensemble_result['validated'] = self.validate(corrected_text)
        return ensemble_result
