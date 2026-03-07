import numpy as np

class ConformalPredictor:
    """
    Split-conformal wrapper providing calibrated prediction bounds.
    Derives nonconformity scores from token-level probabilities.
    """
    def __init__(self, alpha=0.05, calibration_scores=None):
        self.alpha = alpha # Target error rate (e.g., 5% -> 95% coverage)
        
        # Nonconformity scores computed on the calibration set (Stage 10)
        # Score s_i = 1 - P(correct_class)
        if calibration_scores is not None:
            n = len(calibration_scores)
            q_level = np.ceil((n + 1) * (1 - alpha)) / n
            self.q_hat = np.quantile(calibration_scores, q_level, method='higher')
        else:
            # Dummy quantile for structural pipeline if not fitted
            self.q_hat = 0.05 

    def get_prediction_interval(self, token_probs: list) -> dict:
        """
        Given a sequence of token probabilities, calculates if the sequence
        falls within the conformal prediction set.
        """
        if not token_probs:
            return {"in_set": False, "interval_lower": 0.0, "interval_upper": 0.0}

        # Sequence probability is roughly the product of token probs
        # Nonconformity score for the sequence
        seq_prob = np.prod(token_probs)
        nonconformity_score = 1.0 - seq_prob
        
        # If the nonconformity score is less than our calibrated quantile (q_hat),
        # the prediction is inside our valid conformal set.
        in_set = nonconformity_score <= self.q_hat
        
        # Dummy bounds based on the quantile
        return {
            "in_set": bool(in_set),
            "interval_lower": max(0.0, seq_prob - self.q_hat),
            "interval_upper": min(1.0, seq_prob + self.q_hat)
        }
