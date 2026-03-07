import pytest
import numpy as np
import cv2
from ag_module.enhancer import AGMEnhancer

def test_enhancer_clahe():
    enhancer = AGMEnhancer()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create some variation
    cv2.rectangle(img, (20, 20), (80, 80), (128, 128, 128), -1)
    
    enhanced = enhancer.apply_clahe(img)
    assert enhanced.shape == img.shape
    assert not np.array_equal(enhanced, img)

def test_enhancer_gamma():
    enhancer = AGMEnhancer(gamma=2.0)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    enhanced = enhancer.apply_gamma(img)
    assert enhanced.shape == img.shape
    # Gamma 2.0 should darken the mid-tones
    assert np.mean(enhanced) < np.mean(img)

def test_enhancer_full_pipeline():
    enhancer = AGMEnhancer()
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    result = enhancer.enhance(img)
    assert "enhanced_image" in result
    assert "debug_artifacts" in result
    assert result["enhanced_image"].shape == img.shape
