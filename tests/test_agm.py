import pytest
import numpy as np
from ag_module.dewarp import DewarpProcessor
from ag_module.fuse import TemporalFuser

def test_dewarp_fallbacks():
    dewarper = DewarpProcessor()
    # Create a blank image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Process
    warped = dewarper.apply_dewarp(img)
    # Since no contour is found, it should fallback to image bounds and return 100x100
    assert warped.shape == (100, 100, 3)

def test_fuse_median():
    fuser = TemporalFuser()
    
    # Create 3 identical frames with 1 outlier pixel
    frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 50
    frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 50
    frame3 = np.ones((10, 10, 3), dtype=np.uint8) * 255 # Glare frame
    
    fused = fuser.fuse([frame1, frame2, frame3])
    
    # The median should reject the 255 glare and return 50
    assert np.all((fused > 48) & (fused < 52))
