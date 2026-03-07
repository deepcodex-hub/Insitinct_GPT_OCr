import pytest
from ocr_pipeline.ensemble_rover import ROVEREnsemble

def test_needleman_wunsch():
    # Implicitly tested through align_hypotheses/vote, but good to have
    ensemble = ROVEREnsemble()
    hyps = ["12345", "12845"]
    conf = [0.9, 0.8]
    best, net = ensemble.vote(hyps, conf)
    assert best in ["12345", "12845"]

def test_rover_majority():
    ensemble = ROVEREnsemble()
    hyps = ["12345", "12345", "99999"]
    conf = [0.9, 0.9, 0.95]
    # Despite 99999 having higher individual conf, 12345 has 1.8 total weight
    best, net = ensemble.vote(hyps, conf)
    assert best == "12345"

def test_rover_gap_handling():
    ensemble = ROVEREnsemble()
    hyps = ["1234", "12834", "12-34"] # Simulating missing chars
    conf = [1.0, 1.0, 1.0]
    best, net = ensemble.vote(hyps, conf)
    # The expected behavior depends on exact weights, but it should not crash
    assert len(best) >= 3 # at least core chars
