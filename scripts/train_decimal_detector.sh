#!/usr/bin/env bash
# train_decimal_detector.sh
echo "[INFO] Training PyTorch Decimal CNN Detector..."
export PYTHONPATH=.
python scripts/train_decimal_cnn.py
echo "[SUCCESS] Decimal Detector training finished (best weights saved into models/weights)."
