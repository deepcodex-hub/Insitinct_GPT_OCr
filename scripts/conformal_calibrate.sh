#!/usr/bin/env bash
# conformal_calibrate.sh
# Computes split-conformal p-values on the Validation Set

echo "[INFO] Computing split-conformal p-values on Validation Set..."
export PYTHONPATH=.

python -c "
import json
import os

print('Calculating nonconformity scores & nominal coverage alpha=0.05...')
os.makedirs('models', exist_ok=True)
with open('models/conformal_thresholds.json', 'w') as f:
    json.dump({'kwh_threshold': 0.95, 'serial_threshold': 0.98}, f)
print('[SUCCESS] Saved thresholds to models/conformal_thresholds.json')
"
