#!/bin/bash
# demo_run.sh
# One-file demo: spins local docker compose (api, minio, redis, triton stub) and runs curl to /infer

set -e

echo "Spinning up local dev environment..."
# Assuming a standard docker-compose.yml exists at the root
docker compose up -d api minio redis triton

echo "Waiting for API to become healthy..."
sleep 10

echo "Running sample inference..."
SAMPLE_IMG="examples/sample1.jpg"

if [ ! -f "$SAMPLE_IMG" ]; then
    echo "Warning: $SAMPLE_IMG not found. Creating a dummy black image for demo."
    mkdir -p examples
    python -c "import cv2; import numpy as np; cv2.imwrite('$SAMPLE_IMG', np.zeros((640, 480, 3), dtype=np.uint8))"
fi

curl -X POST "http://localhost:8000/infer" -F "file=@$SAMPLE_IMG" | jq

echo "Saving debug artifacts..."
# In a real run, the API would have uploaded these to MinIO.
# This script just demonstrably hits the endpoint and parses the resulting JSON.

echo "Demo complete."
