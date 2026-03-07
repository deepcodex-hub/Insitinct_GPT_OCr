#!/bin/bash
# Advanced AGM OCR Pipeline - Dev Run Script

echo "Starting Anti-Gravity Pipeline Local Stack..."

# Start MinIO, Redis, MLflow tracking server, and API/Worker
docker-compose -f docker-compose.yml up -d minio redis

echo "Waiting for backing services to initialize..."
sleep 3

echo "Starting Main Pipeline Services (API, Worker)..."
docker-compose -f docker-compose.yml up --build -d api worker

echo "Triton Server (Local Stub) assumption applied based on app.py fallback logic."

echo "Services running:"
docker-compose ps

echo "=================================="
echo "API Available: http://localhost:8000/"
echo "Health check: http://localhost:8000/health"
echo "MinIO Console: http://localhost:9001/"
echo "=================================="

# Wait for API to be healthy
echo "Running smoke test..."
sleep 5
curl -f http://localhost:8000/health || echo "Health check failed!"

echo "Local environment ready."
