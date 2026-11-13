#!/bin/bash
# Usage: ./deploy.sh YOUR_API_KEY
set -e

API_KEY=${1:-$VULTR_INFERENCE_API_KEY}
if [ -z "$API_KEY" ]; then
  echo "Usage: ./deploy.sh <API_KEY>"
  exit 1
fi

echo "Building sandbox image (pandas, no app code)..."
docker build -t data-pipeline-sandbox -f sandbox/Dockerfile sandbox

echo "Building app image..."
docker build -t data-pipeline .

docker rm -f data-pipeline 2>/dev/null || true
docker run -d \
  --name data-pipeline \
  -p 8501:8501 \
  -e VULTR_INFERENCE_API_KEY="$API_KEY" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  data-pipeline

echo "Open http://$(curl -s ifconfig.me):8501 and click Run judge demo."
