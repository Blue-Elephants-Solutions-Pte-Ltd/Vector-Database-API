#!/usr/bin/env bash
set -euo pipefail

# Ensure required tools
command -v docker >/dev/null 2>&1 || { echo "Docker is required"; exit 1; }

# 1) Ensure external named volume for Qdrant
docker volume create ai-assistant-qdrant >/dev/null

# 2) Read or set QDRANT_API_KEY
ENV_FILE="$(dirname "$0")/.env"
if [ ! -f "$ENV_FILE" ]; then
  read -rp "Enter Qdrant API Key: " QDRANT_API_KEY
  echo "QDRANT_API_KEY=${QDRANT_API_KEY}" > "$ENV_FILE"
  echo "Wrote $ENV_FILE"
else
  echo "Using existing $ENV_FILE"
fi

# 3) Start the stack
docker compose -f "$(dirname "$0")/docker-compose.yml" --env-file "$ENV_FILE" up -d
echo "Services started. API: http://localhost:2100  Qdrant: http://localhost:6333"


