#!/usr/bin/env bash
set -euo pipefail

#!/usr/bin/env bash
set -euo pipefail

# Env defaults
export QDRANT_API_KEY="${QDRANT_API_KEY:-changeme}"
export QDRANT__SERVICE__API_KEY="${QDRANT__SERVICE__API_KEY:-$QDRANT_API_KEY}"
export QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

# Start Qdrant in background (env var QDRANT__SERVICE__API_KEY enables API key auth)
mkdir -p /qdrant/storage
export QDRANT__STORAGE__STORAGE_PATH=/qdrant/storage
qdrant &

# Wait for Qdrant to be healthy
echo "Waiting for Qdrant to start..."
for i in {1..30}; do
  if curl -s -H "api-key: ${QDRANT_API_KEY}" http://localhost:6333/healthz >/dev/null; then
    echo "Qdrant is up."
    break
  fi
  sleep 1
done

# Start API
exec python app.py


