# Vector Database API

A powerful API service for managing vector embeddings and document retrieval using various LLM providers.

## Prerequisites
- Docker (Windows: Docker Desktop; Linux/macOS: Docker Engine + docker compose)
- Docker installed on your server
- Sudo access to create directories

### Important Note
> ⚠️ Please execute the commands exactly as shown below. Do not modify any parameters or flags as they are specifically configured for optimal performance.

## Quick Start (docker run, no compose)
Run Qdrant and the API with simple docker run commands (no compose).

### Run pulled image
```bash
# Create persistent volume for Qdrant storage (once)
docker volume create ai-assistant-qdrant

# Run the single container
docker network create vector-net

docker run -d \
  --name qdrant \
  --network vector-net \
  -p 6333:6333 \
  -e QDRANT__SERVICE__API_KEY=your-strong-key \
  -v ai-assistant-qdrant:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant:latest

docker run -d \
  --name vector-db-api \
  --network vector-net \
  -e QDRANT_API_KEY=your-strong-key \
  -e QDRANT_URL=http://qdrant:6333 \
  -p 2100:2100 \
  --restart unless-stopped \
  ghcr.io/blue-elephants-solutions-pte-ltd/vector-db-api:latest


# Verify
curl -H "api-key: your-strong-key" http://localhost:6333/healthz
curl http://localhost:2100/
```

### What gets started
- Qdrant (vector database) with API key authentication, data persisted in the volume `ai-assistant-qdrant`.
- The Vector Database API (port 2100) in a separate container.

### Authentication
- Qdrant uses its native API key authentication. The UI and API require the header `api-key: <YOUR_QDRANT_API_KEY>`.

### Updating to a new version
```bash
docker pull ghcr.io/blue-elephants-solutions-pte-ltd/vector-db-api:latest
docker rm -f vector-db-api || true
docker run -d --name vector-db-api \
  --network vector-net \
  -e QDRANT_API_KEY=your-strong-key \
  -e QDRANT_URL=http://qdrant:6333 \
  -p 2100:2100 \
  --restart unless-stopped \
  ghcr.io/blue-elephants-solutions-pte-ltd/vector-db-api:latest
```
Your data remains intact because it lives in the external volume.

---

### Container Management

- View container logs:
```bash
# Monitor the container logs in real-time
docker logs -f vector-db-api
```

## Notes
- The API runs on port 2100 by default
---

## Environment Template

An example env file is provided at `docker_container/env.template`. Copy it to `.env` if you want to pre-provision the key without prompts:

```bash
cp docker_container/env.template docker_container/.env
```
