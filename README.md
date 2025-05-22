# Vector Database API

A powerful API service for managing vector embeddings and document retrieval using various LLM providers.

## Docker Setup

### Prerequisites
- Docker installed on your server
- Sudo access to create directories

### Important Note
> ⚠️ Please execute the commands exactly as shown below. Do not modify any parameters or flags as they are specifically configured for optimal performance.

### Installation Steps

1. Update system packages and create the required directory for vector storage:
```bash
# Update system packages
sudo apt update

# Create a directory to store vector data with proper permissions
# The -p flag will create all necessary parent directories including /var/www if they don't exist
sudo mkdir -p /var/www/VECTOR_DB
sudo chmod 777 /var/www/VECTOR_DB
```

2. Pull and run the Docker container:
```bash

# Download the latest version of the vector-db-api image
docker pull ghcr.io/blue-elephants-solutions-pte-ltd/vector-db-api:latest

# Run the container in detached mode, mapping port 2100 and mounting the vector storage directory
docker run -d -p 2100:2100 --name vector-db-api -v /var/www/VECTOR_DB:/app/VECTOR_DB ghcr.io/blue-elephants-solutions-pte-ttd/vector-db-api:latest
```

### Container Management

- View container logs:
```bash
# Monitor the container logs in real-time
docker logs -f vector-db-api
```

## Notes

- The API runs on port 2100 by default
- Vector data is persisted in the mounted volume at `/var/www/VECTOR_DB`
- Each user's vectors are stored in separate directories
