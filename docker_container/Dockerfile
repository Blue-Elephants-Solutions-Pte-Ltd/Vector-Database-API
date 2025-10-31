# Use official Python image as a base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install git and system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Copy local code into the container
COPY . .

# Set proper permissions
RUN chown -R appuser:appuser /app && \
    mkdir -p /app/VECTOR_DB && \
    chown -R appuser:appuser /app/VECTOR_DB && \
    chmod -R 777 /app/VECTOR_DB

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Qdrant binary
USER root
RUN set -e; \
    mkdir -p /qdrant/storage /opt/qdrant && \
    chown -R appuser:appuser /qdrant && \
    curl -fsSL -o /tmp/qdrant.tar.gz https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz && \
    tar -xzf /tmp/qdrant.tar.gz -C /opt/qdrant && \
    if [ -x /opt/qdrant/qdrant ]; then \
      ln -sf /opt/qdrant/qdrant /usr/local/bin/qdrant; \
    elif [ -x /opt/qdrant/bin/qdrant ]; then \
      ln -sf /opt/qdrant/bin/qdrant /usr/local/bin/qdrant; \
    else \
      echo "Qdrant binary not found after extract:"; ls -la /opt/qdrant; ls -la /opt/qdrant/bin || true; exit 1; \
    fi && \
    rm /tmp/qdrant.tar.gz && \
    chmod +x /usr/local/bin/qdrant

# Copy entrypoint that starts Qdrant and the API
COPY docker_container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown appuser:appuser /entrypoint.sh

# Expose API and Qdrant ports
EXPOSE 2100 6333

# Switch to non-root user
USER appuser
ENTRYPOINT ["/entrypoint.sh"]
