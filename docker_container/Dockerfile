# Use official Python image as a base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install git and system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
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

# Expose the port your app runs on
EXPOSE 2100

# Switch to non-root user
USER appuser

# Run the application
CMD ["python", "app.py"]
