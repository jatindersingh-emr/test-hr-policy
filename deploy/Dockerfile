

FROM python:3.11-slim

# System dependencies for Azure SDK, cryptography, and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Expose API port
EXPOSE 8080

# Required environment variables (documented for deployment)
#   AZURE_SEARCH_ENDPOINT
#   AZURE_SEARCH_API_KEY
#   AZURE_SEARCH_INDEX_NAME
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_EMBEDDING_DEPLOYMENT

# Startup command
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8080"]

