FROM python:3.11-slim

# Install system dependencies for git and tree-sitter compilation
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY src/ ./src/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

# Create data directory for SQLite
RUN mkdir -p /app/data

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
