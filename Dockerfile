# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim

# Create a non-root user (UID 1000 is default for HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code with correct ownership
COPY --chown=user . $HOME/app

# Set PYTHONPATH so imports resolve correctly
ENV PYTHONPATH="$HOME/app:$PYTHONPATH"

# Expose the server port (7860 is the most compatible with HF)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
