FROM python:3.11-slim AS base

WORKDIR /app

# Dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Hugging Face Spaces expects port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
