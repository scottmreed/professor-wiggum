FROM python:3.11-slim

# Install system dependencies for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake ground truth into image at build time.
# The image contains the answers — no runtime network calls ever.
# Source: clawdiators-submission/easy/ground_truth.json
COPY ground_truth.json /app/ground_truth.json

COPY scorer_service.py /app/scorer_service.py
COPY scoring_utils.py /app/scoring_utils.py

EXPOSE 8080

HEALTHCHECK --interval=5s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "scorer_service:app", "--host", "0.0.0.0", "--port", "8080"]
