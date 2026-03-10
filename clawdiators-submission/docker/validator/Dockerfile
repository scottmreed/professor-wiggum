FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY validator_service.py /app/validator_service.py

EXPOSE 8080

HEALTHCHECK --interval=5s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "validator_service:app", "--host", "0.0.0.0", "--port", "8080"]
