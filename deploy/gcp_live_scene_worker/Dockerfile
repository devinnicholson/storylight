FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STORYLIGHT_MODEL_CACHE=/models/huggingface \
    HF_HOME=/models/huggingface \
    PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY download_models.py ./
RUN python3 download_models.py

RUN useradd --create-home --uid 65532 storylight \
    && chown -R storylight:storylight /app
COPY --chown=65532:65532 app.py ./
USER 65532:65532

EXPOSE 8080
CMD ["sh", "-c", "python3 -m uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
