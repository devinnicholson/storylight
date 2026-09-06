FROM python:3.12-slim@sha256:2fe5997d249a808b8eeea52c58a1dbffbba28754dc11699ef5c029f2d818ce79
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=65532:65532 klein_weights.py klein_scene_runtime.py ./
RUN python -c 'from klein_weights import bake_weights; from klein_scene_runtime import MODEL, MODEL_REVISION, DEPTH_MODEL, DEPTH_REVISION; bake_weights({"klein": [MODEL, MODEL_REVISION], "depth": [DEPTH_MODEL, DEPTH_REVISION]})'
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY --chown=65532:65532 app.py manifest.json ./
RUN useradd --create-home --uid 65532 bookforge
USER 65532:65532
EXPOSE 8080
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
