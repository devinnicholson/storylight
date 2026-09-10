FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

USER 65532:65532

EXPOSE 8080

CMD ["sh", "-c", "uvicorn storylight.api:app --host 0.0.0.0 --port ${PORT}"]
