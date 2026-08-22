.PHONY: install dev test lint model-serve model-pull model-probe asr-model-pull compile-demo

install:
	uv sync --dev

dev:
	uv run uvicorn bookforge.api:app --reload --host 127.0.0.1 --port 8080

test:
	uv run pytest

lint:
	uv run ruff check .

model-serve:
	ollama serve

model-pull:
	ollama pull gemma4:e2b-it-qat

model-probe:
	curl -s http://127.0.0.1:8080/v1/models:probe

asr-model-pull:
	.venv/bin/hf download mlx-community/whisper-base.en-mlx --local-dir .models/whisper-base.en

compile-demo:
	curl -s -X POST http://127.0.0.1:8080/v1/story-packs:compile \
		-H 'Content-Type: application/json' \
		--data @examples/moon-gate.request.json
