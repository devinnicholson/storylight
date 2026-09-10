.PHONY: install dev test lint model-serve model-pull model-probe asr-model-pull compile-demo anticipatory-simulate anticipatory-gke-preflight

install:
	uv sync --dev --extra modal-authoring

dev:
	uv run uvicorn storylight.api:app --reload --host 127.0.0.1 --port 8080 --timeout-graceful-shutdown 3

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts deploy infra

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

anticipatory-simulate:
	@test -n "$(OUT)" || (echo "Set OUT to a new simulation evidence path" >&2; exit 2)
	uv run python -m storylight.anticipatory_simulator --output "$(OUT)"

anticipatory-gke-preflight:
	./infra/gcp/gke/preflight-anticipatory.sh
