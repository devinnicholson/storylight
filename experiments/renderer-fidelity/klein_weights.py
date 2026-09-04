"""CPU-only, pinned model-image build step, independent of inference code."""

import json
from pathlib import Path


def bake_weights(weights: dict):
    from huggingface_hub import snapshot_download

    for name, (model, revision) in weights.items():
        snapshot_download(
            model,
            revision=revision,
            local_dir=f"/models/{name}",
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        )
    Path("/models/identities.json").write_text(json.dumps(weights, sort_keys=True))
