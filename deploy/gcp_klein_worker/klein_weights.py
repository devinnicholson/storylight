"""Bake the pinned Diffusers components without the unused single-file checkpoint."""

import json
from pathlib import Path

from klein_scene_runtime import MODEL, MODEL_REVISION


def bake_weights(weights: dict):
    from huggingface_hub import snapshot_download

    for name, (model, revision) in weights.items():
        snapshot_download(
            model,
            revision=revision,
            local_dir=f"/models/{name}",
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
            ignore_patterns=["flux-2-klein-4b.safetensors"]
            if (model, revision) == (MODEL, MODEL_REVISION)
            else None,
        )
    Path("/models/identities.json").write_text(json.dumps(weights, sort_keys=True))
