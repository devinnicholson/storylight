"""Build trusted Gemma 4 checkpoint round-trip evidence from real artifacts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .configuration import EOS_TOKEN_IDS, ExperimentConfig
from .integrity import artifact_manifest, sha256_file

_MAX_KL_PATTERN = re.compile(
    r"Max KL divergence for a single token in the set:\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)
_ARCHITECTURE_FIELDS = (
    "attention_k_eq_v",
    "global_head_dim",
    "head_dim",
    "hidden_size",
    "hidden_size_per_layer_input",
    "intermediate_size",
    "layer_types",
    "num_attention_heads",
    "num_global_key_value_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_kv_shared_layers",
    "rope_parameters",
    "vocab_size",
    "vocab_size_per_layer_input",
)
_TOKENIZER_FILES = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


class CheckpointEvidenceError(ValueError):
    """A checkpoint cannot prove the pinned Gemma 4 round-trip contract."""


def parse_max_kl_divergence(output: str) -> float:
    """Extract the real maximum KL value emitted by pinned MaxText."""

    matches = [float(match.group("value")) for match in _MAX_KL_PATTERN.finditer(output)]
    if not matches:
        raise CheckpointEvidenceError("MaxText output did not contain a maximum KL value")
    value = max(matches)
    if not math.isfinite(value) or value < 0:
        raise CheckpointEvidenceError("MaxText emitted an invalid maximum KL value")
    return value


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointEvidenceError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise CheckpointEvidenceError(f"{label} must contain one JSON object")
    return value


def _text_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _json_object(root / "config.json", "checkpoint config")
    model_type = config.get("model_type")
    if model_type not in {"gemma4", "gemma4_text"}:
        raise CheckpointEvidenceError(f"unexpected Gemma model_type: {model_type!r}")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise CheckpointEvidenceError("Gemma 4 text_config is missing")
    return config, text


def _safetensor_shapes(root: Path) -> dict[str, tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise CheckpointEvidenceError(
            "safetensors is required for checkpoint inspection"
        ) from error

    result: dict[str, tuple[int, ...]] = {}
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise CheckpointEvidenceError("checkpoint contains no SafeTensors weights")
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safe_open is not iterable.
                if name in result:
                    raise CheckpointEvidenceError(f"duplicate SafeTensors key: {name}")
                result[name] = tuple(handle.get_slice(name).get_shape())
    return result


def _tokenizer_binding(root: Path) -> dict[str, str]:
    result = {
        name: sha256_file(root / name)
        for name in _TOKENIZER_FILES
        if (root / name).is_file() and not (root / name).is_symlink()
    }
    if "tokenizer.json" not in result and "tokenizer.model" not in result:
        raise CheckpointEvidenceError("checkpoint has no tokenizer vocabulary")
    if "tokenizer_config.json" not in result:
        raise CheckpointEvidenceError("checkpoint has no tokenizer configuration")
    return result


def _eos_ids(root: Path, config: Mapping[str, Any], text: Mapping[str, Any]) -> list[int]:
    candidates: list[object] = [config.get("eos_token_id"), text.get("eos_token_id")]
    generation = root / "generation_config.json"
    if generation.is_file():
        candidates.insert(0, _json_object(generation, "generation config").get("eos_token_id"))
    for value in candidates:
        if isinstance(value, list) and all(type(item) is int for item in value):
            return value
    raise CheckpointEvidenceError("checkpoint does not declare the complete EOS token set")


def inspect_hf_roundtrip(
    config: ExperimentConfig,
    *,
    base_checkpoint: Path | str,
    exported_checkpoint: Path | str,
) -> dict[str, object]:
    """Compare base and merged HF checkpoints without trusting converter exit status alone."""

    base = Path(base_checkpoint).resolve()
    exported = Path(exported_checkpoint).resolve()
    base_config, base_text = _text_config(base)
    exported_config, exported_text = _text_config(exported)

    missing = [name for name in _ARCHITECTURE_FIELDS if name not in base_text]
    if missing:
        raise CheckpointEvidenceError(f"base text config lacks architecture fields: {missing}")
    if any(exported_text.get(name) != base_text[name] for name in _ARCHITECTURE_FIELDS):
        raise CheckpointEvidenceError("exported Gemma 4 architecture differs from the base")

    base_shapes = _safetensor_shapes(base)
    exported_shapes = _safetensor_shapes(exported)
    if set(base_shapes) != set(exported_shapes):
        raise CheckpointEvidenceError("exported checkpoint tensor names differ from the base")
    if base_shapes != exported_shapes:
        raise CheckpointEvidenceError("exported checkpoint tensor shapes differ from the base")

    base_tokenizer = _tokenizer_binding(base)
    exported_tokenizer = _tokenizer_binding(exported)
    if exported_tokenizer != base_tokenizer:
        raise CheckpointEvidenceError("exported tokenizer bytes differ from the pinned base")

    eos_ids = _eos_ids(exported, exported_config, exported_text)
    if eos_ids != list(EOS_TOKEN_IDS):
        raise CheckpointEvidenceError("exported checkpoint EOS token set changed")

    ple_names = {
        name
        for name in base_shapes
        if "embed_tokens_per_layer" in name
        or "per_layer_model_projection" in name
        or "per_layer_projection" in name
    }
    if not ple_names or not ple_names.issubset(exported_shapes):
        raise CheckpointEvidenceError("Gemma 4 PLE tensors are missing after export")

    kv_sharing = {
        "attention_k_eq_v": base_text["attention_k_eq_v"],
        "num_global_key_value_heads": base_text["num_global_key_value_heads"],
        "num_key_value_heads": base_text["num_key_value_heads"],
        "num_kv_shared_layers": base_text["num_kv_shared_layers"],
    }
    if not isinstance(kv_sharing["num_kv_shared_layers"], int):
        raise CheckpointEvidenceError("Gemma 4 KV-sharing metadata is invalid")

    return {
        "tensor_names": True,
        "tensor_shapes": True,
        "tokenizer": True,
        "special_tokens": True,
        "ple_weights": True,
        "kv_sharing": True,
        "gemma4_metadata": True,
        "eos_token_ids": eos_ids,
        "architecture": {name: base_text[name] for name in _ARCHITECTURE_FIELDS},
        "kv_sharing_metadata": kv_sharing,
        "ple_tensor_names_sha256": _name_digest(ple_names),
        "tensor_names_sha256": _name_digest(base_shapes),
        "tokenizer_files": base_tokenizer,
        "base_checkpoint_manifest": artifact_manifest(base),
        "exported_checkpoint_manifest": artifact_manifest(exported),
        "config_sha256": config.sha256,
    }


def _name_digest(names: Mapping[str, object] | set[str]) -> str:
    import hashlib

    return hashlib.sha256("".join(f"{name}\n" for name in sorted(names)).encode()).hexdigest()
