from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.jax_fidelity.configuration import ConfigError, load_config, validate_config
from training.jax_fidelity.orbax_receipt import (
    OrbaxReceiptError,
    lora_checkpoint_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _entry(path: tuple[str, ...], shape: list[int] | None) -> dict[str, object]:
    value: dict[str, object] = {
        "value_type": "jax.Array",
        "skip_deserialize": False,
    }
    if shape is not None:
        value["write_shape"] = shape
    return {
        "key_metadata": [{"key": component, "key_type": 2} for component in path],
        "value_metadata": value,
    }


def _items(tmp_path: Path, paths: dict[tuple[str, ...], list[int] | None]) -> Path:
    items = tmp_path / "items"
    items.mkdir(parents=True)
    tree = {repr(path): _entry(path, shape) for path, shape in paths.items()}
    (items / "_METADATA").write_text(
        json.dumps({"tree_metadata": tree, "use_ocdbt": True}),
        encoding="utf-8",
    )
    return items


def test_lora_evidence_accepts_paired_maxtext_orbax_paths(tmp_path: Path) -> None:
    prefix = ("params", "params", "decoder", "layers_0", "self_attention", "query")
    items = _items(
        tmp_path,
        {
            ("step",): None,
            (*prefix, "lora_a.kernel"): [2560, 16],
            (*prefix, "lora_b.kernel"): [16, 8, 256],
        },
    )

    evidence = lora_checkpoint_evidence(items, expected_rank=16)

    assert evidence["lora_pair_count"] == 1
    assert evidence["lora_tensor_count"] == 2
    assert evidence["rank"] == 16
    assert evidence["pairs"][0]["module_path"].endswith("self_attention/query")


def test_lora_evidence_binds_exact_topology_step_and_patch(tmp_path: Path) -> None:
    prefix = ("params", "params", "decoder", "layers_0", "self_attention", "query")
    items = _items(
        tmp_path / "checkpoints/99",
        {
            ("step",): None,
            (*prefix, "lora_a.kernel"): [2560, 16],
            (*prefix, "lora_b.kernel"): [16, 8, 256],
        },
    )
    patch_sha256 = "a" * 64

    evidence = lora_checkpoint_evidence(
        items,
        expected_rank=16,
        expected_pair_count=1,
        expected_step=99,
        approved_maxtext_patch_sha256=patch_sha256,
    )

    assert evidence["expected_lora_pair_count"] == 1
    assert evidence["checkpoint_step"] == 99
    assert evidence["checkpoint_step_binding"] == "directory-name-plus-root-step-leaf"
    assert evidence["approved_maxtext_patch_sha256"] == patch_sha256

    with pytest.raises(OrbaxReceiptError, match="model topology"):
        lora_checkpoint_evidence(items, expected_rank=16, expected_pair_count=205)
    with pytest.raises(OrbaxReceiptError, match="terminal step"):
        lora_checkpoint_evidence(items, expected_rank=16, expected_step=98)


def test_v3_config_binds_exact_gemma4_topology_and_maxtext_patch() -> None:
    path = ROOT / "experiments/jax-fidelity-lab/config-v3-canary.json"
    config = load_config(path)

    assert config.training["expected_lora_pair_count"] == 205
    assert len(config.training["approved_maxtext_patch_sha256"]) == 64

    drifted = json.loads(path.read_text(encoding="utf-8"))
    drifted["training"]["expected_lora_pair_count"] = 204
    with pytest.raises(ConfigError, match="expected_lora_pair_count"):
        validate_config(drifted)


def test_lora_evidence_rejects_actual_step_only_orbax_shape(tmp_path: Path) -> None:
    items = _items(tmp_path, {("step",): None})

    with pytest.raises(OrbaxReceiptError, match="contains no LoRA tensors"):
        lora_checkpoint_evidence(items, expected_rank=16)


def test_lora_evidence_rejects_unpaired_and_wrong_rank(tmp_path: Path) -> None:
    prefix = ("params", "decoder", "layers_0", "self_attention", "query")
    unpaired = _items(
        tmp_path / "unpaired",
        {("step",): None, (*prefix, "lora_a.kernel"): [2560, 16]},
    )
    with pytest.raises(OrbaxReceiptError, match="unpaired"):
        lora_checkpoint_evidence(unpaired, expected_rank=16)

    wrong_rank = _items(
        tmp_path / "wrong-rank",
        {
            ("step",): None,
            (*prefix, "lora_a.kernel"): [2560, 8],
            (*prefix, "lora_b.kernel"): [8, 8, 256],
        },
    )
    with pytest.raises(OrbaxReceiptError, match="approved rank"):
        lora_checkpoint_evidence(wrong_rank, expected_rank=16)


def test_lora_evidence_rejects_inconsistent_or_unrecognized_metadata(tmp_path: Path) -> None:
    items = _items(tmp_path / "inconsistent", {("step",): None})
    metadata = json.loads((items / "_METADATA").read_text(encoding="utf-8"))
    metadata["tree_metadata"]["('different',)"] = metadata["tree_metadata"].pop("('step',)")
    (items / "_METADATA").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(OrbaxReceiptError, match="disagree"):
        lora_checkpoint_evidence(items, expected_rank=16)

    unknown = _items(
        tmp_path / "unknown",
        {("step",): None, ("params", "mystery_lora_scale"): [1, 16]},
    )
    with pytest.raises(OrbaxReceiptError, match="unrecognized"):
        lora_checkpoint_evidence(unknown, expected_rank=16)
