from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.jax_fidelity import orbax_receipt
from training.jax_fidelity.configuration import ConfigError, load_config, validate_config
from training.jax_fidelity.orbax_receipt import (
    OrbaxReceiptError,
    lora_checkpoint_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
_RESTORED: dict[Path, object] = {}


def _put(tree: dict[str, object], path: tuple[str, ...], value: object) -> None:
    node = tree
    for component in path[:-1]:
        node = node.setdefault(component, {})  # type: ignore[assignment]
    node[path[-1]] = value


def _evidence(items: Path, **kwargs: object) -> dict[str, object]:
    return lora_checkpoint_evidence(items, restored_tree=_RESTORED[items], **kwargs)


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
    restored: dict[str, object] = {}
    for path, shape in paths.items():
        _put(restored, path, np.zeros(shape or (), dtype=np.float32))
    _RESTORED[items] = restored
    return items


def _paired_lora_tree(
    prefix: tuple[str, ...],
    *,
    a_shape: list[int] | None = None,
    b_shape: list[int] | None = None,
) -> dict[tuple[str, ...], list[int] | None]:
    a_shape = a_shape or [2560, 16]
    b_shape = b_shape or [16, 8, 256]
    paths: dict[tuple[str, ...], list[int] | None] = {
        ("step",): None,
        (*prefix, "lora_a.kernel"): a_shape,
        (*prefix, "lora_b.kernel"): b_shape,
    }
    for moment in ("mu", "nu"):
        optimizer_prefix = ("opt_state", "0", moment, "params", *prefix[2:])
        paths[(*optimizer_prefix, "lora_a.kernel")] = a_shape
        paths[(*optimizer_prefix, "lora_b.kernel")] = b_shape
    return paths


def test_lora_evidence_accepts_paired_maxtext_orbax_paths(tmp_path: Path) -> None:
    prefix = ("params", "params", "decoder", "layers_0", "self_attention", "query")
    items = _items(tmp_path, _paired_lora_tree(prefix))

    evidence = _evidence(items, expected_rank=16)

    assert evidence["lora_pair_count"] == 1
    assert evidence["lora_tensor_count"] == 2
    assert evidence["rank"] == 16
    assert evidence["pairs"][0]["module_path"].endswith("self_attention/query")


def test_lora_evidence_binds_exact_topology_step_and_patch(tmp_path: Path) -> None:
    prefix = ("params", "params", "decoder", "layers_0", "self_attention", "query")
    items = _items(tmp_path / "checkpoints/99", _paired_lora_tree(prefix))
    patch_sha256 = "a" * 64

    evidence = _evidence(
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
        _evidence(items, expected_rank=16, expected_pair_count=205)
    with pytest.raises(OrbaxReceiptError, match="terminal step"):
        _evidence(items, expected_rank=16, expected_step=98)


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
        _evidence(items, expected_rank=16)


def test_lora_evidence_rejects_unpaired_and_wrong_rank(tmp_path: Path) -> None:
    prefix = ("params", "params", "decoder", "layers_0", "self_attention", "query")
    unpaired = _items(
        tmp_path / "unpaired",
        {("step",): None, (*prefix, "lora_a.kernel"): [2560, 16]},
    )
    with pytest.raises(OrbaxReceiptError, match="unpaired"):
        _evidence(unpaired, expected_rank=16)

    wrong_rank = _items(
        tmp_path / "wrong-rank",
        {
            ("step",): None,
            (*prefix, "lora_a.kernel"): [2560, 8],
            (*prefix, "lora_b.kernel"): [8, 8, 256],
        },
    )
    with pytest.raises(OrbaxReceiptError, match="approved rank"):
        _evidence(wrong_rank, expected_rank=16)


def test_lora_evidence_rejects_inconsistent_or_unrecognized_metadata(tmp_path: Path) -> None:
    items = _items(tmp_path / "inconsistent", {("step",): None})
    metadata = json.loads((items / "_METADATA").read_text(encoding="utf-8"))
    metadata["tree_metadata"]["('different',)"] = metadata["tree_metadata"].pop("('step',)")
    (items / "_METADATA").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(OrbaxReceiptError, match="disagree"):
        _evidence(items, expected_rank=16)

    unknown = _items(
        tmp_path / "unknown",
        {("step",): None, ("params", "mystery_lora_scale"): [1, 16]},
    )
    with pytest.raises(OrbaxReceiptError, match="unrecognized"):
        _evidence(unknown, expected_rank=16)


def test_lora_topology_ignores_optimizer_moment_copies(tmp_path: Path) -> None:
    model = ("params", "params", "decoder", "layers_0", "self_attention", "query")
    items = _items(tmp_path, _paired_lora_tree(model))

    evidence = _evidence(items, expected_rank=16, expected_pair_count=1)

    assert evidence["lora_pair_count"] == 1
    assert evidence["lora_tensor_count"] == 2
    assert evidence["optimizer_lora_tensor_count"] == 4


def test_lora_evidence_requires_exact_optimizer_moments(tmp_path: Path) -> None:
    model = ("params", "params", "decoder", "layers_0", "query")
    mu = ("opt_state", "0", "mu", "params", "decoder", "layers_0", "query")
    nu = ("opt_state", "0", "nu", "params", "decoder", "layers_0", "query")

    for label, paths in (
        (
            "missing",
            {
                ("step",): None,
                (*model, "lora_a.kernel"): [2560, 16],
                (*model, "lora_b.kernel"): [16, 256],
                (*mu, "lora_a.kernel"): [2560, 16],
                (*mu, "lora_b.kernel"): [16, 256],
                (*nu, "lora_a.kernel"): [2560, 16],
            },
        ),
        (
            "wrong-shape",
            {
                ("step",): None,
                (*model, "lora_a.kernel"): [2560, 16],
                (*model, "lora_b.kernel"): [16, 256],
                (*mu, "lora_a.kernel"): [2560, 16],
                (*mu, "lora_b.kernel"): [16, 256],
                (*nu, "lora_a.kernel"): [2560, 16],
                (*nu, "lora_b.kernel"): [16, 128],
            },
        ),
    ):
        items = _items(tmp_path / label, paths)
        with pytest.raises(OrbaxReceiptError, match="moments|shape"):
            _evidence(items, expected_rank=16, expected_pair_count=1)


def test_lora_evidence_deserializes_payload_and_rejects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ("params", "params", "decoder", "layers_0", "query")
    items = _items(tmp_path, _paired_lora_tree(model))

    real_restore = orbax_receipt._restore_orbax_tree
    monkeypatch.setattr(orbax_receipt, "_restore_orbax_tree", lambda _items: _RESTORED[items])
    evidence = lora_checkpoint_evidence(items, expected_rank=16, expected_pair_count=1)
    assert evidence["payload_arrays_restored"] is True
    assert evidence["restored_lora_array_count"] == 6

    class CorruptCheckpointer:
        def restore(self, _items: str) -> object:
            raise EOFError("truncated OCDBT payload")

    checkpoint_module = types.ModuleType("orbax.checkpoint")
    checkpoint_module.PyTreeCheckpointer = CorruptCheckpointer  # type: ignore[attr-defined]
    orbax_module = types.ModuleType("orbax")
    orbax_module.checkpoint = checkpoint_module  # type: ignore[attr-defined]
    monkeypatch.setattr(orbax_receipt, "_restore_orbax_tree", real_restore)
    monkeypatch.setitem(sys.modules, "orbax", orbax_module)
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", checkpoint_module)
    with pytest.raises(OrbaxReceiptError, match="could not be restored"):
        lora_checkpoint_evidence(items, expected_rank=16, expected_pair_count=1)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (np.full((2560, 16), np.nan, dtype=np.float32), "non-finite"),
        (np.zeros((2560, 16), dtype=np.int32), "floating"),
        (np.zeros((1, 16), dtype=np.float32), "shape"),
    ],
)
def test_lora_evidence_rejects_invalid_restored_values(
    tmp_path: Path, replacement: np.ndarray, message: str
) -> None:
    model = ("params", "params", "decoder", "layers_0", "query")
    items = _items(tmp_path, _paired_lora_tree(model))
    restored = _RESTORED[items]
    assert isinstance(restored, dict)
    restored["params"]["params"]["decoder"]["layers_0"]["query"][  # type: ignore[index]
        "lora_a.kernel"
    ] = replacement

    with pytest.raises(OrbaxReceiptError, match=message):
        _evidence(items, expected_rank=16, expected_pair_count=1)


def test_restored_orbax_paths_preserve_integer_and_string_key_identity() -> None:
    restored = {
        "opt_state": {
            "0": {
                "mu": {
                    "params": {
                        "decoder": {
                            "lora_a.kernel": np.zeros((1, 1), dtype=np.float32)
                        }
                    }
                }
            }
        }
    }
    expected = {
        (
            "opt_state",
            0,
            "mu",
            "params",
            "decoder",
            "lora_a.kernel",
        ): (1, 1)
    }

    with pytest.raises(OrbaxReceiptError, match="paths differ"):
        orbax_receipt._validate_restored_lora_arrays(restored, expected)
