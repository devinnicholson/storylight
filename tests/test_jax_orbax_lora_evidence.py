from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.jax_fidelity import orbax_receipt
from training.jax_fidelity.orbax_receipt import (
    OrbaxReceiptError,
    lora_checkpoint_evidence,
    lora_checkpoint_progression_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
_RESTORED: dict[Path, object] = {}


def _put(
    tree: dict[str | int, object], path: tuple[str | int, ...], value: object
) -> None:
    node = tree
    for component in path[:-1]:
        child = node.setdefault(component, {})
        if not isinstance(child, dict):
            raise AssertionError("fixture path collides with an existing leaf")
        node = child
    node[path[-1]] = value


def _evidence(items: Path, **kwargs: object) -> dict[str, object]:
    return lora_checkpoint_evidence(items, restored_tree=_RESTORED[items], **kwargs)


def _entry(path: tuple[str | int, ...], shape: list[int] | None) -> dict[str, object]:
    value: dict[str, object] = {
        "value_type": "jax.Array",
        "skip_deserialize": False,
    }
    if shape is not None:
        value["write_shape"] = shape
    return {
        "key_metadata": [
            {
                "key": str(component) if isinstance(component, int) else component,
                "key_type": 1 if isinstance(component, int) else 2,
            }
            for component in path
        ],
        "value_metadata": value,
    }


def _items(
    tmp_path: Path, paths: dict[tuple[str | int, ...], list[int] | None]
) -> Path:
    items = tmp_path / "items"
    items.mkdir(parents=True)
    tree = {}
    for path, shape in paths.items():
        serialized_path = tuple(
            str(component) if isinstance(component, int) else component
            for component in path
        )
        tree[repr(serialized_path)] = _entry(path, shape)
    (items / "_METADATA").write_text(
        json.dumps({"tree_metadata": tree, "use_ocdbt": True}),
        encoding="utf-8",
    )
    restored: dict[str | int, object] = {}
    for path, shape in paths.items():
        _put(restored, path, np.zeros(shape or (), dtype=np.float32))
    _RESTORED[items] = restored
    return items


def _paired_lora_tree(
    prefix: tuple[str, ...],
    *,
    a_shape: list[int] | None = None,
    b_shape: list[int] | None = None,
    a_leaf: str = "lora_a.kernel",
    b_leaf: str = "lora_b.kernel",
) -> dict[tuple[str | int, ...], list[int] | None]:
    a_shape = a_shape or [2560, 16]
    b_shape = b_shape or [16, 8, 256]
    paths: dict[tuple[str | int, ...], list[int] | None] = {
        ("step",): None,
        (*prefix, a_leaf): a_shape,
        (*prefix, b_leaf): b_shape,
    }
    for moment in ("mu", "nu"):
        optimizer_prefix = ("opt_state", 0, moment, "params", *prefix[2:])
        paths[(*optimizer_prefix, a_leaf)] = a_shape
        paths[(*optimizer_prefix, b_leaf)] = b_shape
    return paths


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


def test_lora_checkpoint_progression_proves_restored_parameter_delta(
    tmp_path: Path,
) -> None:
    prefix = ("params", "params", "decoder", "layers_0", "self_attention", "query")
    paths = _paired_lora_tree(
        prefix,
        a_shape=[2, 16],
        b_shape=[16, 2],
        a_leaf="kernel_lora_a",
        b_leaf="kernel_lora_b",
    )
    initial = _items(tmp_path / "checkpoints/0", paths)
    terminal = _items(tmp_path / "checkpoints/99", paths)
    terminal_model = _RESTORED[terminal]["params"]["params"]["decoder"][  # type: ignore[index]
        "layers_0"
    ]["self_attention"]["query"]
    terminal_model["kernel_lora_a"][0, 0] = 0.5
    terminal_model["kernel_lora_b"][1, 1] = -0.25

    evidence = lora_checkpoint_progression_evidence(
        initial,
        terminal,
        expected_rank=16,
        expected_pair_count=1,
        initial_step=0,
        terminal_step=99,
        minimum_relative_delta=1e-6,
        approved_maxtext_patch_sha256="a" * 64,
        initial_restored_tree=_RESTORED[initial],
        terminal_restored_tree=_RESTORED[terminal],
    )

    assert evidence["status"] == "passed"
    assert evidence["model_lora_array_count"] == 2
    assert evidence["changed_model_lora_array_count"] == 2
    assert evidence["changed_model_lora_element_count"] == 2
    assert evidence["checkpoint_delta_l2_norm"] == pytest.approx(0.559016994)
    assert evidence["terminal_adapter"]["checkpoint_step"] == 99


def test_lora_evidence_requires_exact_optimizer_moments(tmp_path: Path) -> None:
    model = ("params", "params", "decoder", "layers_0", "query")
    mu = ("opt_state", 0, "mu", "params", "decoder", "layers_0", "query")
    nu = ("opt_state", 0, "nu", "params", "decoder", "layers_0", "query")

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
