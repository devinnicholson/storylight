"""CPU-only recipe/proof regressions; no independent development or screen reads."""

import copy
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v5_train_under_test", HERE / "train.py")
TRAIN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TRAIN)
PARENT = TRAIN.load_module(
    HERE.parent / "scene-adapter-v2-2026-09-08/train.py", TRAIN.V2_SHA256, "v5_test_parent"
)


@pytest.mark.parametrize("steps,schedule", [(1200, (400, 800, 1200)), (4800, (1200, 2400, 4800))])
def test_complete_schedule_selects_earliest_minimum(steps, schedule):
    assert TRAIN.checkpoint_steps(steps) == schedule
    rows = [
        dict(step=s, rows=256, supervised_tokens=7000, token_weighted_loss=loss)
        for s, loss in zip(schedule, (0.3, 0.1, 0.1), strict=True)
    ]
    assert TRAIN.select_checkpoint(rows, schedule, PARENT) == schedule[1]
    rows[-1]["token_weighted_loss"] = 0.05
    assert TRAIN.select_checkpoint(rows, schedule, PARENT) == schedule[-1]
    with pytest.raises(ValueError):
        TRAIN.checkpoint_steps(2400)


@pytest.mark.parametrize("mutation", ["missing", "reordered", "tokens", "nan", "negative", "rows"])
def test_selection_rejects_incomplete_or_incompatible_development(mutation):
    schedule = TRAIN.checkpoint_steps(4800)
    rows = [
        dict(step=s, rows=256, supervised_tokens=7000, token_weighted_loss=0.1) for s in schedule
    ]
    if mutation == "missing":
        rows.pop()
    elif mutation == "reordered":
        rows.reverse()
    elif mutation == "tokens":
        rows[1]["supervised_tokens"] += 1
    elif mutation == "rows":
        rows[1]["rows"] = 120
    else:
        rows[1]["token_weighted_loss"] = float("nan") if mutation == "nan" else -0.1
    with pytest.raises(PARENT.ExperimentRejected):
        TRAIN.select_checkpoint(rows, schedule, PARENT)


def test_message_inventory_rejects_hash_roles_and_duplicate_ids(tmp_path):
    rows = [
        dict(id=f"synthetic-{i}", messages=[dict(role="user", content=f"fixture-{i}")])
        for i in range(2)
    ]
    path = tmp_path / "fixture.jsonl"

    def write(value):
        path.write_text("".join(json.dumps(r) + "\n" for r in value))
        return {"files": {path.name: {"sha256": PARENT.digest(path), "rows": 2}}}

    manifest = write(rows)
    assert TRAIN.read_rows(tmp_path, manifest, path.name, 2, PARENT, ["user"]) == rows
    path.write_text(path.read_text() + "\n")
    with pytest.raises(PARENT.ExperimentRejected, match="data_file_mismatch"):
        TRAIN.read_rows(tmp_path, manifest, path.name, 2, PARENT, ["user"])
    changed = copy.deepcopy(rows)
    changed[1]["id"] = changed[0]["id"]
    with pytest.raises(PARENT.ExperimentRejected, match="duplicate_data_id"):
        TRAIN.read_rows(tmp_path, write(changed), path.name, 2, PARENT, ["user"])
    changed = copy.deepcopy(rows)
    changed[0]["messages"][0]["role"] = "assistant"
    with pytest.raises(PARENT.ExperimentRejected, match="data_roles"):
        TRAIN.read_rows(tmp_path, write(changed), path.name, 2, PARENT, ["user"])


def test_actual_pinned_gemma_text_linear_layout_on_meta_device():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    assert transformers.__version__ == "5.13.0"
    directory = Path(
        os.environ.get("STORYLIGHT_V3_TOKENIZER_DIR", "/private/tmp/storylight-v3-tokenizer")
    )
    if not directory.exists():
        pytest.skip("Pinned local configuration is unavailable")
    config_path = directory / "config.json"
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == (
        "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330"
    )
    config = transformers.Gemma4Config.from_pretrained(directory, local_files_only=True)
    with torch.device("meta"):
        model = transformers.Gemma4ForConditionalGeneration(config)
    modules = dict(model.named_modules())
    wide = TRAIN.targets("text-linear", model)
    narrow = TRAIN.targets("qv", model)
    assert narrow < wide
    assert len(wide) == 275 and len(narrow) == 50
    assert Counter(name.rsplit(".", 1)[1] for name in wide) == {
        "q_proj": 35,
        "k_proj": 15,
        "v_proj": 15,
        "o_proj": 35,
        "gate_proj": 35,
        "up_proj": 35,
        "down_proj": 35,
        "per_layer_input_gate": 35,
        "per_layer_projection": 35,
    }
    assert all(type(modules[name]) is torch.nn.Linear for name in wide)
    for targets, expected in ((narrow, 2678784), (wide, 26165248)):
        assert (
            sum(16 * (modules[n].in_features + modules[n].out_features) for n in targets)
            == expected
        )
    with pytest.raises(ValueError):
        TRAIN.targets("all", model)


def test_checkpoint_roundtrip_delta_and_tampered_inventory(tmp_path):
    torch = pytest.importorskip("torch")
    safe = pytest.importorskip("safetensors.torch")
    params = {
        f"model.layer.lora_{side}.default.weight": torch.nn.Parameter(torch.full((2, 2), 0.1))
        for side in ("A", "B")
    }
    initial = {n: torch.zeros_like(p) for n, p in params.items()}

    class Model:
        def save_pretrained(self, directory, **kwargs):
            directory.mkdir()
            safe.save_file(
                {n: p.detach() for n, p in params.items()},
                str(directory / "adapter_model.safetensors"),
            )

        def named_parameters(self):
            return params.items()

    def restore(model, saved):
        with torch.no_grad():
            for name, value in saved.items():
                params[name].copy_(value)

    peft = SimpleNamespace(
        get_peft_model_state_dict=lambda model: params, set_peft_model_state_dict=restore
    )
    model = Model()
    receipt = TRAIN.save_checkpoint(model, initial, params, tmp_path, 400, peft, torch, PARENT)
    assert receipt["tensor_count"] == 2 and receipt["delta_l2"] > 0
    with torch.no_grad():
        for p in params.values():
            p.zero_()
    PARENT.reload_checkpoint(model, tmp_path, receipt, peft, torch)
    assert all(torch.equal(p, torch.full_like(p, 0.1)) for p in params.values())
    (tmp_path / "adapter-400" / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(PARENT.ExperimentRejected, match="checkpoint_inventory_changed"):
        PARENT.reload_checkpoint(model, tmp_path, receipt, peft, torch)
    with pytest.raises(PARENT.ExperimentRejected, match="unchanged_adapter"):
        TRAIN.save_checkpoint(
            model,
            {n: p.detach().clone() for n, p in params.items()},
            params,
            tmp_path,
            800,
            peft,
            torch,
            PARENT,
        )
