"""CPU runtime admission and mixed-target adapter isolation, without screen data."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


RUN = module("v5_runtime_under_test", "run.py")
TRAIN = module("v5_runtime_trainer", "train.py")
PARENT = TRAIN.load_module(
    HERE / "support/training-support.py", TRAIN.V2_SHA256, "v5_runtime_parent"
)


def test_balanced_complete_schedule_and_training_only_smoke():
    rows = [{"id": f"synthetic-{i}"} for i in range(128)]
    plan = RUN.schedule(rows)
    assert len(plan) == 512
    assert [r["dispatch_ordinal"] for r in plan] == list(range(512))
    assert len({(r["id"], r["arm"], r["repetition"]) for r in plan}) == 512
    for i in range(128):
        group = plan[4 * i : 4 * i + 4]
        assert {r["id"] for r in group} == {rows[i]["id"]}
        assert [r["arm"] for r in group] == (
            ["old", "new", "new", "old"] if i % 2 == 0 else ["new", "old", "old", "new"]
        )
    smoke = RUN.schedule(rows[:2], preflight=True)
    assert len(smoke) == 4 and {r["repetition"] for r in smoke} == {0}


def checkpoint_fixture(directory, recipe="text-linear", old=False):
    schedule = (400, 800, 1200) if old else (1200, 2400, 4800)
    selected = schedule[1]
    adapter = directory / f"adapter-{selected}"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic-adapter-proof")
    proof = dict(
        step=selected,
        tensor_count=100 if old or recipe == "qv" else 550,
        delta_l2=0.1,
        files={"adapter_model.safetensors": PARENT.digest(adapter / "adapter_model.safetensors")},
    )
    PARENT.write_json(directory / f"checkpoint-{selected}.json", proof)
    runner_sha = RUN.OLD_TRAINER_SHA256 if old else PARENT.digest(HERE / "train.py")
    protocol = dict(runner_sha256=runner_sha, model_manifest_sha256="a" * 64, recipe=recipe)
    PARENT.write_json(directory / "protocol.json", protocol)
    PARENT.write_json(directory / "loaded.json", {})
    development = [
        dict(step=s, rows=120 if old else 256, supervised_tokens=7000, token_weighted_loss=loss)
        for s, loss in zip(schedule, (0.2, 0.1, 0.3), strict=True)
    ]
    PARENT.write_json(
        directory / "selection.json",
        dict(
            selected_step=selected,
            development=development,
            checkpoint_receipt_sha256=PARENT.digest(directory / f"checkpoint-{selected}.json"),
            selected_adapter_reloaded_and_verified=True,
            screen_generation_started=False,
        ),
    )
    completed = dict(
        completed_steps=1200 if old else 4800,
        selected_step=selected,
        files={
            str(p.relative_to(directory)): PARENT.digest(p)
            for p in directory.rglob("*")
            if p.is_file()
        },
    )
    PARENT.write_json(directory / "completed.json", completed)
    return runner_sha


@pytest.mark.parametrize("recipe,old", [("qv", True), ("qv", False), ("text-linear", False)])
def test_checkpoint_binds_correct_training_length_and_adapter_inventory(tmp_path, recipe, old):
    runner_sha = checkpoint_fixture(tmp_path, recipe, old)

    def check():
        return RUN.checkpoint(
            tmp_path,
            PARENT.digest(tmp_path / "completed.json"),
            runner_sha,
            "a" * 64,
            PARENT,
            TRAIN,
            old=old,
        )

    adapter, proof, _ = check()
    assert proof["tensor_count"] == (100 if old or recipe == "qv" else 550)
    completed_path = tmp_path / "completed.json"
    saved = completed_path.read_text()
    wrong = json.loads(saved)
    wrong["completed_steps"] = 4800 if old else 1200
    completed_path.write_text(json.dumps(wrong))
    with pytest.raises(PARENT.ExperimentRejected, match="training_incomplete"):
        check()
    completed_path.write_text(saved)
    (adapter / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(PARENT.ExperimentRejected, match="adapter_inventory"):
        check()
    (adapter / "unexpected.bin").unlink()
    (adapter / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(PARENT.ExperimentRejected, match="training_file_changed"):
        check()


def test_real_peft_mixed_target_adapters_do_not_leak_after_switch_or_reload(tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    assert peft.__version__ == "0.20.0"

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(4, 4, bias=False)
            self.b = torch.nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.a.weight.copy_(torch.eye(4))
                self.b.weight.copy_(torch.eye(4))

        def forward(self, x):
            return self.b(self.a(x))

    model = peft.get_peft_model(
        Tiny(),
        peft.LoraConfig(r=2, lora_alpha=2, target_modules=["a"], lora_dropout=0),
        adapter_name="old",
    )
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".lora_" in n:
                p.fill_(0.1)
    x = torch.ones(1, 4)
    original_old = model(x).detach().clone()
    model.add_adapter(
        "new", peft.LoraConfig(r=2, lora_alpha=2, target_modules=["a", "b"], lora_dropout=0)
    )
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".lora_" in n and ".new." in n:
                p.fill_(0.4)
    model.requires_grad_(False)
    identity = RUN.parameter_identity(model)
    results = {}
    for arm in ("new", "old", "new", "old"):
        RUN.select_adapter(model, arm, identity, PARENT)
        output = model(x).detach()
        if arm in results:
            assert torch.equal(output, results[arm])
        results[arm] = output.clone()
    assert torch.equal(results["old"], original_old)
    assert not torch.equal(results["new"], original_old)
    assert "old" not in model.base_model.model.b.lora_A
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = peft.PeftModel.from_pretrained(
        Tiny(), tmp_path / "old", adapter_name="old", is_trainable=False, local_files_only=True
    )
    restored.load_adapter(
        tmp_path / "new", adapter_name="new", is_trainable=False, local_files_only=True
    )
    restored.requires_grad_(False)
    restored_identity = RUN.parameter_identity(restored)
    for arm in ("new", "old"):
        RUN.select_adapter(restored, arm, restored_identity, PARENT)
        assert torch.equal(restored(x).detach(), results[arm])
    with torch.no_grad():
        next(restored.parameters()).add_(1)
    with pytest.raises(PARENT.ExperimentRejected, match="adapter_selection_integrity"):
        RUN.select_adapter(restored, "old", restored_identity, PARENT)
