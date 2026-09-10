"""Offline receipt corruption checks; synthetic IDs/losses, no evaluation labels."""

import importlib.util
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


V = load("training_receipt_verifier", "verify-training.py")
T = load("training_receipt_trainer", "train.py")
H = T.load_module(
    HERE / "support/training-support.py", T.V2_SHA256, "training_receipt_helper"
)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def lines(path, values):
    path.write_text("".join(json.dumps(v) + "\n" for v in values))


def reseal(directory):
    path = directory / "completed.json"
    complete = V.read(path)
    complete["files"] = {
        str(p.relative_to(directory)): V.digest(p)
        for p in directory.rglob("*")
        if p.is_file() and p != path
    }
    write(path, complete)
    return V.digest(path)


def fixture(directory, recipe="qv", steps=1200):
    directory.mkdir()
    cpu = {
        "rows": [
            {
                "id": f"{split}-{i}",
                "split": split,
                "completion_tokens": i % 3 + 1,
                "total_tokens": 100,
            }
            for split, count in (("training", 4800), ("development", 256))
            for i in range(count)
        ],
        **{
            k: "a" * 64
            for k in (
                "model_manifest_sha256",
                "data_manifest_sha256",
                "development_manifest_sha256",
            )
        },
        "fewshot_prompt_sha256": T.PROMPT_SHA256,
    }
    schedule = T.checkpoint_steps(steps)
    protocol = dict(
        runner_sha256=V.TRAINER_SHA,
        helper_sha256=T.V2_SHA256,
        model_id=H.MODEL_ID,
        revision=H.REVISION,
        **{k: v for k, v in cpu.items() if k != "rows"},
        recipe=recipe,
        max_steps=steps,
        training_rows=4800,
        development_rows=256,
        checkpoint_steps=list(schedule),
        seed=H.SEED,
        max_length=2048,
        max_new_tokens=256,
        batch_size=1,
        gradient_accumulation=1,
        rank=16,
        alpha=32,
        dropout=0,
        learning_rate=0.0001,
        optimizer="AdamW",
        quantization="nf4-double-quant-bf16",
        loss="completion-only",
        fresh_base=True,
        screen_inputs_read=False,
        evaluation_gold_read=False,
        production_export=False,
        selection="lowest token-weighted development completion loss, earliest step on tie",
        versions=H.VERSIONS,
        cuda="12.8",
        device="NVIDIA L4",
        max_runtime_seconds=7200,
        max_train_tokens=100,
        max_development_tokens=100,
    )
    write(directory / "protocol.json", protocol)
    initial = T.INITIAL_SHA256 if recipe == "qv" else "b" * 64
    write(
        directory / "loaded.json",
        dict(
            initial_adapter_sha256=initial,
            expected_initial_adapter_sha256=initial if steps == 4800 else None,
            targets=sorted(V.expected_targets(recipe)),
            trainable_parameters=2678784 if recipe == "qv" else 26165248,
        ),
    )
    order = list(range(4800))
    random.Random(H.SEED).shuffle(order)
    lines(
        directory / "training.jsonl",
        [
            dict(
                step=step,
                id=cpu["rows"][index]["id"],
                supervised_tokens=cpu["rows"][index]["completion_tokens"],
                loss=0.5,
                gradient_norm_before_clip=1.0,
                latency_ms=1.0,
            )
            for step, index in enumerate(order[:steps], 1)
        ],
    )
    aggregates = []
    for step, loss in zip(schedule, (0.5, 0.25, 0.25), strict=True):
        raw = [
            dict(
                id=r["id"],
                step=step,
                supervised_tokens=r["completion_tokens"],
                completion_loss=loss,
            )
            for r in cpu["rows"]
            if r["split"] == "development"
        ]
        lines(directory / f"development-{step}.jsonl", raw)
        aggregate = dict(
            step=step,
            rows=256,
            supervised_tokens=sum(r["supervised_tokens"] for r in raw),
            token_weighted_loss=loss,
            latency_ms=10.0,
        )
        write(directory / f"development-{step}.json", aggregate)
        aggregates.append(aggregate)
        adapter = directory / f"adapter-{step}"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"synthetic weight boundary")
        write(
            directory / f"checkpoint-{step}.json",
            dict(
                step=step,
                initial_adapter_sha256=initial,
                final_adapter_sha256="c" * 64,
                delta_l2=1.0,
                tensor_count=100 if recipe == "qv" else 550,
                files={
                    "adapter_model.safetensors": V.digest(adapter / "adapter_model.safetensors")
                },
            ),
        )
    selected = schedule[1]
    write(
        directory / "selection.json",
        dict(
            selected_step=selected,
            development=aggregates,
            checkpoint_receipt_sha256=V.digest(directory / f"checkpoint-{selected}.json"),
            selection_inputs="development completion loss only",
            selected_adapter_reloaded_and_verified=True,
            screen_generation_started=False,
        ),
    )
    write(
        directory / "completed.json",
        dict(
            completed_steps=steps,
            recipe=recipe,
            selected_step=selected,
            training_compute_ms=float(steps),
            training_ms=float(steps + 100),
            training_ms_includes_development_and_checkpointing=True,
            wall_seconds=10.0,
            screen_inputs_read=False,
            quality_accepted=False,
            production_export=False,
        ),
    )
    return cpu, reseal(directory)


@pytest.mark.parametrize("recipe,steps", [("qv", 1200), ("text-linear", 1200), ("qv", 4800)])
def test_complete_schedules_and_earliest_loss_tie(tmp_path, monkeypatch, recipe, steps):
    directory = tmp_path / "run"
    cpu, sha = fixture(directory, recipe, steps)
    calls = []
    monkeypatch.setattr(V, "weights", lambda *args: calls.append(args[0].name))
    result = V.verify_run(directory, sha, cpu, T, H)
    assert result["selected_step"] == T.checkpoint_steps(steps)[1]
    assert calls == [f"adapter-{s}" for s in T.checkpoint_steps(steps)]


@pytest.mark.parametrize(
    "mutation",
    [
        "training_id",
        "tokens",
        "nan",
        "dev_id",
        "dev_loss",
        "winner",
        "lr",
        "failure",
        "missing_proof",
        "timing",
    ],
)
def test_resealed_mutations_fail(tmp_path, monkeypatch, mutation):
    directory = tmp_path / "run"
    cpu, _ = fixture(directory)
    monkeypatch.setattr(V, "weights", lambda *args: None)
    if mutation in {"training_id", "tokens", "nan"}:
        path = directory / "training.jsonl"
        raw = V.rows(path)
        raw[0][{"training_id": "id", "tokens": "supervised_tokens", "nan": "loss"}[mutation]] = {
            "training_id": raw[1]["id"],
            "tokens": 999,
            "nan": float("nan"),
        }[mutation]
        lines(path, raw)
    elif mutation in {"dev_id", "dev_loss"}:
        path = directory / "development-400.jsonl"
        raw = V.rows(path)
        raw[0]["id" if mutation == "dev_id" else "completion_loss"] = (
            "different" if mutation == "dev_id" else 50.0
        )
        lines(path, raw)
    elif mutation == "winner":
        path = directory / "selection.json"
        value = V.read(path)
        value.update(
            selected_step=1200,
            checkpoint_receipt_sha256=V.digest(directory / "checkpoint-1200.json"),
        )
        write(path, value)
        value = V.read(directory / "completed.json")
        value["selected_step"] = 1200
        write(directory / "completed.json", value)
    elif mutation == "lr":
        path = directory / "protocol.json"
        value = V.read(path)
        value["learning_rate"] = 0.01
        write(path, value)
    elif mutation == "failure":
        write(directory / "failure.json", {"completed": False})
    elif mutation == "missing_proof":
        (directory / "checkpoint-400.json").unlink()
    else:
        value = V.read(directory / "completed.json")
        value["wall_seconds"] = 1.0
        write(directory / "completed.json", value)
    with pytest.raises((ValueError, FileNotFoundError)):
        V.verify_run(directory, reseal(directory), cpu, T, H)


def test_saved_tensor_bytes_and_shape_are_independently_verified(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    saved = {name: torch.ones(shape) for name, shape in V.tensor_shapes("qv").items()}
    path = tmp_path / "adapter_model.safetensors"
    safetensors.save_file(saved, str(path))
    write(
        tmp_path / "adapter_config.json",
        dict(
            target_modules=sorted(V.expected_targets("qv")),
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            modules_to_save=None,
        ),
    )
    normalized = {
        n.replace(".lora_A.", ".lora_A.default.").replace(".lora_B.", ".lora_B.default."): t
        for n, t in saved.items()
    }
    proof = dict(tensor_count=100, final_adapter_sha256=H.parameter_hash(normalized))

    def seal():
        proof["files"] = {p.name: V.digest(p) for p in tmp_path.iterdir()}

    seal()
    V.weights(tmp_path, proof, "qv", H)
    name = next(iter(saved))
    saved[name] = saved[name] + 1
    safetensors.save_file(saved, str(path))
    seal()
    with pytest.raises(ValueError, match="saved parameter hash"):
        V.weights(tmp_path, proof, "qv", H)
    saved[name] = saved[name].T.contiguous()
    safetensors.save_file(saved, str(path))
    seal()
    with pytest.raises(ValueError, match="adapter tensor"):
        V.weights(tmp_path, proof, "qv", H)


def test_context_binds_actual_cpu_proof_without_reading_labels():
    freeze = HERE / "source-freeze.json"
    cpu = HERE / "results/cpu-preflight.json"
    result, _, _ = V.context(freeze, V.digest(freeze), cpu, V.digest(cpu))
    assert len(result["rows"]) == 5056


def test_full_run_requires_pilot_proofs(tmp_path, monkeypatch):
    cpu = {"rows": []}
    monkeypatch.setattr(V, "context", lambda *args: (cpu, T, H))
    monkeypatch.setattr(V, "verify_run", lambda *args: {"steps": 4800})
    args = SimpleNamespace(
        freeze=None,
        freeze_sha256="a",
        cpu_preflight=None,
        cpu_preflight_sha256="b",
        directory=tmp_path,
        completed_sha256="c",
    )
    with pytest.raises(ValueError, match="both pilots"):
        V.verify(args)


@pytest.mark.parametrize("mutation", [None, "wrong_recipe", "wrong_initial", "wrong_tie"])
def test_full_run_binds_both_verified_pilots_and_qv_tie(tmp_path, monkeypatch, mutation):
    cpu = {
        k: "a" * 64
        for k in (
            "model_manifest_sha256",
            "data_manifest_sha256",
            "development_manifest_sha256",
            "fewshot_prompt_sha256",
        )
    }
    common = {**cpu, "seed": H.SEED, "learning_rate": 0.0001, "rank": 16, "alpha": 32}
    pilots = [
        dict(
            recipe=recipe,
            selected_pilot_step=800,
            development_loss=0.5,
            pilot_completed_sha256=pin,
            initial_adapter_sha256=initial,
        )
        for recipe, pin, initial in (
            ("qv", "b" * 64, T.INITIAL_SHA256),
            ("text-linear", "c" * 64, "d" * 64),
        )
    ]
    selector = dict(
        selector_sha256=V.SELECTOR_SHA,
        trainer_sha256=V.TRAINER_SHA,
        candidates=pilots,
        selected=pilots[1 if mutation == "wrong_tie" else 0],
        test_inputs_or_gold_read=False,
        quality_accepted=False,
        common_protocol=common,
        selection_rule="lowest selected pilot development loss; Q/V on exact tie",
        next_run="fresh initialization, full 4800-example pass",
    )
    path = tmp_path / "recipe.json"
    write(path, selector)
    args = SimpleNamespace(
        freeze=None,
        freeze_sha256="a",
        cpu_preflight=None,
        cpu_preflight_sha256="b",
        directory=tmp_path / "full",
        completed_sha256="e" * 64,
        qv_pilot=tmp_path / "qv",
        qv_completed_sha256="b" * 64,
        text_linear_pilot=tmp_path / "text-linear",
        text_linear_completed_sha256="c" * 64,
        recipe_selection=path,
        recipe_selection_sha256=V.digest(path),
    )
    monkeypatch.setattr(V, "context", lambda *args: (cpu, T, H))
    called = []

    def verified(directory, pin, *_):
        called.append(directory.name)
        if directory.name == "full":
            return dict(
                steps=4800,
                recipe="text-linear" if mutation == "wrong_recipe" else "qv",
                initial_adapter_sha256="f" * 64
                if mutation == "wrong_initial"
                else T.INITIAL_SHA256,
            )
        pilot = next(p for p in pilots if p["recipe"] == directory.name)
        return dict(
            steps=1200,
            recipe=pilot["recipe"],
            selected_step=800,
            development_loss=0.5,
            completed_sha256=pin,
            initial_adapter_sha256=pilot["initial_adapter_sha256"],
        )

    monkeypatch.setattr(V, "verify_run", verified)
    if mutation is None:
        assert V.verify(args)["verified"] is True
    else:
        with pytest.raises(ValueError):
            V.verify(args)
    assert called == ["full", "qv", "text-linear"]


def test_exact_matrix_shapes_account_for_both_adapter_recipes():
    for recipe, count, parameters in (("qv", 100, 2678784), ("text-linear", 550, 26165248)):
        shapes = V.tensor_shapes(recipe)
        assert len(shapes) == count
        assert sum(a * b for a, b in shapes.values()) == parameters


def test_matrix_shapes_match_actual_pinned_model_on_meta_device():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    directory = Path(
        os.environ.get("STORYLIGHT_V3_TOKENIZER_DIR", "/private/tmp/storylight-v3-tokenizer")
    )
    if not directory.exists():
        pytest.skip("Pinned local configuration is unavailable")
    assert transformers.__version__ == "5.13.0"
    assert V.digest(directory / "config.json") == (
        "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330"
    )
    config = transformers.Gemma4Config.from_pretrained(directory, local_files_only=True)
    with torch.device("meta"):
        model = transformers.Gemma4ForConditionalGeneration(config)
    target_proof = V.pinned(HERE / "results/target-config-proof.json", V.TARGET_PROOF_SHA)
    assert target_proof["module_names"] == [name for name, _ in model.named_modules()]
    from peft.tuners.tuners_utils import _find_minimal_target_modules

    text_targets = T.targets("text-linear", model)
    minimized = _find_minimal_target_modules(
        text_targets, [name for name, _ in model.named_modules() if name not in text_targets])
    assert set(target_proof["minimal_targets"]) == minimized
    assert V.target_configuration(sorted(minimized), text_targets)
    for recipe in ("qv", "text-linear"):
        expected = {}
        for name in T.targets(recipe, model):
            layer = model.get_submodule(name)
            expected[f"base_model.model.{name}.lora_A.weight"] = (16, layer.in_features)
            expected[f"base_model.model.{name}.lora_B.weight"] = (layer.out_features, 16)
        assert V.tensor_shapes(recipe) == expected


def test_compact_targets_require_exact_full_architecture_expansion():
    proof = V.pinned(HERE / "results/target-config-proof.json", V.TARGET_PROOF_SHA)
    compact = proof["minimal_targets"]
    expected = V.expected_targets("text-linear")
    assert V.target_configuration(compact, expected)
    assert not V.target_configuration(compact[:-1], expected)
    assert not V.target_configuration(compact + ["q_proj"], expected)  # Includes vision.
    assert not V.target_configuration(compact + ["unknown_module"], expected)
    with pytest.raises(ValueError, match="target configuration type"):
        V.target_configuration(compact + [compact[0]], expected)


def test_float_accumulation_matches_runner_order(tmp_path, monkeypatch):
    directory = tmp_path / "run"
    cpu, _ = fixture(directory)
    monkeypatch.setattr(V, "weights", lambda *args: None)
    raw = V.rows(directory / "training.jsonl")
    for row in raw:
        row["latency_ms"] = 0.1
    lines(directory / "training.jsonl", raw)
    complete = V.read(directory / "completed.json")
    total = 0.0
    for row in raw:
        total += row["latency_ms"]
    complete["training_compute_ms"] = total
    write(directory / "completed.json", complete)
    aggregates = []
    for step in (400, 800, 1200):
        path = directory / f"development-{step}.jsonl"
        dev = V.rows(path)
        weighted = 0.0
        for row in dev:
            row["completion_loss"] = 0.1 if step == 800 else 0.2
            weighted += row["completion_loss"] * row["supervised_tokens"]
        lines(path, dev)
        aggregate_path = directory / f"development-{step}.json"
        aggregate = V.read(aggregate_path)
        aggregate["token_weighted_loss"] = weighted / aggregate["supervised_tokens"]
        write(aggregate_path, aggregate)
        aggregates.append(aggregate)
    selection = V.read(directory / "selection.json")
    selection["development"] = aggregates
    write(directory / "selection.json", selection)
    assert V.verify_run(directory, reseal(directory), cpu, T, H)["steps"] == 1200
    assert V.accumulated([1e16, 1.0, 1.0]) == 1e16
