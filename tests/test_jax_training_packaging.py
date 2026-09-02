# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The repository's pytest config adds only `src` to sys.path; this package is
# intentionally kept outside the production wheel under `training`.
sys.path.insert(0, str(ROOT))

import training.jax_fidelity.train as train_module
import training.jax_fidelity.verify_runtime as verify_runtime_module
from bookforge.tensorrt_slot_client import _slot_messages
from training.jax_fidelity.commands import build_train_command
from training.jax_fidelity.configuration import ConfigError, load_config, validate_config
from training.jax_fidelity.formatting import (
    completion_only_example,
    format_training_record,
    maxtext_sft_segments,
    production_messages,
    validate_slot_target,
)
from training.jax_fidelity.integrity import (
    DatasetIntegrityError,
    sha256_file,
    validate_dataset_manifest,
)
from training.jax_fidelity.manifests import ManifestError, complete_run, stable_run_id, start_run
from training.jax_fidelity.prepare import (
    prepare_pair_deduplicated_training_jsonl,
    prepare_training_jsonl,
)
from training.jax_fidelity.runtime import approval_token
from training.jax_fidelity.verify_runtime import validate_runtime_lock, write_runtime_lock

CONFIG_PATH = ROOT / "experiments/jax-fidelity-lab/config.json"
CONFIG_V2_PATH = ROOT / "experiments/jax-fidelity-lab/config-v2.json"
CONFIG_V3_PATH = ROOT / "experiments/jax-fidelity-lab/config-v3-canary.json"
TARGET = (
    "SETTING: moonlit library\n"
    "ACTOR: small copper fox\n"
    "ACTION: opens wooden door\n"
    "MAGIC: paper birds rise"
)


class FakeGemmaTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return [byte + 1 for byte in rendered.encode()]
        return rendered

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return bytes(token_id - 1 for token_id in token_ids).decode()

    def __call__(self, text, *, truncation, max_length):
        assert truncation is False
        assert max_length > 0
        return {"input_ids": [byte + 1 for byte in text.encode()]}


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return sha256_file(path)


def test_config_freezes_text_only_completion_only_lora() -> None:
    config = load_config(CONFIG_PATH)

    assert config.production["model_id"] == "google/gemma-4-E2B-it"
    assert config.production["scan_layers"] is False
    assert config.production["use_multimodal"] is False
    assert config.training["method"] == "lora"
    assert config.training["completion_only"] is True
    assert config.training["weight_quantization"] is None
    assert config.training["rank"] in (8, 16)
    assert config.versions["maxtext_revision"] == "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45"
    assert "@sha256:" in config.versions["container_image"]

    with pytest.raises(TypeError):
        config.training["rank"] = 32


def test_v2_config_binds_prompt_pair_curriculum_and_optimizer() -> None:
    config = load_config(CONFIG_V2_PATH)

    assert config.training["rank"] == 16
    assert config.training["steps"] == 640
    assert config.training["num_epoch"] == 4
    assert config.training["packing"] is False
    assert config.training["enable_data_shuffling"] is False
    assert config.production["prompt_contract_sha256"]

    drifted = json.loads(CONFIG_V2_PATH.read_text())
    drifted["production_contract"]["prompt_contract_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="deployed prompt"):
        validate_config(drifted)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("production_contract", "scan_layers", True),
        ("production_contract", "use_multimodal", True),
        ("training", "completion_only", False),
        ("training", "weight_quantization", "int8"),
        ("training", "rank", 4),
        ("conversion", "max_kl_divergence", 0.04),
    ],
)
def test_config_fails_closed_on_architecture_or_training_drift(section, key, value) -> None:
    document = json.loads(CONFIG_PATH.read_text())
    document[section][key] = value
    with pytest.raises(ConfigError):
        validate_config(document)


def test_formatter_is_the_exact_deployed_four_slot_exchange() -> None:
    story = "A small copper fox opens a wooden door while paper birds rise."
    expected = _slot_messages(story)

    assert production_messages(story) == expected
    assert production_messages(story, target=TARGET) == expected + [
        {"role": "assistant", "content": TARGET}
    ]
    assert [message["role"] for message in expected] == ["system", "user"]
    assert "reader@example.invalid" in expected[0]["content"]
    assert "Never reproduce" in expected[0]["content"]


def test_training_record_accepts_dataset_target_mapping() -> None:
    record = {
        "id": "fidelity-1",
        "passage": "A small copper fox opens a wooden door while paper birds rise.",
        "target": {
            "SETTING": "moonlit library",
            "ACTOR": "small copper fox",
            "ACTION": "opens wooden door",
            "MAGIC": "paper birds rise",
        },
    }
    prepared = format_training_record(record)

    assert prepared["record_id"] == "fidelity-1"
    assert prepared["messages"][-1] == {"role": "assistant", "content": TARGET}


@pytest.mark.parametrize(
    "target",
    [
        "SETTING: library\nACTOR: fox\nMAGIC: birds\nACTION: opens door",
        "SETTING: library\nACTOR: fox\nACTION: opens door",
        "SETTING: library\nACTOR: fox\nACTION: opens door\nMAGIC: ",
        f"{TARGET}\nEXTRA: no",
    ],
)
def test_target_validator_rejects_repairable_or_ambiguous_outputs(target: str) -> None:
    with pytest.raises(ValueError):
        validate_slot_target(target)


def test_completion_only_supervises_only_the_final_assistant_turn() -> None:
    tokenizer = FakeGemmaTokenizer()
    messages = production_messages("A fox opens a door.", target=TARGET)
    segments = maxtext_sft_segments(tokenizer, messages)
    example = completion_only_example(
        tokenizer,
        story="A fox opens a door.",
        target=TARGET,
        input_budget_tokens=10_000,
        completion_budget_tokens=10_000,
    )

    assert [is_prompt for _, is_prompt in segments] == [True, False]
    assert example["segment_is_prompt"] == [1, 0]
    assert example["supervised_token_count"] == example["completion_token_count"]
    offset = 0
    for count, is_prompt in zip(
        example["segment_token_counts"], example["segment_is_prompt"], strict=True
    ):
        expected = [0] * count if is_prompt else example["input_ids"][offset : offset + count]
        assert example["labels"][offset : offset + count] == expected
        offset += count


def test_pair_deduplicated_preparation_is_balanced_and_final_answer_only(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "train.jsonl"
    manifest = tmp_path / "preparation.manifest.json"

    result = prepare_pair_deduplicated_training_jsonl(
        ROOT / "datasets/story-fidelity-v1/train.jsonl",
        prepared,
        manifest,
    )
    rows = [json.loads(line) for line in prepared.read_text().splitlines()]

    assert result["source_records"] == 4096
    assert result["source_pairs"] == 2048
    assert result["distinct_pairs"] == 160
    assert result["prepared_records"] == len(rows) == 320
    assert set(result["category_pair_counts"].values()) == {8}
    assert all(
        [message["role"] for message in row["messages"]]
        == ["system", "user", "assistant"]
        for row in rows
    )


def test_dataset_hash_validation_and_preparation(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    development = tmp_path / "development.jsonl"
    rows = [
        {
            "id": "one",
            "passage": "A fox opens a door.",
            "target": {
                "SETTING": "quiet hall",
                "ACTOR": "copper fox",
                "ACTION": "opens wooden door",
                "MAGIC": "paper birds rise",
            },
        }
    ]
    train_sha = _write_jsonl(train, rows)
    development_sha = _write_jsonl(development, rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {"path": "train.jsonl", "sha256": train_sha, "records": 1},
                    "development": {
                        "path": "development.jsonl",
                        "sha256": development_sha,
                        "records": 1,
                    },
                    "hidden": {"path": None, "sha256": "a" * 64, "records": 1},
                }
            },
            sort_keys=True,
        )
        + "\n"
    )
    validated = validate_dataset_manifest(
        manifest,
        expected_manifest_sha256=sha256_file(manifest),
        required_split_records={"train": 1, "development": 1, "hidden": 1},
    )
    prepared = tmp_path / "prepared.jsonl"
    result = prepare_training_jsonl(train, prepared)

    assert {split.name for split in validated.splits} == {"train", "development"}
    assert result["records"] == 1
    assert json.loads(prepared.read_text())["messages"][-1]["content"].startswith("SETTING:")

    train.write_text(train.read_text().replace("fox", "wolf", 1))
    with pytest.raises(DatasetIntegrityError, match="SHA-256 mismatch"):
        validate_dataset_manifest(manifest)


def test_repository_dataset_satisfies_the_training_contract() -> None:
    config = load_config(CONFIG_PATH)
    manifest = ROOT / config.dataset["manifest_path"]
    validated = validate_dataset_manifest(
        manifest,
        expected_manifest_sha256=sha256_file(manifest),
        required_split_records=config.dataset["required_split_records"],
    )

    assert {split_.name: split_.records for split_ in validated.splits} == {
        "development": 512,
        "train": 4096,
    }


def test_run_and_completion_manifests_are_append_only(tmp_path: Path) -> None:
    manifest = start_run(
        tmp_path,
        run_id="cpu-smoke-abc",
        stage="cpu-smoke",
        config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        command=["python3", "-m", "training.jax_fidelity"],
    )
    assert (
        start_run(
            tmp_path,
            run_id="cpu-smoke-abc",
            stage="cpu-smoke",
            config_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            command=["python3", "-m", "training.jax_fidelity"],
        )
        == manifest
    )

    artifact = tmp_path / "result.json"
    artifact.write_text("{}\n")
    completion = complete_run(
        tmp_path,
        run_id="cpu-smoke-abc",
        status="succeeded",
        artifacts=[artifact],
        evidence={"passed": True},
    )
    assert completion.is_file()

    with pytest.raises(ManifestError, match="terminal"):
        start_run(
            tmp_path,
            run_id="cpu-smoke-abc",
            stage="cpu-smoke",
            config_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            command=["different"],
        )


def test_maxtext_command_retains_every_safety_override() -> None:
    config = load_config(CONFIG_PATH)
    command = build_train_command(
        config,
        maxtext_checkpoint="/checkpoints/base/items",
        hf_tokenizer_checkpoint="/hf/base",
        prepared_train_jsonl="/data/train.jsonl",
        output_directory="/output",
        run_name="smoke",
        hardware="gpu",
        smoke=True,
    )
    joined = " ".join(command)

    assert "model_name=gemma4-e2b" in command
    assert "tokenizer_path=/hf/base" in command
    assert "src/maxtext/configs/post_train/sft.yml" in command
    assert "dataset_type=hf" in command
    assert "hardware=gpu" in command
    assert "skip_jax_distributed_system=true" in command
    assert "scan_layers=false" in command
    assert "use_multimodal=false" in command
    assert "sft_train_on_completion_only=True" in command
    assert "lora.enable_lora=True" in command
    assert "lora.lora_rank=8" in command
    assert "lora.lora_weight_qtype" not in joined
    assert "steps=5" in command


def test_v2_train_command_makes_exposure_and_optimizer_explicit() -> None:
    config = load_config(CONFIG_V2_PATH)
    command = build_train_command(
        config,
        maxtext_checkpoint="/checkpoints/base/items",
        hf_tokenizer_checkpoint="/hf/base",
        prepared_train_jsonl="/data/train.jsonl",
        output_directory="/output",
        run_name="full-v2",
        hardware="gpu",
        smoke=False,
    )

    assert "steps=640" in command
    assert "lora.lora_rank=16" in command
    assert "lora.lora_alpha=32.0" in command
    assert "packing=false" in command
    assert "num_epoch=4" in command
    assert "enable_data_shuffling=false" in command
    assert "enable_dropout=false" in command
    assert "gradient_accumulation_steps=1" in command
    assert "lr_schedule_type=cosine" in command
    assert "warmup_steps_fraction=0.05" in command
    assert "learning_rate_final_fraction=0.1" in command
    assert "adam_weight_decay=0.0" in command
    assert "opt_type=adamw" in command
    assert "skip_step_on_spikes=false" in command
    assert "trainable_parameters_mask=[]" in command
    assert "checkpoint_period=160" in command


def test_v3_train_command_uses_l4_safe_attention() -> None:
    config = load_config(CONFIG_V3_PATH)
    command = build_train_command(
        config,
        maxtext_checkpoint="/checkpoints/base/items",
        hf_tokenizer_checkpoint="/hf/base",
        prepared_train_jsonl="/data/train.jsonl",
        output_directory="/output",
        run_name="recovery-smoke",
        hardware="gpu",
        smoke=True,
    )

    assert "attention=dot_product" in command


def test_pinned_native_maxtext_patch_materializes_lora_before_optimizer() -> None:
    patch = (
        ROOT
        / "training/jax_fidelity/patches/maxtext-native-lora-materialization.patch"
    ).read_text()

    assert "model = lora_utils.apply_lora_to_model(model, None, config)" in patch
    assert "model = lora_utils.apply_lora_to_model(model, mesh, config)" not in patch
    assert "if lora_enabled:" in patch
    assert "src/maxtext/utils/train_utils.py" in patch
    assert "nnx.state(new_state.model, train_param_type)" in patch
    assert "src/maxtext/trainers/pre_train/train.py" in patch
    assert "Bookforge native LoRA census" in patch
    assert "BOOKFORGE_EXPECTED_LORA_PAIR_COUNT" in patch
    assert "index 35f47e59..73b808a5 100644" in patch
    assert 'scalar_metrics["learning/update_norm"]' in patch
    assert 'scalar_metrics["learning/changed_trainable_leaves"]' in patch
    assert "typed path collision" in patch
    assert "isinstance(value, jax.Array)" in patch
    assert "jnp.issubdtype(value.dtype, jnp.floating)" in patch
    assert "jnp.all(jnp.isfinite(value))" in patch
    assert "value.shape != adapter.shape" in patch
    assert "value.sharding.mesh != adapter.sharding.mesh" in patch
    assert "value.sharding.spec != adapter.sharding.spec" in patch
    assert "value.sharding.memory_kind != adapter.sharding.memory_kind" in patch
    assert 'any(part in ("mu", "nu") for part in parts)' in patch
    assert "optimizer_typed_paths[parts][3:] != lora_typed_paths[parts[3:]]" in patch

    lines = patch.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("@@ "):
            continue
        header = line.split("@@", 2)[1].strip().split()
        old_count = int(header[0].split(",", 1)[1]) if "," in header[0] else 1
        new_count = int(header[1].split(",", 1)[1]) if "," in header[1] else 1
        body = []
        for candidate in lines[index + 1 :]:
            if candidate.startswith(("@@ ", "diff --git ")):
                break
            body.append(candidate)
        assert sum(not row.startswith("+") for row in body) == old_count
        assert sum(not row.startswith("-") for row in body) == new_count


def test_container_and_direct_dependencies_are_immutable() -> None:
    dockerfile = (ROOT / "training/jax_fidelity/Dockerfile").read_text()
    lock = (ROOT / "training/jax_fidelity/requirements.lock").read_text()

    assert "FROM python:3.12.11-slim-bookworm@sha256:" in dockerfile
    assert "jax[cuda12]==0.11.0" in lock
    assert "jax-cuda12-pjrt==0.11.0" in lock
    assert "jax-cuda12-plugin==0.11.0" in lock
    assert "flax==0.12.8" in lock
    assert "optax==0.2.8" in lock
    assert "safetensors==0.8.0" in lock
    assert "transformers==5.13.0" in lock
    assert "pydantic-settings==2.15.0" in lock
    assert "orbax-checkpoint==0.12.2" in lock
    assert "google-cloud-secret-manager" not in lock
    assert "google-cloud-storage==3.13.1" in lock
    assert "maxtext[cuda12] @ file:///opt/MaxText" in lock
    assert "git+https://github.com/AI-Hypercomputer/maxtext" not in lock
    assert "torch==2.10.0+cpu" in lock
    assert "https://download.pytorch.org/whl/cpu" in lock
    assert "maxtext[cuda12]" in lock
    assert "nvidia-nvtx-cu12==12.9.79" in lock
    assert "nvidia-curand-cu12==10.3.10.19" in lock
    assert "transformer-engine-jax==2.18.0" in lock
    assert "tpu-post-train" not in lock
    assert "git clone --filter=blob:none --no-checkout" in dockerfile
    assert "build-essential" in dockerfile
    assert "NVTE_BUILD_USE_NVIDIA_WHEELS=1" in dockerfile
    assert "nvidia/nccl/lib/libnccl.so.2" in dockerfile
    assert "nvidia/curand/lib" in dockerfile
    assert "ln -sfnT cuda_runtime" in dockerfile
    assert "nvidia/cudart/lib/lib*.so.*[0-9]" in dockerfile
    assert "LD_LIBRARY_PATH=" in dockerfile
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95" in dockerfile
    assert (
        "BOOKFORGE_MAXTEXT_APPROVED_PATCH=/opt/bookforge/patches/"
        "maxtext-native-lora-materialization.patch"
    ) in dockerfile
    assert "git -C /opt/MaxText apply --check --unidiff-zero" in dockerfile
    assert (
        'git -C /opt/MaxText apply --unidiff-zero "$BOOKFORGE_MAXTEXT_APPROVED_PATCH"'
        in dockerfile
    )
    assert (
        "git -C /opt/MaxText diff --no-ext-diff --binary --abbrev=8 --unified=0"
        in dockerfile
    )
    assert "src/maxtext/trainers/pre_train/train.py" in dockerfile
    assert "$(printf '%s\\n%s'" in dockerfile
    assert 'cmp -s - "$BOOKFORGE_MAXTEXT_APPROVED_PATCH"' in dockerfile
    assert "-Wl,-rpath,/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib" in dockerfile
    assert "--no-build-isolation 'transformer-engine-jax==2.18.0'" in dockerfile
    assert (
        "git -C /opt/MaxText checkout --detach "
        "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45" in dockerfile
    )
    assert "--write-lock /opt/bookforge/runtime.lock.json" in dockerfile
    assert "PYTHONPATH=/opt/MaxText/src:/opt/bookforge:/opt/bookforge/src" in dockerfile
    assert "--maxtext-root /opt/MaxText" in dockerfile


def test_full_runtime_lock_detects_installed_dependency_drift(tmp_path: Path) -> None:
    lock = tmp_path / "runtime.lock.json"
    write_runtime_lock(lock)
    validate_runtime_lock(lock)

    document = json.loads(lock.read_text())
    document["packages"][0]["version"] = "changed"
    lock.chmod(0o600)
    lock.write_text(json.dumps(document) + "\n")
    with pytest.raises(RuntimeError, match="full installed dependency set"):
        validate_runtime_lock(lock)


def test_runtime_lock_ignores_modal_control_plane_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Distribution:
        def __init__(self, version: str, root: Path) -> None:
            self.metadata = {"Name": "multidict"}
            self.version = version
            self.root = root

        def locate_file(self, _: str) -> Path:
            return self.root

    environment_package = Distribution(
        "6.7.1", Path(sys.prefix) / "lib/python3.12/site-packages"
    )
    modal_overlay = Distribution("6.6.0", tmp_path / "modal-control-plane")
    monkeypatch.setattr(
        verify_runtime_module.importlib.metadata,
        "distributions",
        lambda: [environment_package, modal_overlay],
    )

    document = verify_runtime_module.runtime_lock_document()

    assert document["packages"] == [{"name": "multidict", "version": "6.7.1"}]


def test_executed_training_writes_nonempty_terminal_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    dataset_manifest = ROOT / config.dataset["manifest_path"]
    prepared = tmp_path / "prepared.jsonl"
    prepared.write_text('{"messages":[]}\n')
    base = tmp_path / "base"
    tokenizer = tmp_path / "tokenizer"
    base.mkdir()
    tokenizer.mkdir()
    (base / "checkpoint").write_bytes(b"base")
    (tokenizer / "tokenizer.json").write_bytes(b"tokenizer")
    output = tmp_path / "output"
    runs = tmp_path / "runs"
    run_id = stable_run_id(
        stage="lora-smoke",
        config_sha256=config.sha256,
        dataset_manifest_sha256=sha256_file(dataset_manifest),
    )
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv(
        "BOOKFORGE_JAX_EXECUTION_APPROVAL",
        approval_token(
            stage="lora-smoke",
            run_id=run_id,
            config_sha256=config.sha256,
            input_sha256=sha256_file(prepared),
        ),
    )
    monkeypatch.setattr(train_module, "validate_maxtext_checkout", lambda *_: tmp_path)
    monkeypatch.setattr(train_module, "validate_maxtext_import_provenance", lambda *_: {})
    monkeypatch.setattr(train_module, "validate_runtime", lambda: None)
    runtime_lock = tmp_path / "runtime.lock.json"
    write_runtime_lock(runtime_lock)
    monkeypatch.setenv("BOOKFORGE_JAX_RUNTIME_LOCK", str(runtime_lock))

    def fake_run(command, *, cwd, environment=None):
        assert cwd == tmp_path
        assert "hardware=gpu" in command
        assert environment is not None
        assert "BOOKFORGE_EXPECTED_LORA_PAIR_COUNT" not in environment
        output.mkdir()
        (output / "adapter-checkpoint").write_bytes(b"lora")

    monkeypatch.setattr(train_module, "run_checked", fake_run)
    monkeypatch.setattr(
        train_module,
        "verify_tensorboard_learning",
        lambda *_args, **_kwargs: {
            "schema_version": "bookforge-jax-learning-evidence-v1",
            "status": "passed",
            "optimizer_steps": 5,
        },
    )
    monkeypatch.setattr(
        train_module,
        "discover_orbax_items",
        lambda *_args, **_kwargs: output,
    )
    monkeypatch.setattr(
        train_module,
        "lora_checkpoint_evidence",
        lambda *_args, **_kwargs: {
            "schema_version": "1.0",
            "format": "maxtext-orbax-lora-tree",
            "lora_pair_count": 1,
            "rank": 8,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            str(CONFIG_PATH),
            "--dataset-manifest",
            str(dataset_manifest),
            "--dataset-manifest-sha256",
            sha256_file(dataset_manifest),
            "--prepared-train-jsonl",
            str(prepared),
            "--prepared-train-sha256",
            sha256_file(prepared),
            "--base-checkpoint",
            str(base),
            "--hf-tokenizer-checkpoint",
            str(tokenizer),
            "--output-directory",
            str(output),
            "--run-directory",
            str(runs),
            "--maxtext-root",
            str(tmp_path),
            "--smoke",
            "--execute",
        ],
    )

    train_module.main()

    completion = json.loads((runs / run_id / "completion.json").read_text())
    assert completion["status"] == "succeeded"
    assert completion["artifacts"]
    assert completion["evidence"]["inputs"]["base_checkpoint"]["content_sha256"]
    assert completion["evidence"]["learning"]["status"] == "passed"
    assert completion["evidence"]["terminal_adapter"]["lora_pair_count"] == 1
