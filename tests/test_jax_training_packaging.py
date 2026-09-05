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
from training.jax_fidelity.configuration import load_config
from training.jax_fidelity.formatting import (
    completion_only_example,
    maxtext_sft_segments,
    production_messages,
)
from training.jax_fidelity.integrity import (
    DatasetIntegrityError,
    sha256_file,
    validate_dataset_manifest,
)
from training.jax_fidelity.manifests import stable_run_id
from training.jax_fidelity.prepare import (
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
