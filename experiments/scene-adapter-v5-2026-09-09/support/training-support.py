"""One finite, offline Gemma 4 text-only QLoRA experiment; never a serving export.

The setup owner downloads and hashes the pinned checkpoint before this process.
Only training/development messages and frozen screen inputs are read. Checkpoint
selection uses development completion loss; screen labels stay with the evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import signal
import statistics
import time
from pathlib import Path

MODEL_ID = "google/gemma-4-E2B-it"
REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
VERSIONS = {
    "torch": "2.10.0",
    "transformers": "5.13.0",
    "peft": "0.20.0",
    "accelerate": "1.12.0",
    "bitsandbytes": "0.49.2",
    "safetensors": "0.8.0",
}
SEED = 20260908
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 256
CHECKPOINT_STEPS = (400, 800, 1200)
PARENT_RUNNER_SHA256 = "e3fdd9845274a6ca47ae8afbfe5f04304e6a49a38eab6ed6981156abe6346364"
TARGET = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)")
SHA = re.compile(r"[0-9a-f]{64}")


class ExperimentRejected(ValueError):
    """A source-defined diagnostic code, never a third-party exception message."""


def require(condition, code):
    if not condition:
        raise ExperimentRejected(code)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def pinned_json(path, expected):
    require(isinstance(expected, str) and SHA.fullmatch(expected), "invalid_external_pin")
    require(digest(path) == expected, "external_pin_mismatch")
    return json.loads(Path(path).read_text())


def verify_model(directory, manifest_path, expected):
    manifest = pinned_json(manifest_path, expected)
    require(
        manifest.get("model_id") == MODEL_ID and manifest.get("revision") == REVISION,
        "wrong_checkpoint",
    )
    files = manifest.get("files")
    require(isinstance(files, dict) and files, "missing_checkpoint_files")
    base = directory.resolve()
    for name, sha in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe_checkpoint_path")
        path = base / relative
        # HF snapshots may contain symlinks into their local blob cache. Their bytes
        # are authenticated by the externally pinned manifest, not path trust.
        require(
            path.is_file() and SHA.fullmatch(sha) and digest(path) == sha,
            "checkpoint_file_mismatch",
        )
    actual = {str(path.relative_to(base)) for path in base.rglob("*") if path.is_file()}
    require(actual == set(files), "unlisted_checkpoint_file")
    require(
        "config.json" in files and any(name.endswith(".safetensors") for name in files),
        "checkpoint_incomplete",
    )
    config = json.loads((base / "config.json").read_text())
    require(
        config.get("architectures") == ["Gemma4ForConditionalGeneration"], "checkpoint_architecture"
    )
    text = config.get("text_config", {})
    require(
        text.get("num_hidden_layers") == 35 and text.get("num_kv_shared_layers") == 20,
        "checkpoint_attention_layout",
    )
    return manifest


def read_data(directory, manifest_sha, screen_directory, screen_manifest_sha):
    manifest = pinned_json(directory / "manifest.json", manifest_sha)
    require(manifest.get("legacy_data_read") is False, "unapproved_data_provenance")
    screen_manifest = pinned_json(screen_directory / "screen-manifest.json", screen_manifest_sha)
    require(
        screen_manifest.get("training_manifest_sha256") == manifest_sha,
        "screen_training_manifest_mismatch",
    )
    datasets = []
    for base, proof, filename, roles, count in (
        (directory, manifest, "train-messages.jsonl", ["system", "user", "assistant"], 1200),
        (directory, manifest, "development-messages.jsonl", ["system", "user", "assistant"], 120),
        (screen_directory, screen_manifest, "screen-inputs.jsonl", ["system", "user"], 64),
    ):
        entry = proof["files"][filename]
        require(digest(base / filename) == entry["sha256"], "data_file_mismatch")
        rows = [json.loads(line) for line in (base / filename).read_text().splitlines()]
        require(len(rows) == entry["rows"] == count, "unexpected_data_count")
        for row in rows:
            require(
                set(row) == {"id", "messages"} and isinstance(row["id"], str),
                "unexpected_record_schema",
            )
            messages = row["messages"]
            require([message.get("role") for message in messages] == roles, "message_roles")
            require(
                all(
                    set(message) == {"role", "content"}
                    and isinstance(message["content"], str)
                    and message["content"].strip()
                    for message in messages
                ),
                "message_content",
            )
        datasets.append(rows)
    combined = [row for split in datasets for row in split]
    require(len({row["id"] for row in combined}) == len(combined), "duplicate_id")
    sources = [{row["messages"][1]["content"] for row in split} for split in datasets]
    require(
        not any(sources[a] & sources[b] for a, b in ((0, 1), (0, 2), (1, 2))),
        "split_source_overlap",
    )
    require(len({row["messages"][0]["content"] for row in combined}) == 1, "prompt_mismatch")
    require(
        hashlib.sha256(combined[0]["messages"][0]["content"].encode()).hexdigest()
        == screen_manifest.get("fewshot_prompt_sha256"),
        "screen_prompt_pin_mismatch",
    )
    return tuple(datasets)


def chat_tokens(tokenizer, messages, generation):
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=generation,
        enable_thinking=False,
    )
    require(
        isinstance(tokens, list) and tokens and all(type(v) is int for v in tokens),
        "invalid_chat_tokens",
    )
    return tokens


def completion_example(tokenizer, row):
    prompt = chat_tokens(tokenizer, row["messages"][:-1], True)
    tokens = chat_tokens(tokenizer, row["messages"], False)
    require(
        tokens[: len(prompt)] == prompt and len(tokens) > len(prompt),
        "chat_completion_boundary_mismatch",
    )
    require(len(tokens) <= MAX_LENGTH, "training_truncation_forbidden")
    completion = tokens[len(prompt) :]
    require(len(completion) < MAX_NEW_TOKENS, "target_exceeds_generation_budget")
    decoded = tokenizer.decode(completion, skip_special_tokens=True).strip()
    require(decoded == row["messages"][-1]["content"].strip(), "completion_mask_text_mismatch")
    labels = [-100] * len(prompt) + completion
    require(
        all(label == -100 for label in labels[: len(prompt)])
        and labels[len(prompt) :] == completion,
        "completion_mask_invalid",
    )
    return {
        "id": row["id"],
        "input_ids": tokens,
        "labels": labels,
        "prompt_tokens": len(prompt),
        "completion_tokens": len(completion),
    }


def elapsed_ms(start):
    return (time.monotonic() - start) * 1000


def runtime_evidence(torch, transformers, peft, bnb):
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(
        all(versions[name].split("+")[0] == version for name, version in VERSIONS.items()),
        "dependency_version_mismatch",
    )
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1, "requires_one_cuda_device"
    )
    require(
        "L4" in torch.cuda.get_device_name(0) and torch.cuda.is_bf16_supported(), "requires_bf16_l4"
    )
    require(torch.version.cuda == "12.8", "unexpected_torch_cuda")
    sources = {}
    for obj in (
        transformers.Gemma4ForConditionalGeneration,
        peft.get_peft_model,
        peft.prepare_model_for_kbit_training,
        bnb.nn.Linear4bit,
    ):
        source = inspect.getsourcefile(obj)
        require(source is not None, "missing_runtime_source")
        sources[obj.__name__] = {"basename": Path(source).name, "sha256": digest(source)}
    return {
        "versions": versions,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "runtime_sources": sources,
    }


def memory(torch):
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def parameter_hash(parameters):
    value = hashlib.sha256()
    for name, tensor in sorted(parameters.items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().float().contiguous().numpy().tobytes())
    return value.hexdigest()


def evaluate(model, tokenizer, examples, output, arm, torch):
    model.eval()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    require(isinstance(eos, list) and eos and all(type(v) is int for v in eos), "missing_eos")
    results = []
    for row in examples:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        actual_tokens = chat_tokens(tokenizer, row["messages"], True)
        require(actual_tokens == row["input_ids"], "evaluation_tokenization_changed")
        inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
        generation_start = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                cache_implementation="dynamic",
                eos_token_id=eos,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        generation_ms = elapsed_ms(generation_start)
        tokens = generated[0, len(row["input_ids"]) :].tolist()
        ended = bool(tokens and tokens[-1] in eos)
        prediction = tokenizer.decode(tokens, skip_special_tokens=True)
        milliseconds = elapsed_ms(start)
        result = {
            "id": row["id"],
            "arm": arm,
            "latency_ms": milliseconds,
            "generation_ms": generation_ms,
            "input_tokens": len(row["input_ids"]),
            "generated_tokens": len(tokens),
            "token_ids": tokens,
            "terminated_with_eos": ended,
            "finish_reason": "eos" if ended else "generation_limit",
            "prediction": prediction,
            "output": prediction if ended else "INVALID_GENERATION_LIMIT",
            "prompt_sha256": hashlib.sha256(json.dumps(row["input_ids"]).encode()).hexdigest(),
            **memory(torch),
        }
        append_json(output / f"{arm}.jsonl", result)
        results.append(result)
        del inputs, generated
    return results


def development_loss(model, examples, output, step, torch):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    for row in examples:
        inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
        labels = torch.tensor([row["labels"]], dtype=torch.long, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                labels=labels,
                use_cache=False,
            )
        loss = float(result.loss)
        require(math.isfinite(loss) and loss >= 0, "nonfinite_development_loss")
        count = sum(value != -100 for value in row["labels"][1:])
        require(count == row["completion_tokens"] and count > 0, "development_mask_invalid")
        total_loss += loss * count
        total_tokens += count
        append_json(
            output / f"development-{step}.jsonl",
            {
                "id": row["id"],
                "step": step,
                "completion_loss": loss,
                "supervised_tokens": count,
            },
        )
        del inputs, labels, result
    torch.cuda.synchronize()
    return {
        "step": step,
        "rows": len(examples),
        "supervised_tokens": total_tokens,
        "token_weighted_loss": total_loss / total_tokens,
        "latency_ms": elapsed_ms(started),
        **memory(torch),
    }


def save_checkpoint(model, initial, trainable, output, step, peft, torch):
    from safetensors.torch import load_file

    current = {name: value.detach().cpu().float() for name, value in trainable.items()}
    delta_squared = sum(float((current[name] - initial[name]).square().sum()) for name in initial)
    require(math.isfinite(delta_squared) and delta_squared > 0, "unchanged_adapter")
    directory = output / f"adapter-{step}"
    model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
    saved = load_file(str(directory / "adapter_model.safetensors"))
    expected = peft.get_peft_model_state_dict(model)
    require(
        set(saved) == set(expected)
        and all(torch.equal(saved[name], expected[name].cpu()) for name in saved),
        "checkpoint_tensor_mismatch",
    )
    require(len(saved) == 100, "checkpoint_contains_nonadapter_weights")
    receipt = {
        "step": step,
        "tensor_count": len(saved),
        "delta_l2": math.sqrt(delta_squared),
        "initial_adapter_sha256": parameter_hash(initial),
        "final_adapter_sha256": parameter_hash(current),
        "files": {
            str(path.relative_to(directory)): digest(path)
            for path in directory.rglob("*")
            if path.is_file()
        },
    }
    write_json(output / f"checkpoint-{step}.json", receipt)
    return receipt


def select_checkpoint(development):
    require(
        [row["step"] for row in development] == list(CHECKPOINT_STEPS),
        "checkpoint_schedule_incomplete",
    )
    require(
        all(
            row["rows"] == 120
            and math.isfinite(row["token_weighted_loss"])
            and row["token_weighted_loss"] >= 0
            and row["supervised_tokens"] > 0
            for row in development
        ),
        "invalid_development",
    )
    require(
        len({row["supervised_tokens"] for row in development}) == 1, "development_tokens_changed"
    )
    return min(development, key=lambda row: (row["token_weighted_loss"], row["step"]))["step"]


def reload_checkpoint(model, output, receipt, peft, torch):
    from safetensors.torch import load_file

    directory = output / f"adapter-{receipt['step']}"
    require(
        {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
        == set(receipt["files"]),
        "checkpoint_inventory_changed",
    )
    require(
        all(digest(directory / name) == sha for name, sha in receipt["files"].items()),
        "checkpoint_hash_changed",
    )
    saved = load_file(str(directory / "adapter_model.safetensors"))
    peft.set_peft_model_state_dict(model, saved)
    loaded = peft.get_peft_model_state_dict(model)
    require(
        set(saved) == set(loaded)
        and all(torch.equal(saved[name], loaded[name].cpu()) for name in saved),
        "selected_checkpoint_reload_mismatch",
    )
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    require(
        parameter_hash(trainable) == receipt["final_adapter_sha256"],
        "selected_checkpoint_parameter_mismatch",
    )


def run(args, output, state):
    # No downloader, dataset library, remote code, or online experiment tracker is used.
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    state["stage"] = "local_input_validation"
    verify_model(args.model_dir, args.model_manifest, args.model_manifest_sha256)
    train, development, screen = read_data(
        args.data_dir,
        args.data_manifest_sha256,
        args.screen_dir,
        args.screen_manifest_sha256,
    )
    import bitsandbytes as bnb
    import peft
    import torch
    import transformers

    evidence = runtime_evidence(torch, transformers, peft, bnb)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    require(tokenizer.pad_token_id is not None, "missing_pad_token")
    examples = [completion_example(tokenizer, row) for row in train]
    development_examples = [completion_example(tokenizer, row) for row in development]
    screen_examples = [
        {
            "id": row["id"],
            "messages": row["messages"],
            "input_ids": chat_tokens(tokenizer, row["messages"], True),
        }
        for row in screen
    ]
    require(
        all(len(row["input_ids"]) + MAX_NEW_TOKENS <= MAX_LENGTH for row in screen_examples),
        "screen_context_overflow",
    )
    write_json(
        output / "protocol.json",
        {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "runner_sha256": digest(__file__),
            "parent_runner_sha256": PARENT_RUNNER_SHA256,
            "model_manifest_sha256": args.model_manifest_sha256,
            "data_manifest_sha256": args.data_manifest_sha256,
            "screen_manifest_sha256": args.screen_manifest_sha256,
            "training_rows": len(examples),
            "development_rows": len(development_examples),
            "screen_rows": len(screen_examples),
            "max_steps": args.max_steps,
            "max_runtime_seconds": args.max_runtime_seconds,
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_train_tokens": max(len(row["input_ids"]) for row in examples),
            "max_completion_tokens": max(row["completion_tokens"] for row in examples),
            "max_development_tokens": max(len(row["input_ids"]) for row in development_examples),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "selection": "lowest token-weighted development completion loss, earliest step on tie",
            "screen_generation_after_selection": True,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "rank": 16,
            "alpha": 32,
            "learning_rate": 0.0001,
            "dropout": 0,
            "optimizer": "AdamW",
            "quantization": "nf4-double-quant-bf16",
            "loss": "completion-only",
            "model_class": "Gemma4ForConditionalGeneration",
            "attention": "sdpa",
            "evaluation_gold_read": False,
            "latency_scope": (
                "chat-template-to-decoded-text; excludes wire validation and transport"
            ),
            "production_export": False,
            **evidence,
        },
    )
    state["stage"] = "model_load"
    started = time.monotonic()
    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = peft.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    targets = sorted(
        name
        for name, module in model.named_modules()
        if TARGET.fullmatch(name) and isinstance(module, bnb.nn.Linear4bit)
    )
    expected = {
        f"model.language_model.layers.{index}.self_attn.{projection}"
        for index in range(35)
        for projection in ("q_proj", "v_proj")
        if projection == "q_proj" or index < 15
    }
    require(set(targets) == expected, "unexpected_lora_module_layout")
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            target_modules=targets,
        ),
    )
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    require(
        len(trainable) == 100
        and all(".lora_A." in name or ".lora_B." in name for name in trainable),
        "unexpected_trainable_parameters",
    )
    initial = {name: value.detach().cpu().float().clone() for name, value in trainable.items()}
    torch.cuda.synchronize()
    write_json(
        output / "loaded.json",
        {
            "milliseconds": elapsed_ms(started),
            "targets": targets,
            "trainable_parameters": sum(value.numel() for value in trainable.values()),
            "initial_adapter_sha256": parameter_hash(initial),
            **memory(torch),
        },
    )
    state["stage"] = "training"
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, weight_decay=0)
    order = list(range(len(examples)))
    random.Random(SEED).shuffle(order)
    require(
        len(order) == args.max_steps and len(set(order)) == args.max_steps,
        "full_epoch_coverage_required",
    )
    development_receipts = []
    checkpoint_receipts = {}
    training_compute_ms = 0.0
    start_training = time.monotonic()
    for step in range(args.max_steps):
        row = examples[order[step]]
        inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
        labels = torch.tensor([row["labels"]], dtype=torch.long, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                labels=labels,
                use_cache=False,
            )
            loss = result.loss
        require(bool(torch.isfinite(loss).item()), "nonfinite_loss")
        loss.backward()
        require(
            all(value.grad is not None for value in trainable.values()), "missing_adapter_gradient"
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(trainable.values()), 1.0, error_if_nonfinite=True
        )
        require(math.isfinite(float(grad_norm)) and float(grad_norm) > 0, "zero_adapter_gradient")
        optimizer.step()
        torch.cuda.synchronize()
        step_ms = elapsed_ms(started)
        training_compute_ms += step_ms
        append_json(
            output / "training.jsonl",
            {
                "step": step + 1,
                "id": row["id"],
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(grad_norm),
                "supervised_tokens": row["completion_tokens"],
                "latency_ms": step_ms,
                **memory(torch),
            },
        )
        state["completed_steps"] = step + 1
        del inputs, labels, result, loss
        if step + 1 in CHECKPOINT_STEPS:
            optimizer.zero_grad(set_to_none=True)
            state["stage"] = "checkpoint"
            checkpoint_receipts[step + 1] = save_checkpoint(
                model,
                initial,
                trainable,
                output,
                step + 1,
                peft,
                torch,
            )
            state["stage"] = "development_evaluation"
            receipt = development_loss(model, development_examples, output, step + 1, torch)
            development_receipts.append(receipt)
            write_json(output / f"development-{step + 1}.json", receipt)
            model.train()
            state["stage"] = "training"
    training_ms = elapsed_ms(start_training)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    state["stage"] = "checkpoint_selection"
    selected_step = select_checkpoint(development_receipts)
    reload_checkpoint(model, output, checkpoint_receipts[selected_step], peft, torch)
    write_json(
        output / "selection.json",
        {
            "selected_step": selected_step,
            "development": development_receipts,
            "checkpoint_receipt_sha256": digest(output / f"checkpoint-{selected_step}.json"),
            "selection_inputs": "development completion loss only",
            "selected_adapter_reloaded_and_verified": True,
            "screen_generation_started": False,
        },
    )
    state["stage"] = "base_evaluation"
    with model.disable_adapter():
        base = evaluate(model, tokenizer, screen_examples, output, "base", torch)
    # Recheck the selected saved bytes and their loaded tensors immediately before
    # tuned generation, independently of the adapter-disabled baseline context.
    reload_checkpoint(model, output, checkpoint_receipts[selected_step], peft, torch)
    state["stage"] = "tuned_evaluation"
    tuned = evaluate(model, tokenizer, screen_examples, output, "tuned", torch)
    require([row["id"] for row in base] == [row["id"] for row in tuned], "evaluation_pair_mismatch")
    require(
        all(a["prompt_sha256"] == b["prompt_sha256"] for a, b in zip(base, tuned, strict=True)),
        "evaluation_prompt_mismatch",
    )
    return {
        "completed_steps": args.max_steps,
        "training_ms": training_ms,
        "training_compute_ms": training_compute_ms,
        "training_ms_includes_development_and_checkpointing": True,
        "selected_step": selected_step,
        "evaluation_rows_per_arm": len(base),
        "same_frozen_screen": True,
        "generation_limit_hits": {
            arm: sum(not row["terminated_with_eos"] for row in rows)
            for arm, rows in (("base", base), ("tuned", tuned))
        },
        "median_latency_ms": {
            arm: statistics.median(row["latency_ms"] for row in rows)
            for arm, rows in (("base", base), ("tuned", tuned))
        },
        "quality_accepted": False,
        "production_export": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "model-manifest", "data-dir", "screen-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--data-manifest-sha256", required=True)
    parser.add_argument("--screen-manifest-sha256", required=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--max-runtime-seconds", type=int, default=4200)
    args = parser.parse_args()
    require(
        args.max_steps == 1200 and 1 <= args.max_runtime_seconds <= 4200,
        "finite_bound_exceeded",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"stage": "startup", "completed_steps": 0}
    started = time.monotonic()

    def stop(signum, frame):
        raise TimeoutError("bounded_runner_stopped")

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(args.max_runtime_seconds)
    try:
        result = run(args, args.output, state)
        result["wall_seconds"] = time.monotonic() - started
        result["files"] = {
            str(path.relative_to(args.output)): digest(path)
            for path in args.output.rglob("*")
            if path.is_file()
        }
        write_json(args.output / "completed.json", result)
    except BaseException as error:
        write_json(
            args.output / "failure.json",
            {
                **state,
                "error_type": type(error).__name__,
                "error_code": str(error) if isinstance(error, ExperimentRejected) else None,
                "wall_seconds": time.monotonic() - started,
                "runner_sha256": digest(__file__),
                "completed": False,
            },
        )
        raise SystemExit(1) from None
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
