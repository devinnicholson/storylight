"""Finite V5 QLoRA recipe comparison; development selection, no test input access."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import signal
import time
import traceback
from pathlib import Path

V2_SHA256 = "376016c465efdb0a030ad306d758ccd832360532ef4ee2c5cfa6b0c4bac9948d"
PROMPT_SHA256 = "16d15aca5f1fe5be105bfcdf9e5773ca3e35ce71cbd8ca16d80d823828497433"
INITIAL_SHA256 = "edb0bfc127c3e362811c341b01cae758740370ddcb5f12569338737e47c448e5"


def load_module(path, expected, name):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("unreviewed helper")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(directory, manifest, filename, count, helper, roles):
    entry = manifest["files"][filename]
    path = directory / filename
    helper.require(helper.digest(path) == entry["sha256"], "data_file_mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    helper.require(len(rows) == entry["rows"] == count, "data_count")
    for row in rows:
        helper.require(set(row) == {"id", "messages"} and isinstance(row["id"], str)
                       and row["id"], "data_schema")
        messages = row["messages"]
        helper.require([m.get("role") for m in messages] == roles, "data_roles")
        helper.require(all(set(m) == {"role", "content"} and isinstance(m["content"], str)
                           and m["content"].strip() for m in messages), "data_content")
    helper.require(len({r["id"] for r in rows}) == count, "duplicate_data_id")
    return rows


def data(args, helper):
    manifest = helper.pinned_json(args.data_dir / "manifest.json", args.data_manifest_sha256)
    development = helper.pinned_json(args.development_dir / "development-manifest.json",
                                     args.development_manifest_sha256)
    helper.require(development["training_manifest_sha256"] == args.data_manifest_sha256,
                   "development_training_binding")
    train = read_rows(args.data_dir, manifest, "train-messages.jsonl", 4800, helper,
                      ["system", "user", "assistant"])
    dev = read_rows(args.development_dir, development, "development-messages.jsonl", 256, helper,
                    ["system", "user", "assistant"])
    helper.require(not {r["id"] for r in train} & {r["id"] for r in dev}, "split_id_overlap")
    helper.require(not {r["messages"][1]["content"] for r in train}
                   & {r["messages"][1]["content"] for r in dev}, "split_source_overlap")
    prompts = {r["messages"][0]["content"] for r in train + dev}
    helper.require(len(prompts) == 1, "prompt_mismatch")
    prompt_sha = hashlib.sha256(next(iter(prompts)).encode()).hexdigest()
    helper.require(prompt_sha == development["fewshot_prompt_sha256"] == PROMPT_SHA256,
                   "prompt_pin")
    return train, dev, prompt_sha


def checkpoint_steps(steps):
    if steps == 1200:
        return (400, 800, 1200)
    if steps == 4800:
        return (1200, 2400, 4800)
    raise ValueError("unsupported training length")


def targets(recipe, model):
    if recipe == "qv":
        return {f"model.language_model.layers.{i}.self_attn.{projection}"
                for i in range(35) for projection in ("q_proj", "v_proj")
                if projection == "q_proj" or i < 15}
    if recipe != "text-linear":
        raise ValueError("unsupported adapter recipe")
    import torch

    return {name for name, module in model.named_modules()
            if re.fullmatch(r"model\.language_model\.layers\.[0-9]+\..+", name)
            and isinstance(module, torch.nn.Linear)}


def select_checkpoint(development, schedule, helper):
    helper.require([row["step"] for row in development] == list(schedule),
                   "checkpoint_schedule_incomplete")
    helper.require(all(row["rows"] == 256 and row["supervised_tokens"] > 0
                       and math.isfinite(row["token_weighted_loss"])
                       and row["token_weighted_loss"] >= 0 for row in development),
                   "invalid_development")
    helper.require(len({row["supervised_tokens"] for row in development}) == 1,
                   "development_tokens_changed")
    return min(development, key=lambda row: (row["token_weighted_loss"], row["step"]))["step"]


def save_checkpoint(model, initial, trainable, output, step, peft, torch, helper):
    from safetensors.torch import load_file

    current = {name: value.detach().cpu().float() for name, value in trainable.items()}
    delta_squared = sum(float((current[name] - initial[name]).square().sum()) for name in initial)
    helper.require(math.isfinite(delta_squared) and delta_squared > 0, "unchanged_adapter")
    directory = output / f"adapter-{step}"
    model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
    saved = load_file(str(directory / "adapter_model.safetensors"))
    expected = peft.get_peft_model_state_dict(model)
    helper.require(set(saved) == set(expected)
                   and all(torch.equal(saved[name], expected[name].cpu()) for name in saved),
                   "checkpoint_tensor_mismatch")
    helper.require(len(saved) == len(initial)
                   and all(".lora_A." in name or ".lora_B." in name for name in saved),
                   "checkpoint_contains_nonadapter_weights")
    receipt = {"step": step, "tensor_count": len(saved), "delta_l2": math.sqrt(delta_squared),
               "initial_adapter_sha256": helper.parameter_hash(initial),
               "final_adapter_sha256": helper.parameter_hash(current),
               "files": {str(path.relative_to(directory)): helper.digest(path)
                         for path in directory.rglob("*") if path.is_file()}}
    helper.write_json(output / f"checkpoint-{step}.json", receipt)
    return receipt


def train(args, helper, state):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    state["stage"] = "local_input_validation"
    helper.verify_model(args.model_dir, args.model_manifest, args.model_manifest_sha256)
    rows, dev_rows, prompt_sha = data(args, helper)
    schedule = checkpoint_steps(args.steps)
    import bitsandbytes as bnb
    import peft
    import torch
    import transformers

    runtime = helper.runtime_evidence(torch, transformers, peft, bnb)
    random.seed(helper.SEED)
    torch.manual_seed(helper.SEED)
    torch.cuda.manual_seed_all(helper.SEED)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False)
    helper.require(tokenizer.pad_token_id is not None, "missing_pad_token")
    examples = [helper.completion_example(tokenizer, row) for row in rows]
    development = [helper.completion_example(tokenizer, row) for row in dev_rows]
    helper.write_json(args.output / "protocol.json", {
        "runner_sha256": helper.digest(__file__), "helper_sha256": V2_SHA256,
        "model_id": helper.MODEL_ID, "revision": helper.REVISION,
        "model_manifest_sha256": args.model_manifest_sha256,
        "data_manifest_sha256": args.data_manifest_sha256,
        "development_manifest_sha256": args.development_manifest_sha256,
        "fewshot_prompt_sha256": prompt_sha, "training_rows": 4800, "development_rows": 256,
        "recipe": args.recipe,
        "max_steps": args.steps, "checkpoint_steps": list(schedule),
        "seed": helper.SEED, "max_length": 2048, "max_new_tokens": 256,
        "batch_size": 1, "gradient_accumulation": 1, "rank": 16, "alpha": 32,
        "dropout": 0, "learning_rate": 0.0001, "optimizer": "AdamW",
        "quantization": "nf4-double-quant-bf16", "loss": "completion-only",
        "max_train_tokens": max(len(r["input_ids"]) for r in examples),
        "max_development_tokens": max(len(r["input_ids"]) for r in development),
        "selection": "lowest token-weighted development completion loss, earliest step on tie",
        "max_runtime_seconds": args.max_runtime_seconds, "fresh_base": True,
        "screen_inputs_read": False, "evaluation_gold_read": False, "production_export": False,
        **runtime})
    state["stage"] = "model_load"
    started = time.monotonic()
    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16), dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    model = peft.prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    actual = targets(args.recipe, model)
    expected_modules = 50 if args.recipe == "qv" else 275
    expected_parameters = 2678784 if args.recipe == "qv" else 26165248
    helper.require(len(actual) == expected_modules, "unexpected_lora_module_layout")
    modules = dict(model.named_modules())
    helper.require(all(isinstance(modules[name], bnb.nn.Linear4bit) for name in actual),
                   "target_not_quantized")
    model = peft.get_peft_model(model, peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0,
        bias="none", target_modules=sorted(actual)))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    helper.require(len(trainable) == expected_modules * 2
                   and sum(p.numel() for p in trainable.values()) == expected_parameters
                   and all(".lora_A." in n or ".lora_B." in n
                                               for n in trainable), "trainable_parameters")
    initial = {n: p.detach().cpu().float().clone() for n, p in trainable.items()}
    initial_sha = helper.parameter_hash(initial)
    torch.cuda.synchronize()
    helper.write_json(args.output / "loaded.json", {
        "milliseconds": helper.elapsed_ms(started), "targets": sorted(actual),
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "initial_adapter_sha256": initial_sha,
        "expected_initial_adapter_sha256": args.expected_initial_sha256,
        **helper.memory(torch)})
    if args.recipe == "qv":
        helper.require(initial_sha == INITIAL_SHA256, "qv_initial_adapter_changed")
    if args.expected_initial_sha256:
        helper.require(initial_sha == args.expected_initial_sha256, "initial_adapter_changed")
    helper.require(args.steps != 4800 or args.expected_initial_sha256 is not None,
                   "full_run_requires_pilot_initial_pin")
    model.train()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, weight_decay=0)
    order = list(range(len(examples)))
    random.Random(helper.SEED).shuffle(order)
    checkpoints, losses = {}, []
    training_compute_ms = 0.0
    training_started = time.monotonic()
    for step, index in enumerate(order[:args.steps], 1):
        state["stage"] = "training"
        row = examples[index]
        inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda")
        labels = torch.tensor([row["labels"]], dtype=torch.long, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                           labels=labels, use_cache=False)
            loss = result.loss
        helper.require(bool(torch.isfinite(loss).item()), "nonfinite_loss")
        loss.backward()
        helper.require(all(p.grad is not None for p in trainable.values()), "missing_gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            list(trainable.values()), 1.0, error_if_nonfinite=True)
        helper.require(math.isfinite(float(norm)) and float(norm) > 0, "zero_gradient")
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = helper.elapsed_ms(started)
        training_compute_ms += elapsed
        helper.append_json(args.output / "training.jsonl", {
            "step": step, "id": row["id"], "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(norm), "supervised_tokens": row["completion_tokens"],
            "latency_ms": elapsed, **helper.memory(torch)})
        state["completed_steps"] = step
        del inputs, labels, result, loss
        if step in schedule:
            optimizer.zero_grad(set_to_none=True)
            state["stage"] = "checkpoint"
            checkpoints[step] = save_checkpoint(
                model, initial, trainable, args.output, step, peft, torch, helper)
            state["stage"] = "development_evaluation"
            receipt = helper.development_loss(model, development, args.output, step, torch)
            losses.append(receipt)
            helper.write_json(args.output / f"development-{step}.json", receipt)
            model.train()
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    state["stage"] = "checkpoint_selection"
    selected = select_checkpoint(losses, schedule, helper)
    helper.reload_checkpoint(model, args.output, checkpoints[selected], peft, torch)
    helper.write_json(args.output / "selection.json", {
        "selected_step": selected, "development": losses,
        "checkpoint_receipt_sha256": helper.digest(args.output / f"checkpoint-{selected}.json"),
        "selection_inputs": "development completion loss only",
        "selected_adapter_reloaded_and_verified": True, "screen_generation_started": False})
    return {"completed_steps": args.steps, "recipe": args.recipe, "selected_step": selected,
            "training_compute_ms": training_compute_ms,
            "training_ms": helper.elapsed_ms(training_started),
            "training_ms_includes_development_and_checkpointing": True,
            "screen_inputs_read": False, "quality_accepted": False, "production_export": False}


def bounded(args, helper, operation, maximum):
    helper.require(1 <= args.max_runtime_seconds <= maximum, "finite_bound_exceeded")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"stage": "startup", "completed_steps": 0}
    started = time.monotonic()

    def stop(signum, frame):
        raise TimeoutError("bounded_runner_stopped")

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(args.max_runtime_seconds)
    try:
        result = operation(args, helper, state)
        result.update(wall_seconds=time.monotonic() - started,
                      files={str(p.relative_to(args.output)): helper.digest(p)
                             for p in args.output.rglob("*") if p.is_file()})
        helper.write_json(args.output / "completed.json", result)
    except BaseException as error:
        helper.write_json(args.output / "failure.json", {
            **state, "error_type": type(error).__name__,
            "error_code": str(error) if isinstance(error, helper.ExperimentRejected) else None,
            "frames": [{"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
                       for f in traceback.extract_tb(error.__traceback__)[-12:]],
            "wall_seconds": time.monotonic() - started, "completed": False})
        raise SystemExit(1) from None
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths = ("v2-trainer", "model-dir", "model-manifest", "data-dir", "development-dir", "output")
    for name in paths:
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("model-manifest", "data-manifest", "development-manifest"):
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--max-runtime-seconds", type=int, required=True)
    parser.add_argument("--recipe", choices=("qv", "text-linear"), required=True)
    parser.add_argument("--steps", type=int, choices=(1200, 4800), required=True)
    parser.add_argument("--expected-initial-sha256")
    args = parser.parse_args()
    helper = load_module(args.v2_trainer, V2_SHA256, "v5_training_parent")
    bounded(args, helper, train, 7200)


if __name__ == "__main__":
    main()
