"""Finite paired constrained-decoding screen; no training, downloads, or serving.

The setup owner supplies verified local weights, V2 proofs, a frozen grammar,
and a separately authored input-only screen. Every constrained request owns a
fresh matcher. Failed or truncated generations are never rewritten as REFUSE.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import signal
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

PARENT_SHA256 = "a7f42e5331aa8b2ff286833b6fcdc27b28f3f889324203cedea80b354e95803e"
TRAINER_SHA256 = "376016c465efdb0a030ad306d758ccd832360532ef4ee2c5cfa6b0c4bac9948d"
XGRAMMAR_VERSION = "0.2.6"
ARMS = ("unconstrained", "constrained")


def load_parent(directory):
    path = directory / "inference_variants.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA256:
        raise ValueError("unreviewed inference helper")
    spec = importlib.util.spec_from_file_location("reviewed_v2_inference", path)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    trainer = parent.trainer_module()
    if trainer.digest(directory / "train.py") != TRAINER_SHA256:
        raise ValueError("unreviewed trainer")
    return parent, trainer


def schedule(rows, preflight=False):
    expected = 2 if preflight else 64
    if len(rows) != expected or len({row["id"] for row in rows}) != expected:
        raise ValueError("unexpected distinct case count")
    result = []
    for index, row in enumerate(rows):
        for repetition in range(1 if preflight else 2):
            arms = ARMS if (index + repetition) % 2 == 0 else tuple(reversed(ARMS))
            for arm in arms:
                result.append(
                    {
                        "dispatch_ordinal": len(result),
                        "id": row["id"],
                        "case_index": index,
                        "repetition": repetition,
                        "arm": arm,
                    }
                )
    return result


def input_proofs(args, parent, trainer):
    old_args = SimpleNamespace(
        training_run=args.training_run,
        completed_sha256=args.completed_sha256,
        screen_dir=args.v2_screen_dir,
        screen_manifest_sha256=args.v2_screen_manifest_sha256,
        model_manifest_sha256=args.model_manifest_sha256,
    )
    old_rows, adapter_dir, checkpoint, _, old_pins = parent.run_inputs(old_args, trainer)
    trainer.require(checkpoint["step"] == 800, "wrong_selected_checkpoint")
    old_prompt = old_rows[0]["messages"][0]["content"]
    prompt_sha = hashlib.sha256(old_prompt.encode()).hexdigest()
    if args.preflight_only:
        trainer.require(
            args.training_manifest and args.training_messages, "preflight_files_missing"
        )
        manifest = trainer.pinned_json(args.training_manifest, old_pins["data_manifest_sha256"])
        training_entry = manifest["files"]["train-messages.jsonl"]
        trainer.require(
            trainer.digest(args.training_messages) == training_entry["sha256"],
            "preflight_training_changed",
        )
        training = [json.loads(line) for line in args.training_messages.read_text().splitlines()]
        trainer.require(len(training) == training_entry["rows"] == 1200, "preflight_training_count")
        trainer.require(
            all(
                [item["role"] for item in row["messages"]] == ["system", "user", "assistant"]
                for row in training
            ),
            "preflight_training_roles",
        )
        positive = next(
            (row for row in training if row["messages"][-1]["content"].startswith("V2\n")), None
        )
        refusal = next(
            (row for row in training if row["messages"][-1]["content"] == "REFUSE"), None
        )
        trainer.require(positive is not None and refusal is not None, "preflight_branches_missing")
        rows = [{"id": row["id"], "messages": row["messages"][:-1]} for row in (positive, refusal)]
        input_sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    else:
        trainer.require(args.screen_dir and args.screen_manifest_sha256, "screen_files_missing")
        manifest = trainer.pinned_json(
            args.screen_dir / "screen-manifest.json",
            args.screen_manifest_sha256,
        )
        entry = manifest["files"]["screen-inputs.jsonl"]
        trainer.require(
            trainer.digest(args.screen_dir / "screen-inputs.jsonl") == entry["sha256"],
            "screen_input_changed",
        )
        trainer.require(
            manifest["training_manifest_sha256"] == old_pins["data_manifest_sha256"]
            and manifest["selected_step"] == 800
            and manifest["selected_adapter_sha256"]
            == checkpoint["files"]["adapter_model.safetensors"],
            "screen_adapter_binding",
        )
        rows = [
            json.loads(line)
            for line in (args.screen_dir / "screen-inputs.jsonl").read_text().splitlines()
        ]
        trainer.require(len(rows) == entry["rows"] == 64, "screen_count")
        trainer.require(prompt_sha == manifest["fewshot_prompt_sha256"], "prompt_pin_changed")
        input_sha = entry["sha256"]
    for row in rows:
        trainer.require(
            set(row) == {"id", "messages"}
            and isinstance(row["id"], str)
            and [item["role"] for item in row["messages"]] == ["system", "user"]
            and all(
                set(item) == {"role", "content"}
                and isinstance(item["content"], str)
                and item["content"].strip()
                for item in row["messages"]
            )
            and row["messages"][0]["content"] == old_prompt,
            "screen_schema",
        )
    trainer.require(
        not (
            {row["messages"][1]["content"] for row in rows}
            & {row["messages"][1]["content"] for row in old_rows}
        ),
        "old_screen_overlap",
    )
    trainer.require(trainer.digest(args.grammar) == args.grammar_sha256, "grammar_pin_changed")
    grammar = args.grammar.read_text()
    trainer.require(0 < len(grammar.encode()) <= 65536, "grammar_bounds")
    plan = schedule(rows, args.preflight_only)
    return (
        rows,
        adapter_dir,
        checkpoint,
        grammar,
        plan,
        {
            "data_manifest_sha256": old_pins["data_manifest_sha256"],
            "input_sha256": input_sha,
            "fewshot_prompt_sha256": prompt_sha,
        },
    )


class TimedProcessor:
    """Host callback wall time includes CPU work, synchronizations and GPU dispatch."""

    def __init__(self, processor):
        self.processor = processor
        self.calls = 0
        self.wall_ms = 0.0

    def __call__(self, input_ids, scores):
        started = time.monotonic()
        try:
            return self.processor(input_ids, scores)
        finally:
            self.calls += 1
            self.wall_ms += (time.monotonic() - started) * 1000


def grammar_accepts(compiled, tokens, xgr):
    matcher = xgr.GrammarMatcher(compiled)
    return all(matcher.accept_token(token) for token in tokens) and matcher.is_terminated()


def generate_one(model, tokenizer, example, dispatch, compiled, xgr, hf_processor, torch, trainer):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    input_ids = trainer.chat_tokens(tokenizer, example["messages"], True)
    trainer.require(input_ids == example["input_ids"], "tokenization_changed")
    inputs = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    processor = TimedProcessor(hf_processor(compiled)) if dispatch["arm"] == "constrained" else None
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    torch.cuda.synchronize()
    generation_started = time.monotonic()
    with torch.inference_mode():
        result = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            cache_implementation="dynamic",
            eos_token_id=eos,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[processor] if processor else [],
        )
    torch.cuda.synchronize()
    generation_ms = trainer.elapsed_ms(generation_started)
    tokens = result[0, len(input_ids) :].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    latency_ms = trainer.elapsed_ms(started)
    ended = bool(tokens and tokens[-1] in eos)
    verify_started = time.monotonic()
    accepted = grammar_accepts(compiled, tokens, xgr)
    verify_ms = trainer.elapsed_ms(verify_started)
    output = text if ended and len(text.encode()) <= 4096 else "INVALID_GENERATION_LIMIT_OR_SIZE"
    return {
        **dispatch,
        "prediction": text,
        "output": output,
        "token_ids": tokens,
        "input_tokens": len(input_ids),
        "generated_tokens": len(tokens),
        "prompt_sha256": hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(),
        "terminated_with_eos": ended,
        "finish_reason": "eos" if ended else "generation_limit",
        "grammar_accepts_complete_tokens": accepted,
        "grammar_verification_ms": verify_ms,
        "grammar_host_callback_ms": processor.wall_ms if processor else 0.0,
        "grammar_callback_calls": processor.calls if processor else 0,
        "latency_ms": latency_ms,
        "generation_ms": generation_ms,
        **trainer.memory(torch),
    }


def execute(args, parent, trainer, state):
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    state["stage"] = "input_proofs"
    rows, adapter_dir, checkpoint, grammar, plan, pins = input_proofs(args, parent, trainer)
    trainer.verify_model(args.model_dir, args.model_manifest, args.model_manifest_sha256)
    import bitsandbytes as bnb
    import peft
    import torch
    import transformers
    import xgrammar as xgr
    from safetensors.torch import load_file
    from xgrammar.contrib.hf import LogitsProcessor

    trainer.require(importlib.metadata.version("xgrammar") == XGRAMMAR_VERSION, "xgrammar_version")
    runtime = trainer.runtime_evidence(torch, transformers, peft, bnb)
    runtime["versions"]["xgrammar"] = XGRAMMAR_VERSION
    runtime["xgrammar_sources"] = {
        name: trainer.digest(inspect.getsourcefile(value))
        for name, value in (
            ("hf_processor", LogitsProcessor),
            ("tokenizer_info", xgr.TokenizerInfo),
            ("matcher", xgr.GrammarMatcher),
            ("compiler", xgr.GrammarCompiler),
        )
    }
    torch.manual_seed(trainer.SEED)
    torch.cuda.manual_seed_all(trainer.SEED)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    trainer.require(tokenizer.pad_token_id is not None, "missing_pad_token")
    examples = [
        {**row, "input_ids": trainer.chat_tokens(tokenizer, row["messages"], True)} for row in rows
    ]
    trainer.require(
        all(len(row["input_ids"]) + 256 <= 2048 for row in examples), "context_overflow"
    )
    trainer.write_json(
        args.output / "protocol.json",
        {
            "runner_sha256": trainer.digest(__file__),
            "parent_sha256": PARENT_SHA256,
            "trainer_sha256": TRAINER_SHA256,
            "model_manifest_sha256": args.model_manifest_sha256,
            "training_completed_sha256": args.completed_sha256,
            "v2_screen_manifest_sha256": args.v2_screen_manifest_sha256,
            "screen_manifest_sha256": args.screen_manifest_sha256,
            "grammar_sha256": args.grammar_sha256,
            "selected_step": 800,
            "adapter_files": checkpoint["files"],
            "calls": len(plan),
            "repetitions": 1 if args.preflight_only else 2,
            "preflight_only": args.preflight_only,
            "preflight_selection": "first V2 and first REFUSE training rows"
            if args.preflight_only
            else None,
            "arms": list(ARMS),
            "max_new_tokens": 256,
            "max_runtime_seconds": args.max_runtime_seconds,
            "schedule": plan,
            "greedy": True,
            "precision": "bf16-base-safe-merged-selected-adapter",
            "gold_read": False,
            "production_export": False,
            **pins,
            **runtime,
        },
    )
    state["stage"] = "model_load"
    load_started = time.monotonic()
    base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    targets = {
        f"model.language_model.layers.{index}.self_attn.{projection}"
        for index in range(35)
        for projection in ("q_proj", "v_proj")
        if projection == "q_proj" or index < 15
    }
    modules = {name: module for name, module in base.named_modules() if name in targets}
    trainer.require(
        set(modules) == targets
        and not any(isinstance(module, bnb.nn.Linear4bit) for module in base.modules())
        and all(
            isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.bfloat16
            for module in modules.values()
        ),
        "bf16_base_layout",
    )
    model = peft.PeftModel.from_pretrained(
        base, adapter_dir, is_trainable=False, local_files_only=True
    )
    config = model.peft_config["default"]
    trainer.require(
        set(config.target_modules) == targets
        and config.r == 16
        and config.lora_alpha == 32
        and config.lora_dropout == 0
        and config.bias == "none"
        and not config.modules_to_save,
        "adapter_config_changed",
    )
    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    loaded = peft.get_peft_model_state_dict(model)
    trainer.require(
        set(saved) == set(loaded)
        and len(saved) == 100
        and all(torch.equal(saved[name], loaded[name].cpu()) for name in saved),
        "adapter_tensor_mismatch",
    )
    lora = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    trainer.require(
        trainer.parameter_hash(lora) == checkpoint["final_adapter_sha256"],
        "adapter_parameter_mismatch",
    )
    torch.cuda.synchronize()
    merge_started = time.monotonic()
    model = model.merge_and_unload(safe_merge=True)
    model.eval()
    torch.cuda.synchronize()
    merge_ms = trainer.elapsed_ms(merge_started)
    trainer.require(
        not any(parameter.requires_grad for parameter in model.parameters())
        and not any(".lora_" in name for name, _ in model.named_parameters())
        and all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()),
        "invalid_merged_model",
    )
    model_ms = trainer.elapsed_ms(load_started)
    state["stage"] = "grammar_compile"
    compile_started = time.monotonic()
    vocab_size = model.get_output_embeddings().weight.shape[0]
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    trainer.require(
        vocab_size == 262144 and isinstance(eos, list) and set(eos) == {1, 106, 50},
        "unexpected_vocabulary_or_eos",
    )
    token_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=vocab_size, stop_token_ids=eos
    )
    compiler = xgr.GrammarCompiler(token_info)
    compiled = compiler.compile_grammar(xgr.Grammar.from_ebnf(grammar))
    compile_ms = trainer.elapsed_ms(compile_started)
    trainer.write_json(
        args.output / "loaded.json",
        {
            "model_load_and_merge_ms": model_ms,
            "merge_ms": merge_ms,
            "grammar_compile_ms": compile_ms,
            "vocab_size": vocab_size,
            "stop_token_ids": eos,
            "dtype_inventory": parent.parameter_inventory(model),
            "adapter_tensors_verified": 100,
            **trainer.memory(torch),
        },
    )
    state["stage"] = "generation"
    records = []
    for dispatch in plan:
        state["dispatch"] = dispatch
        trainer.append_json(args.output / "dispatch.jsonl", dispatch)
        record = generate_one(
            model,
            tokenizer,
            examples[dispatch["case_index"]],
            dispatch,
            compiled,
            xgr,
            LogitsProcessor,
            torch,
            trainer,
        )
        trainer.append_json(args.output / "raw.jsonl", record)
        records.append(record)
        if dispatch["arm"] == "constrained" and record["terminated_with_eos"]:
            trainer.require(record["grammar_accepts_complete_tokens"], "constraint_violation")
    arms = {}
    if args.preflight_only:
        trainer.require(
            all(
                row["terminated_with_eos"] and row["grammar_accepts_complete_tokens"]
                for row in records
                if row["arm"] == "constrained"
            ),
            "preflight_constrained_completion_failed",
        )
    for arm in ARMS:
        selected = [row for row in records if row["arm"] == arm]
        arms[arm] = {
            "calls": len(selected),
            "median_latency_ms": trainer.statistics.median(row["latency_ms"] for row in selected),
            "median_generated_tokens": trainer.statistics.median(
                row["generated_tokens"] for row in selected
            ),
            "generation_limit_hits": sum(not row["terminated_with_eos"] for row in selected),
            "grammar_complete": sum(row["grammar_accepts_complete_tokens"] for row in selected),
            "host_callback_ms": sum(row["grammar_host_callback_ms"] for row in selected),
        }
    trainer.write_json(
        args.output / "predictions.json",
        {
            "schema_version": 1,
            "input_sha256": pins["input_sha256"],
            "provenance": {
                "implementation": (
                    f"V3 BF16-merged paired XGrammar screen @{trainer.digest(__file__)}"
                ),
                "runner_sha256": trainer.digest(__file__),
                "grammar_sha256": args.grammar_sha256,
                "screen_manifest_sha256": args.screen_manifest_sha256,
                "training_completed_sha256": args.completed_sha256,
                "raw_sha256": trainer.digest(args.output / "raw.jsonl"),
                "preflight_only": args.preflight_only,
                "evaluation_gold_read": False,
                **pins,
            },
            "records": [
                {key: row[key] for key in ("id", "arm", "repetition", "output", "latency_ms")}
                for row in records
            ],
        },
    )
    return {
        "calls": len(records),
        "preflight_only": args.preflight_only,
        "arms": arms,
        "quality_accepted": False,
        "production_export": False,
        "callback_timing_scope": (
            "host elapsed includes synchronization and GPU dispatch, not pure masking cost"
        ),
        "latency_scope": (
            "tokenization through decoded text; excludes grammar verification, load and transport"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-dir",
        "model-dir",
        "model-manifest",
        "training-run",
        "v2-screen-dir",
        "grammar",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "model-manifest-sha256",
        "completed-sha256",
        "v2-screen-manifest-sha256",
        "grammar-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--max-runtime-seconds", type=int, default=1500)
    parser.add_argument("--screen-dir", type=Path)
    parser.add_argument("--screen-manifest-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--training-messages", type=Path)
    args = parser.parse_args()
    parent, trainer = load_parent(args.parent_dir)
    trainer.require(1 <= args.max_runtime_seconds <= 1500, "finite_bound_exceeded")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {"stage": "startup"}

    def stop(signum, frame):
        raise TimeoutError("bounded screen stopped")

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(args.max_runtime_seconds)
    try:
        result = execute(args, parent, trainer, state)
        result["wall_seconds"] = time.monotonic() - started
        result["files"] = {
            str(path.relative_to(args.output)): trainer.digest(path)
            for path in args.output.rglob("*")
            if path.is_file()
        }
        trainer.write_json(args.output / "completed.json", result)
    except BaseException as error:
        trainer.write_json(
            args.output / "failure.json",
            {
                **state,
                "error_type": type(error).__name__,
                "error_code": str(error) if isinstance(error, trainer.ExperimentRejected) else None,
                "frames": [
                    {
                        "file": Path(frame.filename).name,
                        "function": frame.name,
                        "line": frame.lineno,
                    }
                    for frame in traceback.extract_tb(error.__traceback__)[-12:]
                ],
                "wall_seconds": time.monotonic() - started,
                "runner_sha256": trainer.digest(__file__),
                "completed": False,
            },
        )
        raise SystemExit(1) from None
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
