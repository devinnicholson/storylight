"""Offline audit of the fixed training-only BF16 merge diagnostic; no model inference."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE_SHA = "300cd393fc0ba40b1a34270579adb6d4d173ad19bbf6451563f5b94a8adb1ade"
VERIFIER_SHA = "76bb8b2cfd135962c42ad7af9dc5600ae0a2c4c7424fe3bbbf3ea9083f17c78d"
DATA_SHA = "a8d5cd1c73258e858f4eaee3c140d50ac61c1e0f0e871e4dad96dac7160862a9"
MODEL_SHA = "703bbb89d61aaed083846d7cb3d4ee1a68220e25de93a035b1f4b49d24062f2d"


def load(path, sha, name):
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError("verifier source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v = load(HERE / "verify-training.py", VERIFIER_SHA, "profile_inventory")
profile = load(HERE / "serving-profile.py", PROFILE_SHA, "profile_source")


def check_records(raw, plan):
    v.require(len(raw) == len(plan) == 32, "incomplete profile")
    fields = {
        "arm",
        "prediction",
        "output",
        "token_ids",
        "input_tokens",
        "generated_tokens",
        "prompt_sha256",
        "terminated_with_eos",
        "finish_reason",
        "grammar_accepts_complete_tokens",
        "grammar_verification_ms",
        "grammar_host_callback_ms",
        "grammar_callback_calls",
        "latency_ms",
        "generation_ms",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
    for row, dispatch in zip(raw, plan, strict=True):
        v.require(set(row) == fields | set(dispatch), "raw fields")
        v.require(all(v.same(row[k], val) for k, val in dispatch.items()), "dispatch order")
        tokens = row["token_ids"]
        v.require(
            isinstance(tokens, list)
            and 1 <= len(tokens) <= 256
            and all(type(t) is int and 0 <= t < 262144 for t in tokens),
            "token IDs",
        )
        v.require(
            tokens[-1] in (1, 106, 50) and not any(t in (1, 106, 50) for t in tokens[:-1]),
            "EOS placement",
        )
        v.require(
            row["arm"] == "constrained"
            and row["terminated_with_eos"] is True
            and row["grammar_accepts_complete_tokens"] is True
            and row["finish_reason"] == "eos",
            "complete grammar output",
        )
        v.require(
            type(row["generated_tokens"]) is int
            and row["generated_tokens"] == len(tokens)
            and type(row["input_tokens"]) is int
            and 1 <= row["input_tokens"] <= 1792,
            "token count",
        )
        v.require(
            isinstance(row["prediction"], str)
            and len(row["prediction"].encode()) <= 4096
            and row["output"] == row["prediction"],
            "output mapping",
        )
        v.require(
            isinstance(row["prompt_sha256"], str) and v.SHA.fullmatch(row["prompt_sha256"]),
            "prompt hash",
        )
        for key in (
            "latency_ms",
            "generation_ms",
            "grammar_verification_ms",
            "grammar_host_callback_ms",
        ):
            v.require(v.finite(row[key], positive=key in ("latency_ms", "generation_ms")), "timing")
        v.require(
            row["grammar_host_callback_ms"] <= row["generation_ms"] <= row["latency_ms"],
            "timing scope",
        )
        v.require(
            type(row["grammar_callback_calls"]) is int
            and row["grammar_callback_calls"] == len(tokens),
            "callback count",
        )
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            v.require(type(row[key]) is int and row[key] >= 0, "memory type")


def prefix_metrics(directory):
    import torch
    from safetensors.torch import load_file

    values = {}
    for phase in ("unmerged", "merged"):
        saved = load_file(str(directory / f"logits-{phase}.safetensors"), device="cpu")
        v.require(set(saved) == {"prefix_logits"}, "logits tensor keys")
        values[phase] = tensor = saved["prefix_logits"]
        v.require(
            tensor.dtype == torch.float32
            and tuple(tensor.shape) == (4, 262144)
            and bool(torch.isfinite(tensor).all()),
            "logits tensor",
        )
    delta = values["merged"] - values["unmerged"]
    return {
        "max_absolute_difference": delta.abs().max().item(),
        "root_mean_square_difference": delta.square().mean().sqrt().item(),
        "argmax_equal": (values["merged"].argmax(-1) == values["unmerged"].argmax(-1)).tolist(),
    }


def lifecycle(protocol, complete, raw):
    limit, elapsed = protocol["max_runtime_seconds"], complete["wall_seconds"]
    v.require(
        type(limit) is int
        and 1 <= limit <= 600
        and v.finite(elapsed, positive=True)
        and elapsed <= limit,
        "finite lifecycle",
    )
    v.require(
        v.accumulated(r["latency_ms"] for r in raw) <= elapsed * 1000,
        "observations exceed lifecycle",
    )


def replay_tokens(tokenizer_path, model_path, grammar_path, examples, raw, runner, helper):
    model = v.pinned(model_path, MODEL_SHA)
    names = {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    v.require({p.name for p in tokenizer_path.iterdir()} == names, "tokenizer inventory")
    v.require(
        all(v.digest(tokenizer_path / n) == model["files"][n] for n in names), "tokenizer bytes"
    )
    v.require(v.digest(grammar_path) == runner.GRAMMAR_SHA256, "grammar bytes")
    v.require(
        importlib.metadata.version("transformers") == "5.13.0"
        and importlib.metadata.version("xgrammar") == "0.2.6",
        "token replay versions",
    )
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import transformers
    import xgrammar as xgr

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=262144, stop_token_ids=[1, 106, 50]
    )
    compiled = xgr.GrammarCompiler(info).compile_grammar(
        xgr.Grammar.from_ebnf(grammar_path.read_text())
    )
    prompts = {r["id"]: helper.chat_tokens(tokenizer, r["messages"], True) for r in examples}
    for row in raw:
        ids = prompts[row["id"]]
        v.require(
            len(ids) == row["input_tokens"]
            and hashlib.sha256(json.dumps(ids).encode()).hexdigest() == row["prompt_sha256"],
            "prompt replay",
        )
        v.require(
            tokenizer.decode(row["token_ids"], skip_special_tokens=True) == row["prediction"],
            "decode replay",
        )
        matcher = xgr.GrammarMatcher(compiled)
        v.require(
            all(matcher.accept_token(t) for t in row["token_ids"]) and matcher.is_terminated(),
            "grammar replay",
        )


def verify(args):
    complete = v.pinned(args.directory / "completed.json", args.completed_sha256)
    files = {
        "protocol.json",
        "dispatch.jsonl",
        "raw.jsonl",
        "summary.json",
        "loaded-merged.json",
        "loaded-unmerged.json",
        "logits-merged.safetensors",
        "logits-unmerged.safetensors",
    }
    v.require(set(complete["files"]) == files, "profile artifact set")
    v.inventory(args.directory, complete["files"], exclude=("completed.json",))
    wrapper = load(HERE / "train.py", profile.TRAINER_SHA, "profile_training")
    runner = load(HERE / "run.py", profile.RUNNER_SHA, "profile_runtime")
    helper = wrapper.load_module(
        HERE.parent / "scene-adapter-v2-2026-09-08/train.py", wrapper.V2_SHA256, "profile_parent"
    )
    training = v.pinned(args.training_proof, args.training_proof_sha256)
    v.require(
        training["verified"] is True
        and training["verifier_sha256"] == VERIFIER_SHA
        and training["quality_accepted"] is False,
        "training proof",
    )
    proof = training["training"]
    v.require(
        proof["recipe"] == "qv"
        and type(proof["steps"]) is int
        and proof["steps"] == 4800
        and proof["initial_adapter_sha256"] == wrapper.INITIAL_SHA256,
        "full QV training",
    )
    _, checkpoint, trained = runner.checkpoint(
        args.training_run,
        proof["completed_sha256"],
        profile.TRAINER_SHA,
        MODEL_SHA,
        helper,
        wrapper,
    )
    v.require(
        checkpoint["step"] == proof["selected_step"]
        and checkpoint["files"]["adapter_model.safetensors"] == proof["selected_adapter_sha256"],
        "selected adapter",
    )
    data = v.pinned(HERE / "manifest.json", DATA_SHA)
    rows = wrapper.read_rows(
        HERE, data, "train-messages.jsonl", 4800, helper, ["system", "user", "assistant"]
    )
    examples = [{"id": rows[i]["id"], "messages": rows[i]["messages"][:2]} for i in profile.INDICES]
    phases = (
        ["unmerged", "merged"] if args.phase_order == "unmerged-first" else ["merged", "unmerged"]
    )
    schedules = {phase: profile.plan(examples, phase) for phase in phases}
    protocol = v.read(args.directory / "protocol.json")
    expected = {
        "source_sha256": PROFILE_SHA,
        "trainer_sha256": profile.TRAINER_SHA,
        "paired_runner_sha256": profile.RUNNER_SHA,
        "generation_helper_sha256": runner.V3_SHA256,
        "completed_sha256": proof["completed_sha256"],
        "selected_step": proof["selected_step"],
        "model_manifest_sha256": MODEL_SHA,
        "adapter_files": checkpoint["files"],
        "data_manifest_sha256": DATA_SHA,
        "input_indices": list(profile.INDICES),
        "examples": examples,
        "phase_order": phases,
        "schedule": schedules,
        "gold_read": False,
        "latency_scope": (
            "resident tokenization to decoded text; excludes load, merge, "
            "logits probes, warmups and grammar verification"
        ),
        "order_limitation": (
            "phases are sequential; a reverse-order process is a separate confirmation"
        ),
    }
    v.require(all(v.same(protocol[k], val) for k, val in expected.items()), "profile protocol")
    v.require(
        protocol["versions"] == trained["versions"]
        and protocol["cuda"] == "12.8"
        and "L4" in protocol["device"]
        and v.same(protocol["runtime_sources"], trained["runtime_sources"]),
        "runtime binding",
    )
    plan = [row for phase in phases for row in schedules[phase]]
    v.require(v.same(v.rows(args.directory / "dispatch.jsonl"), plan), "dispatch journal")
    raw = v.rows(args.directory / "raw.jsonl")
    check_records(raw, plan)
    lifecycle(protocol, complete, raw)
    for phase in phases:
        loaded = v.read(args.directory / f"loaded-{phase}.json")
        hashes = loaded["merged_weight_hashes"]
        v.require(
            loaded["all_frozen"] is True
            and type(loaded["adapter_tensors_verified"]) is int
            and loaded["adapter_tensors_verified"] == 100
            and isinstance(hashes, list)
            and len(hashes) == (50 if phase == "merged" else 0)
            and all(isinstance(h, str) and v.SHA.fullmatch(h) for h in hashes),
            "loaded proof",
        )
    result = profile.comparison(raw)
    result["prefix_logits"] = prefix_metrics(args.directory)
    v.require(
        v.same(v.read(args.directory / "summary.json"), result)
        and all(v.same(complete[k], val) for k, val in result.items()),
        "derived metrics",
    )
    replay_tokens(args.tokenizer, args.model_manifest, args.grammar, examples, raw, runner, helper)
    return {
        "schema_version": 1,
        "verified": True,
        "verifier_sha256": v.digest(Path(__file__)),
        "profile_sha256": PROFILE_SHA,
        "completed_sha256": args.completed_sha256,
        "training_proof_sha256": args.training_proof_sha256,
        "phase_order": phases,
        "tokenizer_and_grammar_replayed": True,
        "observations": 32,
        **result,
        "merge_weight_scope": (
            "Pinned runtime checked actual matrix arithmetic; this offline audit binds its hashes, "
            "not a fresh full-base merge."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "directory",
        "training-run",
        "training-proof",
        "tokenizer",
        "model-manifest",
        "grammar",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("completed-sha256", "training-proof-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--phase-order", choices=("unmerged-first", "merged-first"), required=True)
    args = parser.parse_args()
    result = verify(args)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
