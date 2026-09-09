"""Offline audit of the fixed 54-call cache diagnostic; no model execution."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGE_VERIFIER_SHA = "ad78e49e4084f01f3c805dbae1590a3126496a35041dfe802b03f714120c2b26"
CACHE_SHA = "1318515bfa5f390fa52ab4d0ee0821770a24a25998d6c80b532ea6d9eb453cc2"


def load(path, sha, name):
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError("unreviewed source")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load(HERE / "verify-serving-profile.py", MERGE_VERIFIER_SHA, "cache_merge_verifier")
cache = load(HERE / "cache-profile.py", CACHE_SHA, "cache_profile_source")
v = merge.v


def check_records(raw, plan):
    v.require(len(raw) == len(plan) == 54, "incomplete cache run")
    fields = {
        "prompt_sha256",
        "input_tokens",
        "token_ids",
        "generated_tokens",
        "prediction",
        "output",
        "terminated_with_eos",
        "grammar_accepts_complete_tokens",
        "latency_ms",
        "generation_ms",
        "cache_reset_ms",
        "reset_verified",
        "initialized_cache_shapes",
        "grammar_host_callback_ms",
        "grammar_callback_calls",
        "profiler_active",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
    for row, dispatch in zip(raw, plan, strict=True):
        v.require(
            set(row) == fields | set(dispatch)
            and all(v.same(row[k], val) for k, val in dispatch.items()),
            "raw schedule",
        )
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
            row["terminated_with_eos"] is True
            and row["grammar_accepts_complete_tokens"] is True
            and isinstance(row["prediction"], str)
            and len(row["prediction"].encode()) <= 4096
            and row["output"] == row["prediction"],
            "complete output",
        )
        v.require(
            type(row["input_tokens"]) is int
            and 0 < row["input_tokens"] <= 768
            and type(row["generated_tokens"]) is int
            and row["generated_tokens"] == len(tokens)
            and type(row["grammar_callback_calls"]) is int
            and row["grammar_callback_calls"] == len(tokens),
            "token counts",
        )
        v.require(
            isinstance(row["prompt_sha256"], str) and v.SHA.fullmatch(row["prompt_sha256"]),
            "prompt fingerprint",
        )
        for name in ("latency_ms", "generation_ms", "cache_reset_ms", "grammar_host_callback_ms"):
            v.require(
                v.finite(row[name], positive=name in ("latency_ms", "generation_ms")), "timing"
            )
        v.require(
            row["grammar_host_callback_ms"] <= row["generation_ms"]
            and row["generation_ms"] + row["cache_reset_ms"] <= row["latency_ms"],
            "timing scope",
        )
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            v.require(type(row[name]) is int and row[name] >= 0, "memory type")
        static = row["mode"] != "dynamic"
        v.require(
            row["reset_verified"] is static
            and row["profiler_active"] is (row["dispatch_ordinal"] == 1),
            "reset/profiler flag",
        )
        shapes = row["initialized_cache_shapes"]
        v.require(isinstance(shapes, list), "cache shapes")
        if not static or row["dispatch_ordinal"] == 0:
            v.require(shapes == [], "initial cache state")
        else:
            v.require(bool(shapes), "missing initialized cache")
            for layer in shapes:
                v.require(
                    set(layer) == {"type", "keys", "values"}
                    and layer["type"] in ("StaticLayer", "StaticSlidingWindowLayer")
                    and v.same(layer["keys"], layer["values"])
                    and len(layer["keys"]) == 4
                    and all(type(n) is int and n > 0 for n in layer["keys"])
                    and layer["keys"][0] == 1
                    and layer["keys"][2] <= cache.CAPACITY,
                    "cache tensor shape",
                )


def mode_proof(directory, mode, raw):
    proof = v.read(directory / f"mode-{mode}.json")
    counters = proof["compile_counters"]
    v.require(
        isinstance(counters, dict) and all(isinstance(g, dict) for g in counters.values()),
        "compiler counters",
    )
    unique = counters.get("stats", {}).get("unique_graphs", 0)
    v.require(
        type(unique) is int
        and unique >= 0
        and proof["compilation_observed"] is (unique > 0)
        and (mode != "static-compile" or unique > 0),
        "compilation observation",
    )
    trace = proof["trace"]
    v.require(
        set(trace)
        == {"trace_sha256", "cuda_graph_launch_events", "cuda_launch_kernel_events", "events"}
        and trace["trace_sha256"] == v.digest(directory / f"trace-{mode}.json"),
        "trace binding",
    )
    for name in ("events", "cuda_graph_launch_events", "cuda_launch_kernel_events"):
        v.require(type(trace[name]) is int and trace[name] >= 0, "trace count")
    v.require(
        trace["cuda_graph_launch_events"] <= trace["events"]
        and trace["cuda_launch_kernel_events"] <= trace["events"]
        and proof["cuda_graph_launch_observed"] is (trace["cuda_graph_launch_events"] > 0),
        "CUDA graph observation",
    )
    # Kineto event lists and Chrome trace include different metadata/events; do not
    # claim identical event counts. Bind bytes and check reported launch presence.
    events = v.read(directory / f"trace-{mode}.json")["traceEvents"]
    names = [e.get("name", "") for e in events]
    v.require(
        any("cudaGraphLaunch" in n for n in names) == proof["cuda_graph_launch_observed"],
        "CUDA graph trace evidence",
    )
    preparation = [r["latency_ms"] for r in raw if r["mode"] == mode and r["warmup"]]
    v.require(v.same(preparation, proof["preparation_latencies_ms"]), "preparation timing")
    for name in ("model_load_ms", "verified_merge_ms"):
        v.require(v.finite(proof[name], positive=True), "load/merge timing")
    hashes = proof["merged_weight_hashes"]
    v.require(
        isinstance(hashes, list)
        and len(hashes) == 50
        and all(isinstance(h, str) and v.SHA.fullmatch(h) for h in hashes),
        "merged matrix hashes",
    )
    return proof


def verify(args):
    complete = v.pinned(args.directory / "completed.json", args.completed_sha256)
    names = {"protocol.json", "dispatch.jsonl", "raw.jsonl", "summary.json", "grammar.json"}
    names |= {f"{prefix}-{mode}.json" for mode in cache.MODES for prefix in ("mode", "trace")}
    v.require(set(complete["files"]) == names, "cache inventory")
    v.inventory(args.directory, complete["files"], exclude=("completed.json",))
    wrapper = merge.load(HERE / "train.py", merge.profile.TRAINER_SHA, "cache_training")
    runner = merge.load(HERE / "run.py", merge.profile.RUNNER_SHA, "cache_runtime")
    helper = wrapper.load_module(
        HERE.parent / "scene-adapter-v2-2026-09-08/train.py", wrapper.V2_SHA256, "cache_parent"
    )
    training = v.pinned(args.training_proof, args.training_proof_sha256)
    v.require(
        training["verified"] is True
        and training["verifier_sha256"] == merge.VERIFIER_SHA
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
    _, selected, trained = runner.checkpoint(
        args.training_run,
        proof["completed_sha256"],
        merge.profile.TRAINER_SHA,
        merge.MODEL_SHA,
        helper,
        wrapper,
    )
    v.require(
        selected["step"] == proof["selected_step"]
        and selected["files"]["adapter_model.safetensors"] == proof["selected_adapter_sha256"],
        "adapter selection",
    )
    manifest = v.pinned(HERE / "manifest.json", merge.DATA_SHA)
    rows = wrapper.read_rows(
        HERE, manifest, "train-messages.jsonl", 4800, helper, ["system", "user", "assistant"]
    )
    examples = [{"id": rows[i]["id"], "messages": rows[i]["messages"][:2]} for i in cache.INDICES]
    raw = v.rows(args.directory / "raw.jsonl")
    v.require(len(raw) == 54, "incomplete raw")
    counts = {r["id"]: r["input_tokens"] for r in raw}
    v.require(
        set(counts) == {r["id"] for r in examples}
        and all(type(n) is int and 0 < n <= 768 for n in counts.values()),
        "probe IDs/lengths",
    )
    # Input lengths are independently replayed below before this schedule is accepted.
    scheduled = [{**r, "input_ids": [0] * counts[r["id"]]} for r in examples]
    schedules = {mode: cache.plan(scheduled, mode) for mode in cache.MODES}
    plan = [row for mode in cache.MODES for row in schedules[mode]]
    protocol = v.read(args.directory / "protocol.json")
    expected = dict(
        source_sha256=CACHE_SHA,
        merge_helper_sha256=merge.PROFILE_SHA,
        trainer_sha256=merge.profile.TRAINER_SHA,
        runner_sha256=merge.profile.RUNNER_SHA,
        generation_helper_sha256=runner.V3_SHA256,
        grammar_sha256=runner.GRAMMAR_SHA256,
        completed_sha256=proof["completed_sha256"],
        selected_step=proof["selected_step"],
        model_manifest_sha256=merge.MODEL_SHA,
        adapter_files=selected["files"],
        data_manifest_sha256=merge.DATA_SHA,
        input_indices=list(cache.INDICES),
        examples=examples,
        schedule=schedules,
        calls=54,
        mode_order=list(cache.MODES),
        static_capacity=1024,
        compile_request=dict(mode="reduce-overhead", fullgraph=False, dynamic=None),
        latency_scope="tokenize/reset/generate/decode; excludes load/merge/grammar compile",
        order_limitation="sequential modes/shared host cache; reverse-order screen separate",
        test_data_read=False,
        production_export=False,
    )
    v.require(all(v.same(protocol[k], val) for k, val in expected.items()), "cache protocol")
    v.require(
        v.same(protocol["versions"], trained["versions"])
        and protocol["cuda"] == "12.8"
        and "L4" in protocol["device"]
        and v.same(protocol["runtime_sources"], trained["runtime_sources"]),
        "runtime binding",
    )
    v.require(v.same(v.rows(args.directory / "dispatch.jsonl"), plan), "dispatch order")
    check_records(raw, plan)
    modes = {m: mode_proof(args.directory, m, raw) for m in cache.MODES}
    v.require(
        all(
            v.same(p["merged_weight_hashes"], modes["dynamic"]["merged_weight_hashes"])
            for p in modes.values()
        ),
        "different merged models",
    )
    grammar_ms = v.read(args.directory / "grammar.json")["compile_ms"]
    v.require(v.finite(grammar_ms), "grammar compile time")
    limit, wall = protocol["max_runtime_seconds"], complete["wall_seconds"]
    work_ms = v.accumulated(r["latency_ms"] for r in raw) + grammar_ms
    work_ms += v.accumulated(p["model_load_ms"] + p["verified_merge_ms"] for p in modes.values())
    v.require(
        type(limit) is int
        and 1 <= limit <= 900
        and v.finite(wall, positive=True)
        and work_ms <= wall * 1000
        and wall <= limit,
        "finite lifecycle",
    )
    result = cache.comparison(raw)
    v.require(
        v.same(v.read(args.directory / "summary.json"), result)
        and all(v.same(complete[k], val) for k, val in result.items()),
        "derived metrics",
    )
    merge.replay_tokens(
        args.tokenizer, args.model_manifest, args.grammar, examples, raw, runner, helper
    )
    return dict(
        schema_version=1,
        verified=True,
        verifier_sha256=v.digest(Path(__file__)),
        cache_source_sha256=CACHE_SHA,
        completed_sha256=args.completed_sha256,
        training_proof_sha256=args.training_proof_sha256,
        observations=54,
        tokenizer_and_grammar_replayed=True,
        **result,
        preparation_latencies_ms={m: p["preparation_latencies_ms"] for m, p in modes.items()},
        compilation_observed={m: p["compilation_observed"] for m, p in modes.items()},
        cuda_graph_launch_observed={m: p["cuda_graph_launch_observed"] for m, p in modes.items()},
        reset_scope=(
            "Pinned runtime checked zero cache values; raw cache tensors were not retained "
            "for offline recomputation."
        ),
    )


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
    args = parser.parse_args()
    result = verify(args)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
