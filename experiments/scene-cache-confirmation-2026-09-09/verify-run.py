"""Offline token/cache audit of the 260-call exposed-input bridge confirmation."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import xgrammar as xgr
from transformers import AutoConfig, AutoTokenizer, StaticCache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "scene-adapter-v5-2026-09-09"
SOURCE_SHA = "1c3f0ecb5111fec4f4756f4f776af2ef5977d2cf222330414e3bc6c3d8132fd3"
BRIDGE_SHA = "e42f6ca4a24d60a756c29122ea323b151ae9d89b4631a6fbba63e64adfd31a31"
TRAIN_SHA = "96a566c1d53f1e6c02607b188a09caece3f40307a2e3f24e33f78d80b2baa1e9"
GRAMMAR_SHA = "622bc2bf36a2182c21a7e8a889f292f9b10a91dd149ee0aa913dd70ad20404df"
MODES = ("dynamic", "bridge-compile")
PROTOCOL_SHA = "1a2579008b49f64d2a2d45f00a2c4ed5c63c0621d1ca6accbf0637ff4f79bd69"
INPUT_SHA = "4ffc9adc3c15fbb6e684197a05b4b8bc2281f616ea02888f38989fff3acf58fb"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def schedule(examples, ids, mode):
    check(
        len(examples) == len({r["id"] for r in examples}) == 130 and mode in MODES,
        "confirmation scope",
    )
    measured = list(range(2, 130))
    if mode == "bridge-compile":
        measured.reverse()
    return [
        dict(
            dispatch_ordinal=n,
            global_ordinal=MODES.index(mode) * 130 + n,
            mode=mode,
            id=examples[i]["id"],
            case_index=i,
            repetition=-1 if n < 2 else 0,
            warmup=n < 2,
        )
        for n, i in enumerate([0, 1, *measured])
    ]


def verify(args):
    directory = args.directory
    check(digest(directory / "completed.json") == args.completed_sha256, "completion hash")
    complete = read(directory / "completed.json")
    names = {
        "model.json",
        "protocol.json",
        "raw.jsonl",
        "dispatch.jsonl",
        "runtime-sources.json",
        "summary.json",
    }
    names |= {f"{prefix}-{mode}.json" for prefix in ("mode", "trace") for mode in MODES}
    check(set(complete["files"]) == names, "completion inventory")
    check({p.name for p in directory.iterdir()} == names | {"completed.json"}, "disk inventory")
    for name, expected in complete["files"].items():
        check(digest(directory / name) == expected, f"file hash: {name}")
    check(digest(HERE / "run.py") == SOURCE_SHA, "local runner source pin")
    check(digest(ROOT / "cache-bridge-reuse.py") == BRIDGE_SHA, "local bridge source pin")
    training_proof_path = ROOT / "results/full-training-verified.json"
    check(
        digest(training_proof_path)
        == "f99ba54ea656e4e3556550ea90701c25c709095ffd005d3149df111e32058040",
        "training verification pin",
    )
    training_proof = read(training_proof_path)
    check(training_proof["verified"] is True, "verified training required")
    selected = training_proof["training"]
    prior = ROOT / "results/gpu/cache-bridge-compiled-01"
    check(
        digest(prior / "completed.json")
        == "3dd88880ffe4c5a34346b2cf81e2bbcf3c4b59425e19f34f2c5135fa5bd50a2d",
        "prior54 completion pin",
    )
    prior_complete = read(prior / "completed.json")
    check(
        digest(prior / "model.json") == prior_complete["files"]["model.json"],
        "prior merged-model proof",
    )
    model_proof = read(directory / "model.json")
    check(
        model_proof["merged_weight_hashes"] == read(prior / "model.json")["merged_weight_hashes"],
        "merged-model equality",
    )
    check(
        all(
            type(model_proof[k]) in (int, float)
            and math.isfinite(model_proof[k])
            and model_proof[k] > 0
            for k in ("model_load_ms", "verified_merge_ms")
        ),
        "model timing",
    )
    protocol = read(directory / "protocol.json")
    check(protocol["source_sha256"] == SOURCE_SHA, "runner pin")
    check(protocol["bridge_sha256"] == BRIDGE_SHA, "bridge pin")
    check(
        protocol["completed_sha256"] == selected["completed_sha256"]
        and protocol["selected_step"] == selected["selected_step"]
        and protocol["adapter_files"]["adapter_model.safetensors"]
        == selected["selected_adapter_sha256"],
        "verified training selection",
    )
    check(
        protocol["selected_step"] == 2400
        and protocol["adapter_files"]["adapter_model.safetensors"]
        == "8acf87bdb245a4e1f92e91fc3b2495b9e5c5492a5038fd7d2432f93c6e980123",
        "adapter pin",
    )
    check(
        protocol["completed_sha256"]
        == "66326bd05dd934feae4ec1375a7b004f6efb0d267c0e0be72bfd3881f781768c",
        "training pin",
    )
    check(
        protocol["calls"] == 260
        and protocol["static_capacity"] == 1024
        and protocol["test_data_read"] is True
        and protocol["held_out"] is False
        and protocol["gold_read"] is False,
        "scope",
    )
    check(
        protocol["compile_request"] == dict(mode="reduce-overhead", fullgraph=False, dynamic=None),
        "compile configuration",
    )
    cache_root = "/tmp/bookforge-cache-confirmation-compiler-01"
    check(
        protocol["compiler_cache_preparation"]
        == {
            "TORCHINDUCTOR_CACHE_DIR": cache_root + "/inductor",
            "TRITON_CACHE_DIR": cache_root + "/triton",
        },
        "compiler cache directories",
    )
    check(digest(ROOT / "train-messages.jsonl") == TRAIN_SHA, "training input pin")
    training = [json.loads(s) for s in (ROOT / "train-messages.jsonl").read_text().splitlines()]
    check(digest(HERE / "inputs.jsonl") == INPUT_SHA, "confirmation inputs pin")
    check(digest(HERE / "protocol.json") == PROTOCOL_SHA, "confirmation protocol pin")
    examples = [json.loads(s) for s in (HERE / "inputs.jsonl").read_text().splitlines()]
    check(
        examples[:2] == [dict(id=r["id"], messages=r["messages"][:2]) for r in training[:2]],
        "training warmups",
    )
    check(protocol["examples"] == examples, "fixed confirmation inputs")
    check(
        protocol["input_sha256"] == INPUT_SHA
        and protocol["preregistered_protocol_sha256"] == PROTOCOL_SHA,
        "protocol binding",
    )
    manifest_path = ROOT / "results/gpu/model-manifest.json"
    check(
        digest(manifest_path) == "703bbb89d61aaed083846d7cb3d4ee1a68220e25de93a035b1f4b49d24062f2d",
        "model manifest pin",
    )
    manifest = read(manifest_path)
    for name in (
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        check(digest(args.tokenizer / name) == manifest["files"][name], "tokenizer file pin")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    config = AutoConfig.from_pretrained(args.tokenizer, local_files_only=True)
    cache = StaticCache(config=config, max_cache_len=1024)
    ids = [
        tokenizer.apply_chat_template(
            r["messages"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for r in examples
    ]
    check(all(len(x) + 256 <= 1024 for x in ids), "capacity")
    schedules = {mode: schedule(examples, ids, mode) for mode in MODES}
    check(
        protocol["schedule"] == schedules == read(HERE / "protocol.json")["schedule"],
        "protocol schedule",
    )
    plan = [r for mode in MODES for r in schedules[mode]]
    dispatch = [json.loads(s) for s in (directory / "dispatch.jsonl").read_text().splitlines()]
    check(dispatch == plan, "dispatch order")
    raw = [json.loads(s) for s in (directory / "raw.jsonl").read_text().splitlines()]
    check(len(raw) == 260, "raw count")
    grammar = ROOT.parent / "scene-adapter-v3-2026-09-09/grammar.ebnf"
    check(digest(grammar) == protocol["grammar_sha256"] == GRAMMAR_SHA, "grammar pin")
    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=262144, stop_token_ids=[1, 106, 50]
    )
    compiled = xgr.GrammarCompiler(info).compile_grammar(xgr.Grammar.from_ebnf(grammar.read_text()))
    pointers = {}
    for row, planned in zip(raw, plan, strict=True):
        check(all(row[k] == v and type(row[k]) is type(v) for k, v in planned.items()), "raw order")
        i, mode = row["case_index"], row["mode"]
        check(
            row["input_tokens"] == len(ids[i])
            and row["prompt_sha256"] == hashlib.sha256(json.dumps(ids[i]).encode()).hexdigest(),
            "prompt replay",
        )
        tokens = row["token_ids"]
        check(
            0 < len(tokens) <= 256 and all(type(t) is int and 0 <= t < 262144 for t in tokens),
            "token range",
        )
        check(
            tokens[-1] in (1, 106, 50) and not any(t in (1, 106, 50) for t in tokens[:-1]),
            "EOS placement",
        )
        matcher = xgr.GrammarMatcher(compiled)
        check(
            all(matcher.accept_token(t) for t in tokens) and matcher.is_terminated(),
            "grammar replay",
        )
        check(
            tokenizer.decode(tokens, skip_special_tokens=True) == row["prediction"]
            and len(row["prediction"].encode()) <= 4096,
            "decode replay",
        )
        check(
            row["terminated_with_eos"] is True and row["grammar_accepts_complete_tokens"] is True,
            "completion flags",
        )
        check(
            row["generated_tokens"] == row["grammar_callback_calls"] == len(tokens),
            "callback counts",
        )
        check(
            math.isfinite(row["latency_ms"])
            and row["latency_ms"] > 0
            and 0 <= row["grammar_host_callback_ms"] <= row["latency_ms"],
            "timing",
        )
        transfer = row["transfer_receipt"]
        if mode == "dynamic":
            check(
                transfer is None and row["prefix_ms"] is None and row["transfer_ms"] is None,
                "dynamic transfer",
            )
        else:
            check(
                transfer["source_lengths"]
                == transfer["target_lengths"]
                == [len(ids[i])] * len(cache.layers),
                "transfer lengths",
            )
            check(transfer["first_token"] == tokens[0], "transfer first token")
            shapes = transfer["target_shapes"]
            check(
                len(shapes) == len(cache.layers)
                and all(
                    s[:2] == [1, 1] and s[2] == layer.max_cache_len and s[3] in (256, 512)
                    for s, layer in zip(shapes, cache.layers, strict=True)
                ),
                "cache shapes",
            )
            addresses = transfer["target_addresses"]
            check(
                len(addresses) == len(cache.layers)
                and all(
                    len(pair) == 2 and all(type(address) is int and address > 0 for address in pair)
                    for pair in addresses
                ),
                "cache addresses",
            )
            check(mode not in pointers or pointers[mode] == addresses, "cache address stability")
            pointers[mode] = addresses
            check(
                row["prefix_ms"] > 0
                and row["transfer_ms"] > 0
                and row["prefix_ms"] + row["transfer_ms"] <= row["latency_ms"],
                "bridge timing",
            )
    comparisons = []
    for i in range(2, 130):
        selected = {
            mode: next(
                r for r in raw if r["case_index"] == i and not r["warmup"] and r["mode"] == mode
            )
            for mode in MODES
        }
        comparisons.append(
            dict(
                case_index=i,
                id=examples[i]["id"],
                identical_tokens=selected["dynamic"]["token_ids"]
                == selected["bridge-compile"]["token_ids"],
                latencies_ms={mode: r["latency_ms"] for mode, r in selected.items()},
            )
        )
    all_identical = all(
        len({tuple(r["token_ids"]) for r in raw if r["id"] == e["id"]}) == 1 for e in examples
    )
    summary = dict(
        calls=260,
        comparisons=comparisons,
        all_tokens_identical=all(r["identical_tokens"] for r in comparisons),
        all_calls_token_identical=all_identical,
        held_out=False,
        quality_accepted=False,
        production_promotion=False,
    )
    check(
        read(directory / "summary.json") == summary
        and all(complete[k] == v for k, v in summary.items()),
        "summary recomputation",
    )
    trace_checks = {}
    for mode in MODES:
        proof = read(directory / f"mode-{mode}.json")
        trace = proof["trace"]
        check(trace["trace_sha256"] == digest(directory / f"trace-{mode}.json"), "trace binding")
        names = [r.get("name", "") for r in read(directory / f"trace-{mode}.json")["traceEvents"]]
        graph = any("cudaGraphLaunch" in name for name in names)
        check(
            graph == proof["cuda_graph_launch_observed"] == (trace["cuda_graph_launch_events"] > 0),
            "CUDA graph observation",
        )
        unique = proof["compile_counters"].get("stats", {}).get("unique_graphs", 0)
        check(
            type(unique) is int and unique >= 0 and proof["compilation_observed"] is (unique > 0),
            "compile evidence",
        )
        check(mode != "bridge-compile" or unique > 0, "missing compilation")
        trace_checks[mode] = dict(unique_graphs=unique, cuda_graph_launch_observed=graph)
    check(0 < complete["wall_seconds"] <= protocol["max_runtime_seconds"] <= 1600, "runtime bound")
    work_ms = math.fsum(r["latency_ms"] for r in raw)
    work_ms += model_proof["model_load_ms"] + model_proof["verified_merge_ms"]
    check(work_ms <= complete["wall_seconds"] * 1000 + 1, "nonoverlapping work timing")
    gains = {}
    for mode in MODES[1:]:
        pairs = [
            (
                next(
                    r
                    for r in raw
                    if r["mode"] == "dynamic"
                    and r["case_index"] == a["case_index"]
                    and r["repetition"] == a["repetition"]
                ),
                a,
            )
            for a in raw
            if a["mode"] == mode and not a["warmup"]
        ]
        matching = [(a, b) for a, b in pairs if a["token_ids"] == b["token_ids"]]
        gains[mode] = dict(
            matching_pairs=len(matching),
            median_fraction_reduction=(
                statistics.median(1 - b["latency_ms"] / a["latency_ms"] for a, b in matching)
                if matching
                else None
            ),
        )
    return dict(
        reviewer="review_followups",
        verified=True,
        observations=260,
        measured_observations=256,
        measured_pairs=128,
        raw_sha256=digest(directory / "raw.jsonl"),
        input_sha256=INPUT_SHA,
        protocol_sha256=PROTOCOL_SHA,
        tokenizer_and_grammar_replayed=True,
        gold_read=False,
        held_out=False,
        completed_sha256=args.completed_sha256,
        review_script_sha256=digest(Path(__file__)),
        training_verification_sha256=digest(training_proof_path),
        all_calls_token_identical=all_identical,
        comparisons=comparisons,
        matched_gains=gains,
        trace_evidence=trace_checks,
        preparation_latencies_ms={
            mode: [r["latency_ms"] for r in raw if r["mode"] == mode and r["warmup"]]
            for mode in MODES
        },
        checks=[
            "Exact inventory/source pins/fixed inputs/schedule",
            "All260 actual tokenizer/grammar/EOS/decode replays",
            "Transfer lengths, first tokens, shapes and stable pointer receipts",
            "Summary, matched-token timing and actual trace observation recomputation",
        ],
        quality_accepted=False,
        limitations=[
            "128 exposed inputs plus two training warmups; no held-out accuracy claim",
            "Cache zeroing is checked by pinned runtime, raw KV tensors are not retained",
            "Fresh compiler-directory receipt does not prove other cache layers cold",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("/private/tmp/bookforge-v3-tokenizer")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
