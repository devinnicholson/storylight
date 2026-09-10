"""Offline token, compiler-cache and process-order audit of a seeded54-call arm."""

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
SOURCE_SHA = "b308c7e236ef81b30915fe416639a6b8032e7a5faa2b94a422754082d62a4939"
BRIDGE_SHA = "e42f6ca4a24d60a756c29122ea323b151ae9d89b4631a6fbba63e64adfd31a31"
TRAIN_SHA = "96a566c1d53f1e6c02607b188a09caece3f40307a2e3f24e33f78d80b2baa1e9"
GRAMMAR_SHA = "622bc2bf36a2182c21a7e8a889f292f9b10a91dd149ee0aa913dd70ad20404df"
MODES = ("dynamic", "bridge-eager", "bridge-compile")
WRAPPER_SHA = "4a3c44494e45821817b56fe5bb846d520e30a0bd0e28797160316710aaa37e00"
BASELINE_SHA = "3dd88880ffe4c5a34346b2cf81e2bbcf3c4b59425e19f34f2c5135fa5bd50a2d"
CPU_SHA = "f871a13638172227673d571b380bb1918010d23b72c377a7286ad1931fb2a92c"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_inventory(receipt):
    check(receipt["root"] == "/tmp/storylight-v5-seeded-compiler-cache-01", "compiler root")
    files = receipt["files"]
    check(isinstance(files, dict), "compiler inventory type")
    for name, value in files.items():
        check(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and Path(name).parts[0] in ("inductor", "triton"),
            "compiler file path",
        )
        check(
            set(value) == {"bytes", "sha256"}
            and type(value["bytes"]) is int
            and value["bytes"] >= 0
            and len(value["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in value["sha256"]),
            "compiler file metadata",
        )
    check(
        type(receipt["file_count"]) is int
        and receipt["file_count"] == len(files)
        and receipt["total_bytes"] == sum(r["bytes"] for r in files.values()),
        "compiler counts",
    )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    check(
        hashlib.sha256(encoded).hexdigest() == receipt["inventory_sha256"],
        "compiler inventory hash",
    )


def check_startup(startup, arm):
    seed = "1" if arm == "reuse-1" else "0"
    check(
        startup["arm"] == arm
        and startup["python_hash_seed"] == seed
        and startup["hash_randomization"] == (0 if seed == "0" else 1),
        "startup seed",
    )
    check(
        type(startup["pid"]) is int
        and startup["pid"] > 0
        and isinstance(startup["python_version"], str)
        and startup["python_version"],
        "process identity",
    )
    check(startup["cpu_proof_sha256"] == CPU_SHA, "startup CPUproof pin")
    check(
        startup["helper_modules"]
        == dict(
            compiled="seeded_compiled_profile",
            merge="seeded_merge_helper",
            trainer="seeded_training_helper",
            runner="seeded_runner",
            parent="seeded_parent",
        ),
        "stable helper module names",
    )
    check(
        set(startup["layer_type_hashes"]) == {"full_attention", "sliding_attention"}
        and all(type(v) is int for v in startup["layer_type_hashes"].values()),
        "runtime string hashes",
    )


def schedule(examples, ids, mode):
    order = sorted(range(8), key=lambda i: (len(ids[i]), i))
    measured = [order[-1], *order[:-1]]
    sequence = [(-1, order[-1]), (-1, order[0])]
    sequence += [(0, i) for i in measured] + [(1, i) for i in reversed(measured)]
    return [
        dict(
            dispatch_ordinal=n,
            mode=mode,
            id=examples[i]["id"],
            case_index=i,
            repetition=rep,
            warmup=rep == -1,
        )
        for n, (rep, i) in enumerate(sequence)
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
        "compiler-cache-before.json",
        "compiler-cache-after.json",
        "model-order.json",
    }
    if args.arm != "cold-0":
        names.add("prior-comparison.json")
    names |= {f"{prefix}-{mode}.json" for prefix in ("mode", "trace") for mode in MODES}
    check(set(complete["files"]) == names, "completion inventory")
    check({p.name for p in directory.iterdir()} == names | {"completed.json"}, "disk inventory")
    for name, expected in complete["files"].items():
        check(digest(directory / name) == expected, f"file hash: {name}")
    protocol = read(directory / "protocol.json")
    check(protocol["source_sha256"] == SOURCE_SHA, "runner pin")
    check(protocol["bridge_sha256"] == BRIDGE_SHA, "bridge pin")
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
        protocol["calls"] == 54
        and protocol["static_capacity"] == 1024
        and protocol["test_data_read"] is False,
        "scope",
    )
    check(
        protocol["compile_request"] == dict(mode="reduce-overhead", fullgraph=False, dynamic=None),
        "compile configuration",
    )
    check(digest(HERE / "run.py") == WRAPPER_SHA, "local seeded wrapper source")
    check(
        digest(ROOT / "cache-bridge-compiled.py") == SOURCE_SHA
        and digest(ROOT / "cache-bridge-reuse.py") == BRIDGE_SHA,
        "local sources",
    )
    check(
        protocol["confirmation_wrapper_sha256"] == WRAPPER_SHA
        and protocol["prior_completed_sha256"] == args.prior_completed_sha256,
        "seeded provenance",
    )
    startup = protocol["startup"]
    seed = "1" if args.arm == "reuse-1" else "0"
    check_startup(startup, args.arm)
    cpu_path = HERE.parent / "scene-compile-order-2026-09-09/cpu-proof.json"
    check(digest(cpu_path) == CPU_SHA, "CPUproof source")
    cpu = next(r for r in read(cpu_path)["processes"] if r["python_hash_seed"] == seed)
    order = read(directory / "model-order.json")
    check(
        order["actual_unique_layer_types"]
        == order["expected_order"]
        == cpu["actual_unique_layer_types"]
        and order["matches_cpu_order"] is True,
        "actual GPU model layer order",
    )
    check(
        order["rotary_buffer_registration_order"] == cpu["rotary_buffer_registration_order"]
        and order["rotary_frequency_shapes"] == cpu["rotary_frequency_shapes"]
        and order["class_module"] == cpu["model_class_module"],
        "actual rotary order",
    )
    baseline = ROOT / "results/gpu/cache-bridge-compiled-01"
    check(digest(baseline / "completed.json") == BASELINE_SHA, "baseline completion")
    baseline_complete = read(baseline / "completed.json")
    check(
        digest(baseline / "model.json") == baseline_complete["files"]["model.json"],
        "baseline model pin",
    )
    model = read(directory / "model.json")
    check(
        model["merged_weight_hashes"] == read(baseline / "model.json")["merged_weight_hashes"],
        "merged model consistency",
    )
    for name in ("model_load_ms", "verified_merge_ms"):
        check(
            type(model[name]) in (int, float) and math.isfinite(model[name]) and model[name] > 0,
            "model timing",
        )
    training_path = ROOT / "results/full-training-verified.json"
    check(
        digest(training_path) == "f99ba54ea656e4e3556550ea90701c25c709095ffd005d3149df111e32058040",
        "training proof pin",
    )
    training_proof = read(training_path)
    check(
        training_proof["verified"] is True
        and protocol["completed_sha256"] == training_proof["training"]["completed_sha256"]
        and protocol["adapter_files"]["adapter_model.safetensors"]
        == training_proof["training"]["selected_adapter_sha256"],
        "verified training binding",
    )
    before = read(directory / "compiler-cache-before.json")
    after = read(directory / "compiler-cache-after.json")
    check_inventory(before)
    check_inventory(after)
    check(
        (before["file_count"] == 0) == (args.arm == "cold-0") and after["file_count"] > 0,
        "cold/reused compiler state",
    )
    prior_complete, prior_rows = None, None
    if args.arm == "cold-0":
        check(args.prior_run is None and args.prior_completed_sha256 is None, "cold prior")
    else:
        check(
            args.prior_run is not None and args.prior_completed_sha256 is not None,
            "reuse requires pinned prior",
        )
        prior = args.prior_run
        check(digest(prior / "completed.json") == args.prior_completed_sha256, "prior completion")
        prior_complete = read(prior / "completed.json")
        for name in ("raw.jsonl", "protocol.json", "compiler-cache-after.json", "model.json"):
            check(digest(prior / name) == prior_complete["files"][name], "prior artifact pin")
        prior_protocol = read(prior / "protocol.json")
        expected_prior_arm = "cold-0" if args.arm == "reuse-0" else "reuse-0"
        check_startup(prior_protocol["startup"], expected_prior_arm)
        check(
            prior_protocol["confirmation_wrapper_sha256"] == WRAPPER_SHA
            and prior_protocol["source_sha256"] == SOURCE_SHA
            and prior_protocol["completed_sha256"] == protocol["completed_sha256"]
            and prior_protocol["model_manifest_sha256"] == protocol["model_manifest_sha256"],
            "prior contract continuity",
        )
        check(prior_protocol["startup"]["pid"] != startup["pid"], "fresh process PID")
        for key in ("helper_modules", "python_version", "hash_info"):
            check(startup[key] == prior_protocol["startup"][key], "stable process configuration")
        if args.arm == "reuse-0":
            check(
                startup["layer_type_hashes"] == prior_protocol["startup"]["layer_type_hashes"],
                "same seed hash stability",
            )
        else:
            check(
                startup["layer_type_hashes"] != prior_protocol["startup"]["layer_type_hashes"],
                "changed seed hashes",
            )
        check(
            before["files"] == read(prior / "compiler-cache-after.json")["files"],
            "compiler cache continuity",
        )
        check(
            model["merged_weight_hashes"] == read(prior / "model.json")["merged_weight_hashes"],
            "prior merged model consistency",
        )
        prior_rows = [json.loads(s) for s in (prior / "raw.jsonl").read_text().splitlines()]
        check(len(prior_rows) == 54, "prior count")
    cache_root = "/tmp/storylight-v5-seeded-compiler-cache-01"
    check(
        protocol["compiler_cache_preparation"]
        == dict(
            mode="new local compiler directories"
            if args.arm == "cold-0"
            else "existing local compiler artifacts",
            directories={
                "TORCHINDUCTOR_CACHE_DIR": cache_root + "/inductor",
                "TRITON_CACHE_DIR": cache_root + "/triton",
            },
            prior_completed_sha256=args.prior_completed_sha256,
            file_count=before["file_count"],
            total_bytes=before["total_bytes"],
            inventory_sha256=before["inventory_sha256"],
            limitation=(
                "Fresh Python process; no claim of cold driver, GPU, OS or filesystem caches."
            ),
        ),
        "seeded cache provenance",
    )
    check(digest(ROOT / "train-messages.jsonl") == TRAIN_SHA, "training input pin")
    training = [json.loads(s) for s in (ROOT / "train-messages.jsonl").read_text().splitlines()]
    examples = [
        dict(id=training[i]["id"], messages=training[i]["messages"][:2])
        for i in (0, 1, 2, 800, 1600, 2400, 3000, 3600)
    ]
    check(protocol["examples"] == examples, "fixed probes")
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
    check(protocol["schedule"] == schedules, "protocol schedule")
    plan = [r for mode in MODES for r in schedules[mode]]
    dispatch = [json.loads(s) for s in (directory / "dispatch.jsonl").read_text().splitlines()]
    check(dispatch == plan, "dispatch order")
    raw = [json.loads(s) for s in (directory / "raw.jsonl").read_text().splitlines()]
    check(len(raw) == 54, "raw count")
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
    expected_prior = None
    if prior_rows is not None:
        prior_comparisons = []
        keys = ("dispatch_ordinal", "mode", "id", "case_index", "repetition", "warmup")
        for old, new in zip(prior_rows, raw, strict=True):
            check(all(old[k] == new[k] for k in keys), "prior dispatch consistency")
            prior_comparisons.append(
                {
                    **{key: new[key] for key in keys},
                    "tokens_identical": old["token_ids"] == new["token_ids"],
                }
            )
        expected_prior = dict(
            prior_completed_sha256=args.prior_completed_sha256,
            prior_raw_sha256=prior_complete["files"]["raw.jsonl"],
            rows=prior_comparisons,
            all_54_tokens_identical=all(r["tokens_identical"] for r in prior_comparisons),
        )
        check(
            read(directory / "prior-comparison.json") == expected_prior
            and complete["all_54_prior_tokens_identical"]
            == expected_prior["all_54_tokens_identical"],
            "all54 prior comparison replay",
        )
    comparisons = []
    for i in range(8):
        for rep in (0, 1):
            selected = {
                mode: next(
                    r
                    for r in raw
                    if r["case_index"] == i and r["repetition"] == rep and r["mode"] == mode
                )
                for mode in MODES
            }
            comparisons.append(
                dict(
                    case_index=i,
                    repetition=rep,
                    id=examples[i]["id"],
                    identical_tokens=all(
                        r["token_ids"] == selected["dynamic"]["token_ids"]
                        for r in selected.values()
                    ),
                    latencies_ms={mode: r["latency_ms"] for mode, r in selected.items()},
                )
            )
    all_identical = all(
        len({tuple(r["token_ids"]) for r in raw if r["id"] == e["id"]}) == 1 for e in examples
    )
    summary = dict(
        calls=54,
        comparisons=comparisons,
        all_tokens_identical=all(r["identical_tokens"] for r in comparisons),
        all_calls_token_identical=all_identical,
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
        trace_checks[mode] = dict(
            unique_graphs=unique,
            cuda_graph_launch_observed=graph,
            compile_counters=proof["compile_counters"],
        )
    check(0 < complete["wall_seconds"] <= protocol["max_runtime_seconds"] <= 900, "runtime bound")
    check(
        math.fsum(r["latency_ms"] for r in raw)
        + model["model_load_ms"]
        + model["verified_merge_ms"]
        <= complete["wall_seconds"] * 1000 + 1,
        "timing scope",
    )
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
        observations=54,
        arm=args.arm,
        startup=startup,
        model_order=order,
        prior_completed_sha256=args.prior_completed_sha256,
        all_54_prior_tokens_identical=(
            expected_prior["all_54_tokens_identical"] if expected_prior else None
        ),
        wall_seconds=complete["wall_seconds"],
        model_load_ms=model["model_load_ms"],
        verified_merge_ms=model["verified_merge_ms"],
        compiler_inventory_counts={"before": before["file_count"], "after": after["file_count"]},
        compiler_inventory_bytes={"before": before["total_bytes"], "after": after["total_bytes"]},
        compiler_inventory_sha256={
            "before": before["inventory_sha256"],
            "after": after["inventory_sha256"],
        },
        completed_sha256=args.completed_sha256,
        review_script_sha256=digest(Path(__file__)),
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
            "All54 actual tokenizer/grammar/EOS/decode replays",
            "Transfer lengths, first tokens, shapes and stable pointer receipts",
            "Summary, matched-token timing and actual trace observation recomputation",
        ],
        quality_accepted=False,
        limitations=[
            "Eight training probes, fixed mode order; not general accuracy or end-to-end timing",
            "Cache zeroing is checked by pinned runtime, raw KV tensors are not retained",
            "Seed and cache state controlled; model/GPU already warm from preceding modes",
            "Not first-request latency or totalservice startup; no cold driver/OS claim",
            "Compiler bytes are represented by runtime hash receipts, not independently downloaded",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--arm", choices=("cold-0", "reuse-0", "reuse-1"), required=True)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--prior-completed-sha256")
    parser.add_argument("--completed-sha256", required=True)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("/private/tmp/storylight-v3-tokenizer")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
