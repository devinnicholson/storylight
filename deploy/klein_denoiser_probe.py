"""Matched denoiser trial with separate, source-free CPU/CUDA diagnostics."""

import gzip
import hashlib
import json
import math
import os
import statistics
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

RANGE_PREFIX = "storylight.denoiser."
MAX_TRACE_BYTES = 64 * 1024 * 1024
SCHEDULE = (
    ("warmup", 0, "baseline"),
    ("warmup", 1, "baseline"),
    ("profiled", 0, "baseline"),
    ("profiled", 1, "baseline"),
    ("preparation", 0, "candidate"),
    ("preparation", 1, "candidate"),
    ("measured", 0, "baseline"),
    ("measured", 0, "candidate"),
    ("measured", 1, "candidate"),
    ("measured", 1, "baseline"),
    ("measured", 0, "candidate"),
    ("measured", 0, "baseline"),
    ("measured", 1, "baseline"),
    ("measured", 1, "candidate"),
)
SDPA_OPERATORS = (
    "aten::_scaled_dot_product_flash_attention",
    "aten::_scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_cudnn_attention",
    "aten::_scaled_dot_product_attention_math",
)


def require(condition):
    if not condition:
        raise ValueError("denoiser probe validation failed")


def union_duration(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def kernel_family(name):
    name = name.lower()
    for family, fragments in (
        ("reduction", ("triton_red", "reduce", "reduction")),
        ("elementwise", ("triton_poi", "elementwise", "vectorized")),
        ("attention", ("flash", "fmha", "attention")),
        ("matrix_multiply", ("gemm", "matmul", "cublas", "cutlass")),
    ):
        if any(fragment in name for fragment in fragments):
            return family
    return "other"


def analyze_trace(raw):
    """Correlate launches with kernels; elapsed gaps are not CPU overhead estimates."""
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_TRACE_BYTES)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("invalid trace number")

    events = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)["traceEvents"]
    require(isinstance(events, list) and len(events) <= 200_000)
    complete = []
    for event in events:
        if event.get("ph") != "X":
            continue
        require(isinstance(event.get("name"), str) and len(event["name"]) <= 16_384)
        require(
            all(
                type(event.get(key)) in (float, int) and math.isfinite(event[key])
                for key in ("ts", "dur")
            )
            and event["dur"] >= 0
        )
        complete.append(event)
    ranges = sorted(
        (
            event
            for event in complete
            if event.get("cat") == "user_annotation" and event["name"].startswith(RANGE_PREFIX)
        ),
        key=lambda event: event["ts"],
    )
    require([event["name"] for event in ranges] == [f"{RANGE_PREFIX}{index}" for index in range(4)])

    def inside(event, region):
        return (
            event.get("pid") == region.get("pid")
            and event.get("tid") == region.get("tid")
            and region["ts"] <= event["ts"]
            and event["ts"] + event["dur"] <= region["ts"] + region["dur"]
        )

    launches, duplicate_correlations = {}, set()
    operator_counts = dict.fromkeys(SDPA_OPERATORS, 0)
    for event in complete:
        containing = [i for i, region in enumerate(ranges) if inside(event, region)]
        if not containing:
            continue
        require(len(containing) == 1)
        if event["name"] in operator_counts:
            operator_counts[event["name"]] += 1
        runtime_launch = event.get("cat") == "cuda_runtime" and event["name"].startswith(
            "cudaLaunch"
        )
        driver_launch = event.get("cat") == "cuda_driver" and event["name"].startswith(
            "cuLaunchKernel"
        )
        if not (runtime_launch or driver_launch):
            continue
        correlation = event.get("args", {}).get("correlation")
        require(type(correlation) is int)
        if correlation in launches:
            duplicate_correlations.add(correlation)
        launches[correlation] = (containing[0], event)
    kernels = [[] for _ in ranges]
    identities, matched = {}, set()
    for event in complete:
        if event.get("cat") != "kernel":
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation not in launches or correlation in duplicate_correlations:
            continue
        index, launch = launches[correlation]
        matched.add(correlation)
        kernels[index].append((event, launch))
        key = hashlib.sha256(event["name"].encode()).hexdigest()
        entry = identities.setdefault(
            key,
            {
                "name_sha256": key,
                "family": kernel_family(event["name"]),
                "count": 0,
                "kernel_seconds_sum": 0.0,
            },
        )
        entry["count"] += 1
        entry["kernel_seconds_sum"] += event["dur"] / 1_000_000
    rows = []
    for index, region in enumerate(ranges):
        intervals = [(event["ts"], event["ts"] + event["dur"]) for event, _ in kernels[index]]
        launch_intervals = [
            (event["ts"], event["ts"] + event["dur"])
            for i, event in launches.values()
            if i == index
        ]
        span = (
            (max(stop for _, stop in intervals) - min(start for start, _ in intervals))
            if intervals
            else 0.0
        )
        busy = union_duration(intervals)
        delays = [
            max(0.0, event["ts"] - launch["ts"] - launch["dur"]) / 1_000_000
            for event, launch in kernels[index]
        ]
        rows.append(
            {
                "forward_index": index,
                "host_range_seconds": region["dur"] / 1_000_000,
                "kernel_count": len(intervals),
                "kernel_seconds_sum": sum(stop - start for start, stop in intervals) / 1_000_000,
                "kernel_union_seconds": busy / 1_000_000,
                "kernel_span_seconds": span / 1_000_000,
                "kernel_gap_seconds": (span - busy) / 1_000_000,
                "launch_api_union_seconds": union_duration(launch_intervals) / 1_000_000,
                "post_launch_api_delay_median_seconds": statistics.median(delays)
                if delays
                else None,
            }
        )
    return {
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "forwards": rows,
        "sdpa_operator_counts": operator_counts,
        "kernel_identities": sorted(
            identities.values(), key=lambda row: (-row["kernel_seconds_sum"], row["name_sha256"])
        ),
        "launch_count": len(launches),
        "unmatched_launch_count": len(set(launches) - matched),
        "ambiguous_correlation_count": len(duplicate_correlations),
        "launch_attribution_complete": bool(launches)
        and len(matched) == len(launches)
        and not duplicate_correlations
        and all(kernels),
        "scope": (
            "profiled transformer only; kernel gaps may include copies, waits or host delay; "
            "intervals overlap and are not additive"
        ),
    }


@contextmanager
def transformer_ranges(transformer, torch):
    missing = object()
    previous, original = vars(transformer).get("forward", missing), transformer.forward
    count = 0

    def forward(*args, **kwargs):
        nonlocal count
        require(not torch.compiler.is_compiling() and count < 4)
        name = f"{RANGE_PREFIX}{count}"
        count += 1
        with torch.profiler.record_function(name):
            return original(*args, **kwargs)

    object.__setattr__(transformer, "forward", forward)
    try:
        yield
        require(count == 4)
    finally:
        if previous is missing:
            object.__delattr__(transformer, "forward")
        else:
            object.__setattr__(transformer, "forward", previous)


def run_comparison(runtime, cases):
    import torch
    from klein_denoiser_graph import DenoiserGraphAdapter

    require(isinstance(cases, list) and len(cases) == 2)
    require([case["expected_bucket"] for case in cases] == [128, 256])
    adapter, records, profiles = None, [], []
    ordinal, stage = None, "setup"

    def event(state):
        print(
            json.dumps({"denoiser_progress": {"ordinal": ordinal, "stage": stage, "state": state}}),
            flush=True,
        )

    try:
        for ordinal, (phase, case_index, variant) in enumerate(SCHEDULE):
            case, stage = cases[case_index], "render"
            event("start")
            if phase == "profiled":
                with (
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                    ) as profiler,
                    transformer_ranges(runtime.pipe.transformer, torch),
                ):
                    metrics, master, depth = runtime.render(case["prompt"], case["seed"])
            else:
                if phase == "preparation" and adapter is None:
                    stage = "graph_setup"
                    adapter = DenoiserGraphAdapter(runtime.pipe.transformer)
                    stage = "render"
                context = (
                    adapter.capture()
                    if phase == "preparation"
                    else (adapter.replay() if variant == "candidate" else nullcontext())
                )
                with context:
                    metrics, master, depth = runtime.render(case["prompt"], case["seed"])
            stage = "verification"
            require(
                metrics["seed"] == case["seed"]
                and metrics["sequence_bucket"] == case["expected_bucket"]
            )
            for name, data in (("master", master), ("depth", depth)):
                require(isinstance(data, bytes))
                require(
                    hashlib.sha256(data).hexdigest()
                    == metrics[f"{name}_sha256"]
                    == case[f"{name}_sha256"]
                )
            records.append(
                {
                    "ordinal": ordinal,
                    "phase": phase,
                    "case_index": case_index,
                    "variant": variant,
                    "metrics": metrics,
                    "master": master,
                    "depth": depth,
                }
            )
            if phase == "profiled":
                profile_row = {
                    "case_index": case_index,
                    "trace_sha256": None,
                    "trace_gzip": None,
                    "analysis": None,
                    "analysis_status": "incomplete",
                    "analysis_error": None,
                }
                profiles.append(profile_row)
                try:
                    stage = "trace_export"
                    with tempfile.TemporaryDirectory(prefix="storylight-denoiser-") as directory:
                        path = Path(directory) / "trace.json"
                        profiler.export_chrome_trace(str(path))
                        stage = "trace_read"
                        os.chmod(path, 0o600)
                        require(path.stat().st_size <= MAX_TRACE_BYTES)
                        raw = path.read_bytes()
                        # Retain bytes before any interpretation; malformed traces are evidence.
                        stage = "trace_compress"
                        profile_row["trace_sha256"] = hashlib.sha256(raw).hexdigest()
                        profile_row["trace_gzip"] = gzip.compress(raw, mtime=0)
                    stage = "trace_analysis"
                    profile_row["analysis"] = analyze_trace(raw)
                    profile_row["analysis_status"] = "complete"
                except Exception as error:
                    error_type = type(error).__name__
                    if error_type not in {
                        "ValueError",
                        "TypeError",
                        "KeyError",
                        "IndexError",
                        "AttributeError",
                        "JSONDecodeError",
                        "OSError",
                        "RuntimeError",
                        "MemoryError",
                    }:
                        error_type = "OtherError"
                    profile_row["analysis_error"] = {"stage": stage, "type": error_type}
                    print(
                        json.dumps(
                            {
                                "denoiser_diagnostic_failure": {
                                    "ordinal": ordinal,
                                    "stage": stage,
                                    "type": error_type,
                                }
                            }
                        ),
                        flush=True,
                    )
            event("verified")
        stage = "graph_report"
        report = adapter.report()
        require(
            report["graph_count"] == 2
            and report["capture_warmup_calls"] == 6
            and report["captured_forward_calls"] == 2
        )
        require([row["bucket"] for row in report["graphs"]] == [128, 256])
        require(all(row["replays"] == 12 for row in report["graphs"]))
    except Exception:
        event("failed")
        return {
            "status": "failed",
            "failure_stage": stage,
            "records": records,
            "profiles": profiles,
            "graph_report": adapter.report()
            if adapter is not None
            else {
                "graph_count": 0,
                "capture_warmup_calls": 0,
                "captured_forward_calls": 0,
                "graphs": [],
            },
        }
    return {
        "status": "complete",
        "failure_stage": None,
        "records": records,
        "profiles": profiles,
        "graph_report": report,
    }
