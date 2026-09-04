"""Summarize kernels inside measured HTTP requests, excluding model load and warmup."""

import argparse
import hashlib
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Evidence already exists")
    measurements = json.loads((args.capture / "measurements.json").read_text())
    connection = sqlite3.connect(f"file:{args.capture / 'planner.sqlite'}?mode=ro", uri=True)
    epoch = connection.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[
        0
    ]
    rows = []
    totals = defaultdict(lambda: [0, 0])
    for request in measurements["profiled"]["requests"]:
        start, end = request["start_epoch_ns"] - epoch, request["end_epoch_ns"] - epoch
        kernels = connection.execute(
            "SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName=s.id WHERE k.end>? AND k.start<? ORDER BY k.start",
            (start, end),
        ).fetchall()
        busy, previous_end = 0, start
        for a, b, name in kernels:
            a, b = max(start, a), min(end, b)
            busy += max(0, b - max(a, previous_end))
            previous_end = max(previous_end, b)
            totals[name][0] += 1
            totals[name][1] += b - a
        rows.append(
            {
                "http_ms": request["http_ms"],
                "kernels": len(kernels),
                "gpu_kernel_union_ms": busy / 1e6,
                "gpu_kernel_fraction_of_http": busy / (end - start),
            }
        )
    connection.close()
    baseline = statistics.mean(r["http_ms"] for r in measurements["baseline"]["requests"])
    profiled = statistics.mean(r["http_ms"] for r in measurements["profiled"]["requests"])
    report = {
        "boundary": measurements["boundary"],
        "graph_detail": measurements["graph_detail"],
        "alignment": (
            "HTTP wall-clock epoch aligned to Nsight session epoch; kernel overlaps clipped"
        ),
        "excludes": ["model load", "warmup", "idle outside measured HTTP requests"],
        "trace_sha256": hashlib.sha256(
            (args.capture / "planner.nsys-rep").read_bytes()
        ).hexdigest(),
        "baseline_mean_http_ms": baseline,
        "profiled_mean_http_ms": profiled,
        "profiling_overhead_percent": 100 * (profiled / baseline - 1),
        "requests": rows,
        "top_kernels": [
            {"name": k, "calls": n, "summed_ms": t / 1e6}
            for k, (n, t) in sorted(totals.items(), key=lambda p: p[1][1], reverse=True)[:12]
        ],
        "interpretation_limit": "Kernel occupancy is profiled, not uninstrumented GPU utilization. "
        "Three short synthetic prompts; not production planning latency.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
