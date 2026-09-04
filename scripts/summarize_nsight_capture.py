"""Summarize the synthetic planner capture without publishing raw process/environment traces."""

import argparse
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path


def summarize(directory: Path) -> dict:
    measurements = json.loads((directory / "measurements.json").read_text())
    with sqlite3.connect(f"file:{directory / 'planner.sqlite'}?mode=ro", uri=True) as db:
        graph_count, graph_ns = db.execute(
            "SELECT count(*), sum(end-start) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
        ).fetchone()
        apis = db.execute(
            "SELECT s.value, count(*), sum(r.end-r.start)/1e6 "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
            "GROUP BY s.value ORDER BY 3 DESC LIMIT 5"
        ).fetchall()
        ranges = db.execute(
            "SELECT coalesce(e.text,s.value), count(*), sum(e.end-e.start)/1e6 "
            "FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id WHERE e.end>e.start "
            "GROUP BY 1 ORDER BY 3 DESC LIMIT 5"
        ).fetchall()
    medians = {
        key: statistics.median(row["http_ms"] for row in measurements[key]["requests"])
        for key in ("baseline", "profiled")
    }
    return {
        "boundary": measurements["boundary"],
        "trace_includes": "model startup, first warmup, three synthetic requests, teardown",
        "kiosk_during_capture": "stopped for memory headroom; restored afterward",
        "nsight_version": "2026.3.1",
        "trace_mode": "cuda,nvtx,osrt; graph-level CUDA graphs; no CPU sampling",
        "trace_sha256": hashlib.sha256((directory / "planner.nsys-rep").read_bytes()).hexdigest(),
        "http_median_ms": medians,
        "observed_profile_overhead_percent": (medians["profiled"] / medians["baseline"] - 1) * 100,
        "responses_identical": all(
            a["response"]["choices"] == b["response"]["choices"]
            for a, b in zip(
                measurements["baseline"]["requests"],
                measurements["profiled"]["requests"],
                strict=True,
            )
        ),
        "cuda_graph_executions": graph_count,
        "summed_graph_ms": graph_ns / 1e6,
        "cuda_api_top5": [
            {"name": name, "calls": count, "summed_ms": ms} for name, count, ms in apis
        ],
        "nvtx_top5": [
            {"name": name, "ranges": count, "summed_ms": ms} for name, count, ms in ranges
        ],
        "interpretation": [
            "CUDA Graphs already execute; enabling graphs is not a new optimization.",
            "cudaStreamSynchronize includes waiting for GPU work, not necessarily removable waste.",
            "Graph-level mode omits inner kernels; do not claim full kernel attribution.",
            "Startup NVTX ranges and nested spans overlap; never add them into end-to-end latency.",
            "Three short synthetic requests are not a production speed or accuracy benchmark.",
        ],
        "measurements": measurements,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("http_median_ms", "cuda_graph_executions", "responses_identical")
            }
        )
    )
