import gzip
import hashlib
import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import klein_denoiser_probe as probe  # noqa: E402


def trace():
    events = []

    def event(name, category, start, duration, correlation=None, device=False):
        return {
            "ph": "X",
            "name": name,
            "cat": category,
            "ts": start,
            "dur": duration,
            "pid": 2 if device else 1,
            "tid": 2 if device else 1,
            "args": {} if correlation is None else {"correlation": correlation},
        }

    for index in range(4):
        start = index * 1000
        events.append(event(f"{probe.RANGE_PREFIX}{index}", "user_annotation", start, 100))
        events.append(event(probe.SDPA_OPERATORS[0], "cpu_op", start + 1, 10))
        for offset, (name, category) in enumerate(
            (("cudaLaunchKernel", "cuda_runtime"), ("cuLaunchKernelEx", "cuda_driver"))
        ):
            correlation = index * 2 + offset
            events.append(event(name, category, start + 20 + offset * 10, 5, correlation))
            events.append(
                event(
                    "flash_private_kernel",
                    "kernel",
                    start + 200 + offset * 10,
                    30,
                    correlation,
                    device=True,
                )
            )
    events.append(event("unrelated_vae", "kernel", 5000, 100, 999, device=True))
    return json.dumps({"traceEvents": events}).encode()


def test_trace_correlates_asynchronous_driver_launches_without_adding_overlaps():
    raw = trace()
    report = probe.analyze_trace(raw)
    assert report["launch_count"] == 8 and report["launch_attribution_complete"] is True
    assert report["sdpa_operator_counts"][probe.SDPA_OPERATORS[0]] == 4
    for row in report["forwards"]:
        assert row["kernel_count"] == 2
        assert row["kernel_seconds_sum"] == pytest.approx(60e-6)
        assert row["kernel_union_seconds"] == pytest.approx(40e-6)
        assert row["kernel_gap_seconds"] == 0
        assert row["launch_api_union_seconds"] == pytest.approx(10e-6)
        assert row["post_launch_api_delay_median_seconds"] == pytest.approx(175e-6)
    assert "private_kernel" not in json.dumps(report)
    missing = json.loads(raw)
    missing["traceEvents"] = [e for e in missing["traceEvents"] if e["cat"] != "kernel"]
    partial = probe.analyze_trace(json.dumps(missing).encode())
    assert partial["launch_attribution_complete"] is False
    assert partial["unmatched_launch_count"] == 8
    assert all(row["kernel_count"] == 0 for row in partial["forwards"])
    for malformed in (b'{"traceEvents":[],"traceEvents":[]}', b'{"traceEvents":[],"x":NaN}'):
        with pytest.raises(ValueError):
            probe.analyze_trace(malformed)


def test_fixed_comparison_retains_profiles_on_capture_failure_and_restores_hooks(
    monkeypatch, capsys
):
    state = SimpleNamespace(mode=None, count=0, profiling=False, failure=None, ranges=[])

    class Transformer:
        compiled_block = object()

        def forward(self):
            if state.failure == "forward" and state.profiling:
                raise RuntimeError("private source")

    transformer = Transformer()
    original_block = transformer.compiled_block

    class Adapter:
        def __init__(self, target):
            assert target is transformer
            self.replays = {128: 0, 256: 0}

        @contextmanager
        def capture(self):
            if state.failure == "capture":
                raise RuntimeError("private source")
            with self.replay():
                yield

        @contextmanager
        def replay(self):
            state.mode = self
            try:
                yield
            finally:
                state.mode = None

        def report(self):
            buckets = [bucket for bucket, count in self.replays.items() if count]
            return {
                "graph_count": len(buckets),
                "capture_warmup_calls": 3 * len(buckets),
                "captured_forward_calls": len(buckets),
                "graphs": [
                    {
                        "bucket": bucket,
                        "replays": self.replays[bucket],
                        "signature_sha256": "a" * 64,
                        "capture_seconds": 0.1,
                    }
                    for bucket in buckets
                ],
            }

    class Profile:
        def __enter__(self):
            assert state.count in (2, 3)
            state.profiling = True
            return self

        def __exit__(self, *args):
            state.profiling = False

        def export_chrome_trace(self, path):
            assert not state.profiling
            Path(path).write_bytes(trace())

    def profile(**kwargs):
        assert kwargs == {
            "activities": ["cpu", "cuda"],
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
        }
        return Profile()

    def record_function(name):
        state.ranges.append(name)
        return nullcontext()

    monkeypatch.setitem(
        sys.modules, "klein_denoiser_graph", SimpleNamespace(DenoiserGraphAdapter=Adapter)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            compiler=SimpleNamespace(is_compiling=lambda: False),
            profiler=SimpleNamespace(
                profile=profile,
                record_function=record_function,
                ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
            ),
        ),
    )
    master, depth = b"master", b"depth"
    cases = [
        {
            "prompt": "private source",
            "seed": bucket,
            "expected_bucket": bucket,
            "master_sha256": hashlib.sha256(master).hexdigest(),
            "depth_sha256": hashlib.sha256(depth).hexdigest(),
        }
        for bucket in (128, 256)
    ]

    def render(prompt, seed):
        state.count += 1
        for _ in range(4):
            transformer.forward()
        if state.mode:
            state.mode.replays[seed] += 4
        return (
            {
                "seed": seed,
                "sequence_bucket": seed,
                "master_sha256": cases[0]["master_sha256"],
                "depth_sha256": cases[0]["depth_sha256"],
            },
            b"corrupt" if state.failure == "jpeg" else master,
            depth,
        )

    runtime = SimpleNamespace(pipe=SimpleNamespace(transformer=transformer), render=render)
    result = probe.run_comparison(runtime, cases)
    assert result["status"] == "complete" and result["failure_stage"] is None
    assert [(r["phase"], r["case_index"], r["variant"]) for r in result["records"]] == list(
        probe.SCHEDULE
    )
    assert len(result["records"]) == 14
    assert [r["replays"] for r in result["graph_report"]["graphs"]] == [12, 12]
    for profile_row in result["profiles"]:
        raw = gzip.decompress(profile_row["trace_gzip"])
        assert probe.analyze_trace(raw) == profile_row["analysis"]
    assert state.ranges == [f"{probe.RANGE_PREFIX}{i}" for _ in range(2) for i in range(4)]
    for failure, count, retained, stage in (
        ("capture", 4, 2, "render"),
        ("forward", 3, 0, "render"),
        ("jpeg", 1, 0, "verification"),
    ):
        state.count, state.failure = 0, failure
        failed = probe.run_comparison(runtime, cases)
        assert failed["status"] == "failed" and failed["failure_stage"] == stage
        assert state.count == count and len(failed["profiles"]) == retained
        assert "forward" not in vars(transformer)
        assert transformer.compiled_block is original_block
        assert state.mode is None and state.profiling is False
    assert "private source" not in capsys.readouterr().out
