"""CPU-only checks for cache reset, schedule and compiler evidence corruption."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("verify-cache-profile.py")
spec = importlib.util.spec_from_file_location("verify_cache", PATH)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def fixture():
    examples = [{"id": str(i), "input_ids": [0] * (600 + i)} for i in range(8)]
    plan = [r for mode in verifier.cache.MODES for r in verifier.cache.plan(examples, mode)]
    raw = []
    for r in plan:
        static = r["mode"] != "dynamic"
        shapes = (
            [{"type": "StaticLayer", "keys": [1, 1, 1024, 256], "values": [1, 1, 1024, 256]}]
            if static and r["dispatch_ordinal"]
            else []
        )
        raw.append(
            {
                **r,
                "prompt_sha256": "a" * 64,
                "input_tokens": 600 + r["case_index"],
                "token_ids": [123, 1],
                "generated_tokens": 2,
                "prediction": "REFUSE",
                "output": "REFUSE",
                "terminated_with_eos": True,
                "grammar_accepts_complete_tokens": True,
                "latency_ms": 100.0,
                "generation_ms": 90.0,
                "cache_reset_ms": 1.0,
                "reset_verified": static,
                "initialized_cache_shapes": shapes,
                "grammar_host_callback_ms": 1.0,
                "grammar_callback_calls": 2,
                "profiler_active": r["dispatch_ordinal"] == 1,
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 2,
            }
        )
    return plan, raw


class Tests(unittest.TestCase):
    def test_actual_schedule_reset_and_timing_corruptions(self):
        plan, raw = fixture()
        verifier.check_records(raw, plan)
        for index, change in [
            (0, {"case_index": 0}),
            (0, {"reset_verified": True}),
            (1, {"profiler_active": False}),
            (19, {"initialized_cache_shapes": []}),
            (19, {"reset_verified": False}),
            (0, {"token_ids": [1, 123]}),
            (0, {"latency_ms": float("nan")}),
            (0, {"cache_reset_ms": 11.0}),
            (0, {"generated_tokens": True}),
            (0, {"output": "repaired"}),
        ]:
            with self.subTest(index=index, change=change), self.assertRaises(ValueError):
                changed = copy.deepcopy(raw)
                changed[index].update(change)
                verifier.check_records(changed, plan)

    def test_compiler_request_does_not_prove_compilation_or_cuda_graphs(self):
        _, raw = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mode = "static-compile"
            trace = root / f"trace-{mode}.json"
            trace.write_text(json.dumps({"traceEvents": [{"name": "cudaLaunchKernel"}]}))
            proof = {
                "compile_counters": {"stats": {"unique_graphs": 1}},
                "compilation_observed": True,
                "cuda_graph_launch_observed": False,
                "trace": {
                    "trace_sha256": verifier.v.digest(trace),
                    "cuda_graph_launch_events": 0,
                    "cuda_launch_kernel_events": 1,
                    "events": 1,
                },
                "preparation_latencies_ms": [100.0, 100.0],
                "model_load_ms": 100.0,
                "verified_merge_ms": 10.0,
                "merged_weight_hashes": ["a" * 64] * 50,
            }
            path = root / f"mode-{mode}.json"
            path.write_text(json.dumps(proof))
            verified = verifier.mode_proof(root, mode, raw)
            self.assertFalse(verified["cuda_graph_launch_observed"])
            for change in (
                {"compilation_observed": False},
                {"cuda_graph_launch_observed": True},
                {"preparation_latencies_ms": [1.0, 1.0]},
                {"compile_counters": {"stats": {"unique_graphs": 0}}},
            ):
                path.write_text(json.dumps({**proof, **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    verifier.mode_proof(root, mode, raw)
            path.write_text(json.dumps(proof))
            trace.write_text(json.dumps({"traceEvents": [{"name": "cudaGraphLaunch"}]}))
            with self.assertRaisesRegex(ValueError, "trace binding"):
                verifier.mode_proof(root, mode, raw)


if __name__ == "__main__":
    unittest.main()
