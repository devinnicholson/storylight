"""CPU-only artifact and schedule corruption checks for the merge diagnostic."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("verify-serving-profile.py")
spec = importlib.util.spec_from_file_location("verify_serving", PATH)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def fixture():
    examples = [{"id": str(i)} for i in range(4)]
    plan = [r for phase in ("unmerged", "merged") for r in verifier.profile.plan(examples, phase)]
    raw = [
        {
            **r,
            "arm": "constrained",
            "prediction": "REFUSE",
            "output": "REFUSE",
            "token_ids": [123, 1],
            "input_tokens": 600,
            "generated_tokens": 2,
            "prompt_sha256": "a" * 64,
            "terminated_with_eos": True,
            "finish_reason": "eos",
            "grammar_accepts_complete_tokens": True,
            "grammar_verification_ms": 1.0,
            "grammar_host_callback_ms": 1.0,
            "grammar_callback_calls": 2,
            "latency_ms": 100.0,
            "generation_ms": 90.0,
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 2,
        }
        for r in plan
    ]
    return plan, raw


class Tests(unittest.TestCase):
    def test_lifecycle_rejects_overrun_and_impossibly_short_receipt(self):
        _, raw = fixture()
        protocol = {"max_runtime_seconds": 600}
        verifier.lifecycle(protocol, {"wall_seconds": 4.0}, raw)
        for value in (99999.0, 1.0, True, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                verifier.lifecycle(protocol, {"wall_seconds": value}, raw)

    def test_schedule_eos_and_metric_tampering(self):
        plan, raw = fixture()
        verifier.check_records(raw, plan)
        mutations = [
            {"phase": "merged"},
            {"warmup": False},
            {"generated_tokens": True},
            {"token_ids": [1, 123]},
            {"token_ids": [1, 1]},
            {"output": "repaired"},
            {"latency_ms": float("nan")},
            {"grammar_host_callback_ms": 95.0},
            {"grammar_callback_calls": 1},
            {"grammar_accepts_complete_tokens": False},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                changed = copy.deepcopy(raw)
                changed[0].update(mutation)
                verifier.check_records(changed, plan)
        with self.assertRaisesRegex(ValueError, "incomplete profile"):
            verifier.check_records(raw[:-1], plan)

    def test_actual_logits_drift_is_reported_and_nonfinite_or_wrong_shape_rejected(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = torch.zeros((4, 262144), dtype=torch.float32)
            shifted = original + 0.25
            save_file({"prefix_logits": original}, str(root / "logits-unmerged.safetensors"))
            merged = root / "logits-merged.safetensors"
            save_file({"prefix_logits": shifted}, str(merged))
            result = verifier.prefix_metrics(root)
            self.assertEqual(result["max_absolute_difference"], 0.25)
            self.assertEqual(result["root_mean_square_difference"], 0.25)
            self.assertEqual(result["argmax_equal"], [True] * 4)
            for invalid in (shifted[:3], shifted.double(), shifted.fill_(float("nan"))):
                save_file({"prefix_logits": invalid}, str(merged))
                with self.assertRaisesRegex(ValueError, "logits tensor"):
                    verifier.prefix_metrics(root)

    def test_inventory_rejects_changed_binary_unlisted_file_and_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "logits.safetensors"
            artifact.write_bytes(b"retained tensor")
            files = {artifact.name: verifier.v.digest(artifact)}
            verifier.v.inventory(root, files)
            artifact.write_bytes(b"changed tensor")
            with self.assertRaisesRegex(ValueError, "inventory bytes"):
                verifier.v.inventory(root, files)
            artifact.write_bytes(b"retained tensor")
            (root / "failure.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unlisted or missing"):
                verifier.v.inventory(root, files)


if __name__ == "__main__":
    unittest.main()
