"""CPU-only merge arithmetic and matched-output diagnostic checks."""

import hashlib
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

PATH = Path(__file__).with_name("serving-profile.py")
spec = importlib.util.spec_from_file_location("serving_profile", PATH)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Tests(unittest.TestCase):
    def test_resident_schedule_and_only_matching_tokens_enter_latency_subset(self):
        rows = [{"id": str(i)} for i in range(4)]
        records = []
        for phase in ("unmerged", "merged"):
            scheduled = profile.plan(rows, phase)
            assert len(scheduled) == 16
            assert sum(r["warmup"] for r in scheduled) == 4
            for row in scheduled:
                records.append(
                    {
                        **row,
                        "token_ids": [1] if phase == "unmerged" or row["id"] != "0" else [2],
                        "latency_ms": 100 if phase == "unmerged" else 80,
                    }
                )
        result = profile.comparison(records)
        assert result["identical_token_pairs"] == 9
        self.assertAlmostEqual(result["matched_token_median_fraction_reduction"], 0.2)
        assert not result["quality_accepted"] and not result["merge_equivalence_established"]
        with self.assertRaisesRegex(ValueError, "incomplete_profile"):
            profile.comparison(records[:-1])

    def test_actual_tiny_peft_bf16_merge_verifies_weights_and_bounds_output_rounding(self):
        import peft
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)

            def forward(self, x):
                return self.linear(x)

        def digest(values):
            return hashlib.sha256(
                values["weight"].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()

        torch.manual_seed(12)
        model = peft.get_peft_model(
            Tiny(),
            peft.LoraConfig(
                r=2, lora_alpha=4, lora_dropout=0, target_modules=["linear"], bias="none"
            ),
        )
        with torch.no_grad():
            model.base_model.model.linear.lora_B.default.weight.fill_(0.1)
        model.eval()
        x = torch.tensor([[1, -0.5, 0.25, 0.5]], dtype=torch.bfloat16)
        before = model(x).detach()
        merged, hashes = profile.checked_merge(
            model, torch, SimpleNamespace(require=require, parameter_hash=digest)
        )
        assert len(hashes) == 1 and not any("lora" in n for n, _ in merged.named_parameters())
        assert torch.allclose(before.float(), merged(x).float(), atol=0.02, rtol=0)

    def test_merge_rejects_nonfinite_delta_before_mutating_base(self):
        import peft
        import torch

        base = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16))
        model = peft.get_peft_model(base, peft.LoraConfig(r=1, target_modules=["0"]))
        layer = model.base_model.model[0]
        original = layer.base_layer.weight.detach().clone()
        with torch.no_grad():
            layer.lora_B.default.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite_merge"):
            profile.checked_merge(model, torch, SimpleNamespace(require=require))
        assert torch.equal(original, layer.base_layer.weight)

    def test_wrong_merge_arithmetic_is_rejected(self):
        import peft
        import torch

        base = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16))
        model = peft.get_peft_model(base, peft.LoraConfig(r=1, target_modules=["0"]))
        merge = model.merge_and_unload

        def corrupt(**kwargs):
            result = merge(**kwargs)
            with torch.no_grad():
                result[0].weight.add_(1)
            return result

        model.merge_and_unload = corrupt

        def digest(values):
            return hashlib.sha256(values["weight"].detach().cpu().contiguous()
                                  .view(torch.uint8).numpy().tobytes()).hexdigest()

        with self.assertRaisesRegex(ValueError, "merged_weight_mismatch"):
            profile.checked_merge(model, torch, SimpleNamespace(
                require=require, parameter_hash=digest))


if __name__ == "__main__":
    unittest.main()
