"""Reproduce case6 attention-mask hashes and check their mathematical meaning on CPU."""

import hashlib
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache, StaticCache
from transformers.masking_utils import (
    create_causal_mask,
    create_masks_for_generate,
    create_sliding_window_causal_mask,
)

p = Path("experiments/scene-adapter-v5-2026-09-09/results/gpu/cache-prefill-01")
assert hashlib.sha256((p / "completed.json").read_bytes()).hexdigest() == (
    "f84fccbcfe8f44fdead3a64e1a691d35f90070cfb9930a67e4cdfef90345e586"
)
runtime_sources = json.loads((p / "runtime-sources.json").read_text())
for name, obj in (("masking", create_causal_mask), ("cache", StaticCache)):
    assert (
        hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
        == (runtime_sources[name]["sha256"])
    )
c = AutoConfig.from_pretrained("/private/tmp/bookforge-v3-tokenizer", local_files_only=True)
c._attn_implementation = "sdpa"
c.get_text_config()._attn_implementation = "sdpa"
n = 641
ids = torch.arange(n).unsqueeze(0)
mask = torch.ones(1, n, dtype=torch.long)
embed = torch.empty((1, n, 0), dtype=torch.bfloat16)
static = StaticCache(config=c, max_cache_len=1024)
dynamic = DynamicCache(config=c.get_text_config())
sk = create_masks_for_generate(
    config=c,
    inputs_embeds=embed,
    attention_mask=mask,
    past_key_values=static,
    position_ids=ids,
    is_first_iteration=True,
)
dk = {
    name: f(
        config=c.get_text_config(),
        inputs_embeds=embed,
        attention_mask=mask,
        past_key_values=dynamic,
        position_ids=ids,
    )
    for name, f in [
        ("full_attention", create_causal_mask),
        ("sliding_attention", create_sliding_window_causal_mask),
    ]
}
actual = next(
    r
    for r in map(json.loads, (p / "raw.jsonl").read_text().splitlines())
    if r["case_index"] == 6 and r["mode"] == "static"
)["prepared_inputs"]["attention_mask"]
for k, v in sk.items():
    h = hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert h == actual[k]["sha256"]
full = torch.arange(1024).unsqueeze(0) <= torch.arange(n).unsqueeze(1)
sliding = (torch.arange(n).unsqueeze(0) <= torch.arange(n).unsqueeze(1)) & (
    torch.arange(n).unsqueeze(0) > torch.arange(n).unsqueeze(1) - 512
)
assert torch.equal(sk["full_attention"][0, 0], full)
assert torch.equal(sk["sliding_attention"][0, 0], sliding)
assert dk["full_attention"] is None
assert torch.equal(dk["sliding_attention"], sk["sliding_attention"])
print(
    json.dumps(
        {
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime_sources_sha256": hashlib.sha256(
                (p / "runtime-sources.json").read_bytes()
            ).hexdigest(),
            "cpu_masking_and_cache_sources_match_gpu": True,
            "raw_sha256": hashlib.sha256((p / "raw.jsonl").read_bytes()).hexdigest(),
            "static_gpu_mask_hashes": {k: v["sha256"] for k, v in actual.items()},
            "static_gpu_mask_hashes_reproduced": True,
            "static_full_is_correct_causal_and_padding": True,
            "sliding_dynamic_static_identical": True,
            "dynamic_full_mask_none_uses_is_causal": True,
            "cause_established": False,
        }
    )
)
