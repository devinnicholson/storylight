"""Prove cache transfer at both sides of the sliding boundary before model use."""

import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, StaticCache

spec = importlib.util.spec_from_file_location(
    "cache_bridge", Path(__file__).with_name("cache-bridge.py")
)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


@pytest.mark.parametrize("length", [3, 7, 8, 11])
@torch.inference_mode()
def test_transfer_preserves_next_token_history_across_sliding_boundary(length):
    config = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=8,
        intermediate_size=16,
    )
    config.layer_types = ["full_attention", "sliding_attention"]
    config.sliding_window = 8
    source = DynamicCache(config=config)
    target = StaticCache(config=config, max_cache_len=32)
    prefix = torch.arange(8 * length, dtype=torch.float32).reshape(1, 2, length, 4)
    for index in range(2):
        source.update(prefix + index, prefix + index + 1000, index)
    bridge.transfer(source, target, torch)
    for step in range(3):
        token = torch.full((1, 2, 1, 4), 9000.0 + step)
        for index in range(2):
            dk, dv = source.update(token, token + 100, index)
            sk, sv = target.update(token, token + 100, index)
            # Static storage is padded before full capacity; only live keys are compared.
            assert torch.equal(dk, sk[:, :, : dk.shape[-2]])
            assert torch.equal(dv, sv[:, :, : dv.shape[-2]])
            assert int(source.layers[index].get_seq_length()) == int(
                target.layers[index].get_seq_length()
            )


@torch.inference_mode()
def test_generate_resume_uses_one_uncached_token_and_matches_baseline():
    from transformers import LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=8,
        intermediate_size=16,
        eos_token_id=None,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config).eval()
    ids = torch.tensor([[3, 4, 5]])
    baseline = model.generate(ids, max_new_tokens=4, do_sample=False, disable_compile=True)
    dynamic = DynamicCache(config=config)
    prepared = model.prepare_inputs_for_generation(
        ids,
        attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(3).unsqueeze(0),
        past_key_values=dynamic,
        use_cache=True,
        logits_to_keep=1,
        is_first_iteration=True,
    )
    logits = model(**prepared).logits[:, -1, :].float()
    first = logits.argmax(dim=-1, keepdim=True)
    static = bridge.transfer(dynamic, StaticCache(config=config, max_cache_len=16), torch)
    combined = torch.cat([ids, first], dim=-1)
    lengths = []

    def record(module, args, kwargs):
        lengths.append(kwargs["input_ids"].shape[-1])

    handle = model.register_forward_pre_hook(record, with_kwargs=True)
    resumed = model.generate(
        combined,
        attention_mask=torch.ones_like(combined),
        past_key_values=static,
        max_new_tokens=3,
        do_sample=False,
        disable_compile=True,
    )
    handle.remove()
    assert lengths == [1, 1, 1]
    assert torch.equal(baseline, resumed)
