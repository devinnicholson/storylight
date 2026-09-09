"""CPU-only cache reset, mode isolation, schedule and parity checks."""

import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cache_profile", Path(__file__).with_name("cache-profile.py"))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class Tests(unittest.TestCase):
    def test_schedule_bounds_long_short_reset_and_token_parity(self):
        examples = [dict(id=str(i), input_ids=list(range(10 + i))) for i in range(8)]
        records = []
        for mode in profile.MODES:
            rows = profile.plan(examples, mode)
            assert [r['id'] for r in rows[:3]] == ['7', '0', '7']
            assert sum(r['warmup'] for r in rows) == 2
            records.extend({**r, 'token_ids': [int(r['id']), 1],
                            'latency_ms': 100 if mode == 'dynamic' else 80} for r in rows)
        result = profile.comparison(records)
        assert result['all_calls_token_identical']
        assert not result['quality_accepted'] and not result['production_promotion']
        assert result['modes']['static-compile']['identical_baseline_token_pairs'] == 16
        self.assertAlmostEqual(
            result['modes']['static-compile']['matched_median_fraction_reduction'], 0.2)
        changed = copy.deepcopy(records)
        changed[-1]['token_ids'] = [50]
        result = profile.comparison(changed)
        assert not result['all_calls_token_identical']
        assert result['modes']['static-compile']['identical_baseline_token_pairs'] == 15
        for invalid in (records[:-1], [*records[:-1], records[-2]]):
            with self.assertRaises(ValueError):
                profile.comparison(invalid)

    def test_modes_never_auto_compile_baseline_or_static_control(self):
        marker = object()
        assert profile.generation_options('dynamic', None) == {
            'cache_implementation': 'dynamic', 'disable_compile': True}
        assert profile.generation_options('static-eager', marker) == {
            'past_key_values': marker, 'disable_compile': True}
        assert profile.generation_options('static-compile', marker) == {
            'past_key_values': marker, 'disable_compile': False}
        with self.assertRaisesRegex(ValueError, 'cache_mode'):
            profile.generation_options('dynamic', marker)

    def test_actual_static_cache_reset_preserves_addresses_and_clears_sliding_state(self):
        import torch
        from transformers import Gemma4TextConfig, StaticCache

        config = Gemma4TextConfig(num_hidden_layers=2, num_kv_shared_layers=0,
                                  layer_types=['sliding_attention', 'full_attention'],
                                  sliding_window=4)
        cache = StaticCache(config=config, max_cache_len=profile.CAPACITY)
        assert profile.reset_cache(cache, torch) == []
        with torch.inference_mode():
            values = torch.ones((1, 1, 7, 2))
            for index in range(2):
                cache.update(values, values, index)
        pointers = [(r.keys.data_ptr(), r.values.data_ptr()) for r in cache.layers]
        shapes = profile.reset_cache(cache, torch)
        assert [r['keys'][-2] for r in shapes] == [4, 1024]
        assert [(r.keys.data_ptr(), r.values.data_ptr()) for r in cache.layers] == pointers
        assert all(int(r.get_seq_length()) == 0 for r in cache.layers)
        # A shorter following request must begin at zero, including sliding Python counters.
        with torch.inference_mode():
            for index in range(2):
                cache.update(values[:, :, :2], values[:, :, :2], index)
        assert all(int(r.get_seq_length()) == 2 for r in cache.layers)
        profile.reset_cache(cache, torch)


if __name__ == '__main__':
    unittest.main()
