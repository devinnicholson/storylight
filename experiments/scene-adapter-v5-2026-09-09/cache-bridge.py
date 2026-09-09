"""Explicit HF 5.13 dynamic-to-static cache transfer for single-token decoding."""


def transfer(dynamic, static, torch):
    from transformers.cache_utils import (
        DynamicLayer,
        DynamicSlidingWindowLayer,
        StaticLayer,
        StaticSlidingWindowLayer,
    )

    if len(dynamic.layers) != len(static.layers):
        raise ValueError("cache layer mismatch")
    with torch.inference_mode():
        for source, target in zip(dynamic.layers, static.layers, strict=True):
            sliding = type(source) is DynamicSlidingWindowLayer
            expected = StaticSlidingWindowLayer if sliding else StaticLayer
            if type(target) is not expected or (not sliding and type(source) is not DynamicLayer):
                raise ValueError("unsupported cache layer")
            if not source.is_initialized or target.is_initialized:
                raise ValueError("requires populated source and fresh target")
            length = int(source.get_seq_length())
            stored = source.keys.shape[-2]
            if length < 1 or (not sliding and length > target.max_cache_len):
                raise ValueError("invalid cache length")
            if sliding and source.sliding_window != target.max_cache_len:
                raise ValueError("sliding window mismatch")
            expected_stored = min(length, source.sliding_window - 1) if sliding else length
            if stored != expected_stored or source.values.shape[-2] != stored:
                raise ValueError("unexpected source storage")
            target.lazy_initialization(source.keys, source.values)
            # The next saturated sliding update drops slot zero before appending.
            offset = 1 if sliding and length >= target.max_cache_len else 0
            target.keys[:, :, offset : offset + stored].copy_(source.keys)
            target.values[:, :, offset : offset + stored].copy_(source.values)
            target.cumulative_length.fill_(length)
            if sliding:
                target.cumulative_length_int = length
            if int(target.get_seq_length()) != length:
                raise ValueError("transferred cache length mismatch")
    return static
