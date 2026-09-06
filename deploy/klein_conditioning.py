"""Opt-in, exclusive-use conditioning experiment for the pinned Klein pipeline.

The original Diffusers helper owns tokenization and hidden-state extraction. Only
its encoder argument changes: Qwen3Model returns the same hidden states without
the causal-LM wrapper's discarded vocabulary projection. Weights stay resident.
"""

from contextlib import contextmanager


@contextmanager
def backbone_conditioning(pipe):
    """Temporarily skip unused logits; callers must serialize use of this pipeline."""
    import diffusers
    import transformers

    if diffusers.__version__ != "0.39.0" or transformers.__version__ != "4.57.1":
        raise ValueError("conditioning experiment requires pinned framework versions")
    from diffusers import Flux2KleinPipeline
    from transformers import Qwen3ForCausalLM, Qwen3Model

    name = "_get_qwen3_prompt_embeds"
    if type(pipe) is not Flux2KleinPipeline or name in vars(pipe):
        raise ValueError("conditioning experiment requires an unmodified Klein pipeline")
    encoder = pipe.text_encoder
    if type(encoder) is not Qwen3ForCausalLM or type(encoder.model) is not Qwen3Model:
        raise ValueError("conditioning experiment requires the original Qwen3 encoder")
    backbone = encoder.model
    if (
        encoder.training
        or backbone.training
        or encoder.dtype != backbone.dtype
        or encoder.device != backbone.device
        or hasattr(encoder, "_hf_hook")
        or hasattr(backbone, "_hf_hook")
    ):
        raise ValueError("conditioning experiment requires matching resident inference models")
    original = pipe._get_qwen3_prompt_embeds

    def prompt_embeds(text_encoder, *args, **kwargs):
        if text_encoder is not encoder:
            raise ValueError("conditioning encoder changed during comparison")
        return original(backbone, *args, **kwargs)

    object.__setattr__(pipe, name, prompt_embeds)
    try:
        yield pipe
    finally:
        object.__delattr__(pipe, name)
