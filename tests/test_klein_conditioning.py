import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.klein_conditioning import backbone_conditioning  # noqa: E402


@pytest.fixture
def models(monkeypatch):
    calls, helpers = [], []
    weights = object()
    states = tuple(object() for _ in range(29))

    class Qwen3Model:
        training = False
        dtype, device = "bfloat16", "cuda:0"

        def __init__(self):
            self.weights = weights

        def __call__(self, **kwargs):
            calls.append(("backbone", kwargs))
            return SimpleNamespace(hidden_states=states)

    class Qwen3ForCausalLM:
        training = False
        dtype, device = "bfloat16", "cuda:0"

        def __init__(self):
            self.model, self.lm_head = Qwen3Model(), object()

        def __call__(self, **kwargs):
            result = self.model(**kwargs)
            calls.append(("lm_head", self.lm_head))
            return result

    class Flux2KleinPipeline:
        def __init__(self):
            self.text_encoder = Qwen3ForCausalLM()
            self.config = object()

        @staticmethod
        def _get_qwen3_prompt_embeds(text_encoder, *args, **kwargs):
            helpers.append((text_encoder, args, kwargs))
            output = text_encoder(
                input_ids=kwargs["input_ids"],
                attention_mask=kwargs["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            return tuple(output.hidden_states[k] for k in kwargs["hidden_states_layers"])

    diffusers = SimpleNamespace(__version__="0.39.0", Flux2KleinPipeline=Flux2KleinPipeline)
    transformers = SimpleNamespace(
        __version__="4.57.1", Qwen3ForCausalLM=Qwen3ForCausalLM, Qwen3Model=Qwen3Model
    )
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return SimpleNamespace(
        pipe=Flux2KleinPipeline(),
        calls=calls,
        helpers=helpers,
        states=states,
        diffusers=diffusers,
        transformers=transformers,
    )


def test_backbone_preserves_helper_arguments_hidden_states_and_restores(models):
    pipe = models.pipe
    encoder, config = pipe.text_encoder, pipe.config
    backbone, head, weights = encoder.model, encoder.lm_head, encoder.model.weights
    original = pipe._get_qwen3_prompt_embeds
    kwargs = dict(
        input_ids=object(),
        attention_mask=object(),
        hidden_states_layers=(9, 18, 27),
        prompt="synthetic watercolor scene",
        max_sequence_length=128,
    )
    baseline = original(encoder, "tokenizer", **kwargs)
    baseline_call = models.calls[0]
    with backbone_conditioning(pipe) as active:
        assert active is pipe
        candidate = active._get_qwen3_prompt_embeds(encoder, "tokenizer", **kwargs)
        assert candidate == baseline == tuple(models.states[k] for k in (9, 18, 27))
        assert models.calls[2] == baseline_call and len(models.calls) == 3
        assert models.helpers[1] == (backbone, ("tokenizer",), kwargs)
        with pytest.raises(ValueError), backbone_conditioning(pipe):
            pass
    assert pipe._get_qwen3_prompt_embeds is original
    assert "_get_qwen3_prompt_embeds" not in vars(pipe)
    assert (pipe.text_encoder, pipe.config, encoder.model, encoder.lm_head, backbone.weights) == (
        encoder,
        config,
        backbone,
        head,
        weights,
    )
    with pytest.raises(RuntimeError, match="render failed"), backbone_conditioning(pipe):
        raise RuntimeError("render failed")
    assert pipe._get_qwen3_prompt_embeds is original
    assert original(encoder, "tokenizer", **kwargs) == baseline
    assert models.calls[-1] == ("lm_head", head)


def test_backbone_rejects_unqualified_versions_models_and_offloading(models):
    pipe = models.pipe
    original = pipe._get_qwen3_prompt_embeds
    for target, field, value in (
        (models.diffusers, "__version__", "0.40.0"),
        (models.transformers, "__version__", "5.0.0"),
        (pipe.text_encoder, "training", True),
        (pipe.text_encoder.model, "dtype", "float16"),
        (pipe.text_encoder.model, "device", "cpu"),
        (pipe, "text_encoder", object()),
    ):
        previous = getattr(target, field)
        setattr(target, field, value)
        with pytest.raises(ValueError), backbone_conditioning(pipe):
            pass
        setattr(target, field, previous)
        assert pipe._get_qwen3_prompt_embeds is original
    pipe.text_encoder._hf_hook = object()
    with pytest.raises(ValueError), backbone_conditioning(pipe):
        pass
    assert not models.calls
