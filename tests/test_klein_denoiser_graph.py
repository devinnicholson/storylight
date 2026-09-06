import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.klein_denoiser_graph import DenoiserGraphAdapter  # noqa: E402


@pytest.fixture
def graph_runtime(monkeypatch):
    state = {"capture": None, "stance": None, "calls": 0, "waits": 0, "fail": False}

    class Device:
        type = "cuda"

        def __str__(self):
            return "cuda:0"

    device = Device()

    class Tensor:
        requires_grad, layout = False, "strided"

        def __init__(self, shape, value=1, dtype="bfloat16"):
            self.shape, self.value, self.dtype, self.device = shape, value, dtype, device

        def stride(self):
            return tuple(range(len(self.shape), 0, -1))

        def clone(self):
            return Tensor(self.shape, self.value, self.dtype)

        def copy_(self, source):
            self.value = source.value

        def record_stream(self, stream):
            assert isinstance(stream, Stream)

    class Stream:
        cuda_stream = 0

        def __init__(self, **kwargs):
            pass

        def wait_stream(self, other):
            state["waits"] += 1

        def synchronize(self):
            pass

    class Graph:
        def replay(self):
            self.output.value = sum(t.value for t in self.inputs)

    @contextmanager
    def capture(graph, *, stream):
        assert state["stance"] == "fail_on_recompile"
        state["capture"] = graph
        try:
            yield
        finally:
            state["capture"] = None

    @contextmanager
    def stance(value):
        state["stance"] = value
        try:
            yield
        finally:
            state["stance"] = None

    @contextmanager
    def stream_context(stream):
        yield

    class Transformer:
        training = False
        transformer_blocks = [SimpleNamespace(_compiled_call_impl=object()) for _ in range(5)]
        single_transformer_blocks = [
            SimpleNamespace(_compiled_call_impl=object()) for _ in range(20)
        ]

        def forward(self, **kwargs):
            state["calls"] += 1
            if state["capture"] is not None and state["fail"]:
                raise RuntimeError("synthetic capture failure")
            inputs = [v for v in kwargs.values() if isinstance(v, Tensor)]
            output = Tensor(kwargs["hidden_states"].shape, sum(t.value for t in inputs))
            if state["capture"] is not None:
                state["capture"].inputs, state["capture"].output = inputs, output
            return (output,)

    torch = SimpleNamespace(
        __version__="2.8.0+cu128",
        int64="int64",
        bfloat16="bfloat16",
        strided="strided",
        is_tensor=lambda t: isinstance(t, Tensor),
        is_inference_mode_enabled=lambda: True,
        is_grad_enabled=lambda: False,
        compiler=SimpleNamespace(set_stance=stance),
        cuda=SimpleNamespace(
            Stream=Stream,
            current_stream=lambda device: Stream(),
            stream=stream_context,
            CUDAGraph=Graph,
            graph=capture,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(__version__="0.39.0", Flux2Transformer2DModel=Transformer),
    )

    def kwargs(bucket):
        return dict(
            hidden_states=Tensor((1, 2304, 128)),
            encoder_hidden_states=Tensor((1, bucket, 7680)),
            timestep=Tensor((1,)),
            img_ids=Tensor((1, 2304, 4), dtype="int64"),
            txt_ids=Tensor((1, bucket, 4), dtype="int64"),
            guidance=None,
            joint_attention_kwargs=None,
            return_dict=False,
        )

    return SimpleNamespace(transformer=Transformer(), state=state, kwargs=kwargs)


def test_graph_replay_updates_all_inputs_and_owns_outputs(graph_runtime):
    fixture = graph_runtime
    transformer = fixture.transformer
    original = transformer.forward
    adapter = DenoiserGraphAdapter(transformer)
    with adapter.capture():
        first = transformer.forward(**fixture.kwargs(128))[0]
        transformer.forward(**fixture.kwargs(256))
    assert first.value == 5 and fixture.state["calls"] == 8
    changed = fixture.kwargs(128)
    for index, name in enumerate(
        ("hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids")
    ):
        changed[name].value = index + 10
    with adapter.replay():
        result = transformer.forward(**changed)[0]
        assert result.value == 60 and first.value == 5
        changed["timestep"].value = 100
        later = transformer.forward(**changed)[0]
        assert later.value == 148 and result.value == 60
    assert fixture.state["calls"] == 8 and fixture.state["waits"] == 6
    assert transformer.forward == original and "forward" not in vars(transformer)
    report = adapter.report()
    assert report["graph_count"] == 2 and report["capture_warmup_calls"] == 6
    assert report["captured_forward_calls"] == 2
    assert [row["replays"] for row in report["graphs"]] == [3, 1]


def test_graph_failures_restore_forward_and_do_not_recapture(graph_runtime):
    fixture = graph_runtime
    transformer = fixture.transformer
    original = transformer.forward
    adapter = DenoiserGraphAdapter(transformer)
    with pytest.raises(ValueError), adapter.replay():
        pass
    with adapter.capture():
        transformer.forward(**fixture.kwargs(128))
        transformer.forward(**fixture.kwargs(256))
    before = fixture.state["calls"]
    with pytest.raises(ValueError), adapter.replay():
        transformer.forward(**fixture.kwargs(512))
    assert fixture.state["calls"] == before and transformer.forward == original
    with pytest.raises(ValueError), adapter.capture():
        pass
    failed = DenoiserGraphAdapter(transformer)
    fixture.state["fail"] = True
    with pytest.raises(RuntimeError, match="capture failure"), failed.capture():
        transformer.forward(**fixture.kwargs(128))
    assert transformer.forward == original and fixture.state["stance"] is None
    assert failed.report()["graph_count"] == 0
    with pytest.raises(ValueError), failed.capture():
        pass
