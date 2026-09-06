import copy
import hashlib
import sys
from types import SimpleNamespace

import numpy as np

from deploy import klein_latent_probe as probe


def test_capture_replay_preserves_exact_initial_bytes_and_original_method(monkeypatch, capsys):
    state = SimpleNamespace(finished=True, calls=0, rng_calls=0, fail=False, altered=False)

    class Tensor:
        def __init__(self, value, dtype="bf16"):
            self.array, self.dtype = value, dtype
            self.shape = value.shape

        def stride(self):
            return tuple(x // self.array.itemsize for x in self.array.strides)

        def detach(self):
            return self

        def clone(self):
            copied = self.array.copy(order="K")
            copied = np.lib.stride_tricks.as_strided(copied, self.shape, self.array.strides)
            return Tensor(copied, self.dtype)

        def contiguous(self):
            return Tensor(np.ascontiguousarray(self.array), self.dtype)

        def permute(self, *axes):
            return Tensor(self.array.transpose(axes), self.dtype)

        def reshape(self, shape):
            return Tensor(self.array.reshape(shape), self.dtype)

        def view(self, dtype):
            assert dtype == "uint8"
            return Tensor(self.array.view(np.uint8), dtype)

        def to(self, **kwargs):
            return self

        def cpu(self):
            assert state.finished, "CPU hashing must follow runtime timer"
            return self

        def numpy(self):
            return self.array

    fake_torch = SimpleNamespace(
        __version__="2.8.0+cu128",
        bfloat16="bf16",
        int64="int64",
        uint8="uint8",
        device=lambda device: SimpleNamespace(type=device),
        isfinite=lambda tensor: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)),
        frombuffer=lambda data, dtype: Tensor(np.frombuffer(data, dtype=np.uint16)),
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    ids = Tensor(
        np.array([(0, h, w, 0) for h in range(36) for w in range(64)], dtype=np.int64).reshape(
            probe.IDS_SHAPE
        ),
        "int64",
    )

    class Pipeline:
        def prepare_latents(
            self,
            batch_size,
            num_latents_channels,
            height,
            width,
            dtype,
            device,
            generator,
            latents=None,
        ):
            if latents is None:
                state.rng_calls += 1
                latents = Tensor(np.full(probe.SHAPE, generator.initial_seed(), dtype=np.uint16))
            packed = latents.reshape((1, 128, 2304)).permute(0, 2, 1)
            return packed, ids

    pipe = Pipeline()
    master, depth = b"master", b"depth"
    cases = [
        {
            "prompt": "private synthetic source",
            "seed": seed,
            "expected_bucket": bucket,
            "master_sha256": hashlib.sha256(master).hexdigest(),
            "depth_sha256": hashlib.sha256(depth).hexdigest(),
        }
        for seed, bucket in ((10, 128), (20, 256))
    ]

    def render(prompt, seed):
        state.finished = False
        state.calls += 1
        noise, _ = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=32,
            height=576,
            width=1024,
            dtype="bf16",
            device="cuda",
            generator=SimpleNamespace(initial_seed=lambda: seed),
        )
        if state.fail:
            raise RuntimeError("private synthetic source")
        noise.array.fill(999)  # A later in-place consumer cannot alter the captured initial noise.
        state.finished = True
        image = b"different valid image" if state.altered else master
        return (
            {
                "seed": seed,
                "sequence_bucket": 128 if seed == 10 else 256,
                "master_sha256": hashlib.sha256(image).hexdigest(),
                "depth_sha256": hashlib.sha256(depth).hexdigest(),
            },
            image,
            depth,
        )

    runtime = SimpleNamespace(pipe=pipe, render=render)
    capture = probe.run_capture(runtime, cases)
    assert capture["status"] == "complete" and state.calls == state.rng_calls == 4
    assert len(capture["artifacts"]) == 2
    assert "prepare_latents" not in vars(pipe)
    for index, artifact in enumerate(capture["artifacts"]):
        assert probe.validate_artifact(artifact, cases[index], index) is artifact
        assert (
            artifact["data"]
            == np.full(probe.SHAPE, cases[index]["seed"], dtype=np.uint16).tobytes()
        )
    state.calls, state.altered = 0, True
    replay = probe.run_replay(runtime, cases, capture["artifacts"])
    assert replay["status"] == "complete" and state.calls == 10 and state.rng_calls == 4
    assert all(row["initial_noise_exact"] for row in replay["records"])
    assert not any(row["historical_images_exact"] for row in replay["records"])
    for mutation in ("data", "case_index", "latent_ids_sha256"):
        artifacts = copy.deepcopy(capture["artifacts"])
        artifacts[0][mutation] = {
            "data": b"x" * probe.NOISE_BYTES,
            "case_index": 1,
            "latent_ids_sha256": "0" * 64,
        }[mutation]
        state.calls = 0
        failed = probe.run_replay(runtime, cases, artifacts)
        assert failed["status"] == "failed" and state.calls == 0
    state.calls, state.fail = 0, True
    assert probe.run_capture(runtime, cases)["status"] == "failed"
    assert state.calls == 1 and "prepare_latents" not in vars(pipe)
    state.fail, state.finished = False, True
    assert probe.run_capture(runtime, cases)["status"] == "failed"  # Changed capture image refuses.
    assert "prepare_latents" not in vars(pipe)
    assert "private synthetic source" not in capsys.readouterr().out
