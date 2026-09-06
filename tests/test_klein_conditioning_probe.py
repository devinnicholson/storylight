from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import klein_conditioning_probe as probe  # noqa: E402


@pytest.fixture
def experiment(monkeypatch):
    status = SimpleNamespace(finished=False, failure=None, candidate=False, renders=0, events=0)

    class Tensor:
        dtype = "torch.bfloat16"

        def __init__(self, bucket, value, dtype="torch.bfloat16"):
            self.shape = (1, bucket, 3)
            self.value = value
            self.dtype = dtype

        def detach(self):
            assert status.finished, "tensor verification must follow timed rendering"
            return self

        def cpu(self):
            assert status.finished
            return self

        def contiguous(self):
            return self

        def clone(self):
            return Tensor(self.shape[1], self.value, self.dtype)

        def view(self, dtype):
            assert dtype == "uint8"
            return self

        def numpy(self):
            return self

        def tobytes(self):
            return str(self.value).encode()

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            status.events += 1

        def record(self):
            assert not status.finished

        def query(self):
            assert status.finished
            return status.failure != "unfinished_cuda"

        def elapsed_time(self, other):
            assert status.finished
            return 2.0

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(Event=Event),
            compiler=SimpleNamespace(is_compiling=lambda: status.failure == "compiled_boundary"),
            is_tensor=lambda value: isinstance(value, Tensor),
            uint8="uint8",
            equal=lambda a, b: a.value == b.value,
        ),
    )

    class Transformer:
        compiled_block = object()

        def forward(self):
            if status.failure == "render":
                raise RuntimeError("private synthetic prompt should never escape")

    class Pipeline:
        def __init__(self):
            self.transformer = Transformer()
            self.vae = SimpleNamespace(decode=lambda: None)

        def encode_prompt(self, bucket):
            value = bucket + int(status.candidate and status.failure == "embedding")
            ids = bucket + 1000 + int(status.candidate and status.failure == "text_ids")
            return Tensor(bucket, value), Tensor(bucket, ids, "torch.int64")

    pipe = Pipeline()
    original_decode = pipe.vae.decode
    master, depth = b"master fixture", b"depth fixture"
    cases = [
        {
            "expected_bucket": bucket,
            "prompt": "private synthetic prompt",
            "seed": bucket,
            "master_sha256": hashlib.sha256(master).hexdigest(),
            "depth_sha256": hashlib.sha256(depth).hexdigest(),
        }
        for bucket in (128, 256)
    ]

    def render(prompt, seed):
        status.finished = False
        status.renders += 1
        pipe.encode_prompt(seed)
        for _ in range(4):
            pipe.transformer.forward()
        pipe.vae.decode()
        if status.failure == "extra_decode":
            pipe.vae.decode()
        status.finished = True
        return (
            {
                "seed": seed,
                "sequence_bucket": seed,
                "master_sha256": cases[0]["master_sha256"],
                "depth_sha256": cases[0]["depth_sha256"],
                "total_seconds": 1.0,
            },
            b"wrong" if status.failure == "jpeg" else master,
            depth,
        )

    @contextmanager
    def conditioning(pipeline):
        assert pipeline is pipe
        status.candidate = True
        try:
            yield
        finally:
            status.candidate = False

    monkeypatch.setattr(probe, "backbone_conditioning", conditioning)
    runtime = SimpleNamespace(pipe=pipe, render=render)
    return runtime, cases, status, original_decode


def test_matched_schedule_verifies_actual_embeddings_and_records_asynchronous_stages(
    experiment, capsys
):
    runtime, cases, status, original_decode = experiment
    block = runtime.pipe.transformer.compiled_block
    rows = probe.run_comparison(runtime, cases)
    events = [
        json.loads(line)["conditioning_progress"] for line in capsys.readouterr().out.splitlines()
    ]
    assert events == [
        {"ordinal": ordinal, "state": state, "stage": stage}
        for ordinal in range(10)
        for state, stage in (("start", "render"), ("verified", "complete"))
    ]
    assert [(r["phase"], r["case_index"], r["variant"]) for r in rows] == list(probe.SCHEDULE)
    assert status.renders == 10 and status.events == 120
    assert [r["variant"] for r in rows[2:]] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    for row in rows:
        assert row["embedding"]["matches_baseline"] is True
        assert row["embedding"]["text_ids"]["matches_baseline"] is True
        assert row["embedding"]["text_ids"]["dtype"] == "torch.int64"
        assert row["embedding"]["shape"] == [1, cases[row["case_index"]]["expected_bucket"], 3]
        assert row["stages"]["conditioning"]["calls"] == 1
        assert row["stages"]["transformer"]["calls"] == 4
        assert row["stages"]["transformer"]["cuda_elapsed_seconds"] == 0.008
        assert row["stages"]["vae_decode"]["calls"] == 1
        assert all(stage["host_seconds"] >= 0 for stage in row["stages"].values())
        assert "private synthetic prompt" not in repr(row)
    assert runtime.pipe.transformer.compiled_block is block
    assert runtime.pipe.vae.decode is original_decode
    assert "forward" not in vars(runtime.pipe.transformer)
    assert "encode_prompt" not in vars(runtime.pipe)


def test_mismatch_failure_and_invalid_timing_restore_all_original_boundaries(experiment, capsys):
    runtime, cases, status, original_decode = experiment
    for failure in (
        "embedding",
        "text_ids",
        "jpeg",
        "extra_decode",
        "unfinished_cuda",
        "render",
        "compiled_boundary",
    ):
        status.failure = failure
        before = status.renders
        with pytest.raises(ValueError, match="^conditioning comparison failed$") as caught:
            probe.run_comparison(runtime, cases)
        assert "private synthetic" not in str(caught.value)
        output = capsys.readouterr().out
        assert "private synthetic" not in output
        event = json.loads(output.splitlines()[-1])["conditioning_progress"]
        stage = {
            "extra_decode": "timing",
            "unfinished_cuda": "timing",
            "compiled_boundary": "render",
        }.get(failure, failure)
        assert event == {
            "ordinal": 3 if failure in {"embedding", "text_ids"} else 0,
            "state": "failed",
            "stage": stage,
        }
        assert status.renders - before == (4 if failure in {"embedding", "text_ids"} else 1)
        assert status.candidate is False
        assert runtime.pipe.vae.decode is original_decode
        assert "forward" not in vars(runtime.pipe.transformer)
        assert "encode_prompt" not in vars(runtime.pipe)
