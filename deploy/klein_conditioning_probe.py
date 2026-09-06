"""One-worker comparison; stage intervals retain normal CUDA asynchrony."""

import hashlib
import json
import math
import time
from contextlib import contextmanager, nullcontext

from klein_conditioning import backbone_conditioning

SCHEDULE = (
    ("warmup", 0, "baseline"),
    ("warmup", 1, "baseline"),
    ("measured", 0, "baseline"),
    ("measured", 0, "candidate"),
    ("measured", 1, "candidate"),
    ("measured", 1, "baseline"),
    ("measured", 0, "candidate"),
    ("measured", 0, "baseline"),
    ("measured", 1, "baseline"),
    ("measured", 1, "candidate"),
)


def require(condition):
    if not condition:
        raise ValueError("conditioning comparison validation failed")


def progress(ordinal, state, stage):
    print(
        json.dumps({"conditioning_progress": {"ordinal": ordinal, "state": state, "stage": stage}}),
        flush=True,
    )


@contextmanager
def measured_stages(pipe, torch, state):
    """Wrap eager boundaries only; compiled transformer blocks remain untouched."""
    saved = []
    missing = object()

    def wrapper(original, stage):
        def measured(*args, **kwargs):
            require(not torch.compiler.is_compiling())
            first = torch.cuda.Event(enable_timing=True)
            last = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            first.record()
            result = original(*args, **kwargs)
            last.record()
            state["events"][stage].append((first, last, time.perf_counter() - started))
            if stage == "conditioning":
                require(
                    isinstance(result, tuple)
                    and len(result) == 2
                    and all(torch.is_tensor(value) for value in result)
                )
                state["embedding"] = result[0]
                state["text_ids"] = result[1]
            return result

        return measured

    try:
        for owner, name, stage in (
            (pipe, "encode_prompt", "conditioning"),
            (pipe.transformer, "forward", "transformer"),
            (pipe.vae, "decode", "vae_decode"),
        ):
            saved.append((owner, name, vars(owner).get(name, missing)))
            object.__setattr__(owner, name, wrapper(getattr(owner, name), stage))
        yield
    finally:
        for owner, name, previous in reversed(saved):
            if previous is missing:
                object.__delattr__(owner, name)
            else:
                object.__setattr__(owner, name, previous)


def completed_stages(state):
    stages = {}
    for stage, expected_calls in (("conditioning", 1), ("transformer", 4), ("vae_decode", 1)):
        events = state["events"][stage]
        require(len(events) == expected_calls)
        host, cuda = 0.0, 0.0
        for first, last, elapsed in events:
            # The original runtime has already synchronized at image/depth completion.
            require(first.query() and last.query())
            duration = first.elapsed_time(last) / 1000
            require(math.isfinite(duration) and duration >= 0)
            require(math.isfinite(elapsed) and elapsed >= 0)
            host += elapsed
            cuda += duration
        stages[stage] = {
            "calls": len(events),
            "host_seconds": host,
            "cuda_elapsed_seconds": cuda,
        }
    return stages


def run_comparison(runtime, cases):
    """Return ten verified records with JPEGs; never export embedding values.

    Host intervals measure eager execution/enqueueing; CUDA intervals measure stream
    elapsed time, including possible launch gaps. They are not additive. Preparation
    is recomputed on every render; no prompt cache contributes to the comparison.
    """
    import torch

    require(isinstance(cases, list) and len(cases) == 2)
    require([case["expected_bucket"] for case in cases] == [128, 256])
    state, baselines, records = {}, {}, []
    ordinal, verification_stage = None, "setup"
    try:
        with measured_stages(runtime.pipe, torch, state):
            for ordinal, (phase, index, variant) in enumerate(SCHEDULE):
                case = cases[index]
                state.clear()
                state["events"] = {key: [] for key in ("conditioning", "transformer", "vae_decode")}
                context = (
                    backbone_conditioning(runtime.pipe) if variant == "candidate" else nullcontext()
                )
                verification_stage = "render"
                progress(ordinal, "start", verification_stage)
                with context:
                    metrics, master, depth = runtime.render(case["prompt"], case["seed"])
                verification_stage = "timing"
                stages = completed_stages(state)
                require(metrics["sequence_bucket"] == case["expected_bucket"])
                require(metrics["seed"] == case["seed"])
                verification_stage = "jpeg"
                for name, data in (("master", master), ("depth", depth)):
                    require(isinstance(data, bytes))
                    require(
                        hashlib.sha256(data).hexdigest()
                        == metrics[f"{name}_sha256"]
                        == case[f"{name}_sha256"]
                    )
                # All copying, equality and hashing happens after timed rendering.
                reports = {}
                for name in ("embedding", "text_ids"):
                    verification_stage = name
                    tensor = state[name].detach().cpu().contiguous().clone()
                    if phase == "warmup":
                        baselines[index, name] = tensor
                    expected = baselines[index, name]
                    require(tensor.shape == expected.shape and tensor.dtype == expected.dtype)
                    require(torch.equal(tensor, expected))
                    reports[name] = {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "sha256": hashlib.sha256(
                            tensor.view(torch.uint8).numpy().tobytes()
                        ).hexdigest(),
                        "matches_baseline": True,
                    }
                reports["embedding"]["text_ids"] = reports["text_ids"]
                records.append(
                    {
                        "ordinal": ordinal,
                        "phase": phase,
                        "case_index": index,
                        "variant": variant,
                        "metrics": metrics,
                        "stages": stages,
                        "embedding": reports["embedding"],
                        "master": master,
                        "depth": depth,
                    }
                )
                progress(ordinal, "verified", "complete")
    except Exception:
        progress(ordinal, "failed", verification_stage)
        raise ValueError("conditioning comparison failed") from None
    return records
