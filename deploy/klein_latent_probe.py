"""Capture historical L4 noise and replay its exact bytes through the original pipeline."""

import hashlib
import inspect
import json
import struct
import sys
from contextlib import contextmanager

CAPTURE_SCHEDULE = (("warmup", 0), ("warmup", 1), ("measured", 0), ("measured", 1))
REPLAY_SCHEDULE = (("warmup", 0), ("warmup", 1)) + tuple(("measured", i % 2) for i in range(8))
SHAPE = (1, 128, 36, 64)
PACKED_SHAPE = (1, 2304, 128)
PACKED_STRIDE = (294912, 1, 2304)
IDS_SHAPE = (1, 2304, 4)
NOISE_BYTES = 589824
IDS_SHA256 = hashlib.sha256(
    b"".join(struct.pack("<4q", 0, h, w, 0) for h in range(36) for w in range(64))
).hexdigest()


def require(condition):
    if not condition:
        raise ValueError("latent probe validation failed")


def case_sha256(case):
    return hashlib.sha256(
        json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_artifact(artifact, case, index):
    expected = {
        "schema_version": 1,
        "case_index": index,
        "case_sha256": case_sha256(case),
        "seed": case["seed"],
        "shape": list(SHAPE),
        "dtype": "bfloat16",
        "byte_order": "little",
        "packed_shape": list(PACKED_SHAPE),
        "packed_stride": list(PACKED_STRIDE),
        "latent_ids_shape": list(IDS_SHAPE),
        "latent_ids_dtype": "int64",
        "latent_ids_sha256": IDS_SHA256,
    }
    require(type(artifact) is dict and set(artifact) == set(expected) | {"data", "sha256"})
    # JSON comparison distinguishes bool/float lookalikes from the fixed integer contract.
    require(
        json.dumps({k: artifact[k] for k in expected}, sort_keys=True)
        == json.dumps(expected, sort_keys=True)
    )
    require(type(artifact["data"]) is bytes and len(artifact["data"]) == NOISE_BYTES)
    require(hashlib.sha256(artifact["data"]).hexdigest() == artifact["sha256"])
    return artifact


def tensor_bytes(tensor, torch):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def encode_artifact(noise, ids, case, index, torch):
    require(sys.byteorder == "little")
    require(tuple(noise.shape) == PACKED_SHAPE and tuple(noise.stride()) == PACKED_STRIDE)
    require(
        noise.dtype == torch.bfloat16 and tuple(ids.shape) == IDS_SHAPE and ids.dtype == torch.int64
    )
    require(bool(torch.isfinite(noise).all().item()))
    unpacked = noise.permute(0, 2, 1).reshape(SHAPE)
    data = tensor_bytes(unpacked, torch)
    artifact = {
        "schema_version": 1,
        "case_index": index,
        "case_sha256": case_sha256(case),
        "seed": case["seed"],
        "shape": list(SHAPE),
        "dtype": "bfloat16",
        "byte_order": "little",
        "data": data,
        "sha256": hashlib.sha256(data).hexdigest(),
        "packed_shape": list(noise.shape),
        "packed_stride": list(noise.stride()),
        "latent_ids_shape": list(ids.shape),
        "latent_ids_dtype": "int64",
        "latent_ids_sha256": hashlib.sha256(tensor_bytes(ids, torch)).hexdigest(),
    }
    return validate_artifact(artifact, case, index)


@contextmanager
def prepared_noise(pipe, case, torch, replay=None):
    missing = object()
    previous, original = vars(pipe).get("prepare_latents", missing), pipe.prepare_latents
    signature, observed = inspect.signature(original), []

    def prepare(*args, **kwargs):
        require(not observed)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        require(
            all(
                values[k] == v
                for k, v in {
                    "batch_size": 1,
                    "num_latents_channels": 32,
                    "height": 576,
                    "width": 1024,
                }.items()
            )
        )
        require(values["dtype"] == torch.bfloat16 and torch.device(values["device"]).type == "cuda")
        require(values["latents"] is None and values["generator"].initial_seed() == case["seed"])
        if replay is not None:
            values["latents"] = replay
        noise, ids = original(*bound.args, **bound.kwargs)
        # GPU clones preserve initial values without CPU synchronization inside rendering.
        observed.append((noise.detach().clone(), ids.detach().clone()))
        return noise, ids

    object.__setattr__(pipe, "prepare_latents", prepare)
    try:
        yield observed
        require(len(observed) == 1)
    finally:
        if previous is missing:
            object.__delattr__(pipe, "prepare_latents")
        else:
            object.__setattr__(pipe, "prepare_latents", previous)


def _run(runtime, cases, artifacts):
    import torch

    capture = artifacts is None
    records, captured, prepared = [], {}, []
    ordinal, stage = None, "setup"

    def progress(state, error=None):
        value = {"ordinal": ordinal, "stage": stage, "state": state}
        if error is not None:
            name = type(error).__name__
            value["exception_type"] = (
                name
                if name
                in {
                    "ValueError",
                    "TypeError",
                    "KeyError",
                    "RuntimeError",
                    "MemoryError",
                    "OSError",
                }
                else "OtherError"
            )
        print(json.dumps({"latent_progress": value}), flush=True)

    try:
        require(torch.__version__ == "2.8.0+cu128" and sys.byteorder == "little")
        require(type(cases) is list and len(cases) == 2)
        require([case["expected_bucket"] for case in cases] == [128, 256])
        if not capture:
            require(type(artifacts) is list and len(artifacts) == 2)
            for index, (artifact, case) in enumerate(zip(artifacts, cases, strict=True)):
                validate_artifact(artifact, case, index)
                tensor = torch.frombuffer(bytearray(artifact["data"]), dtype=torch.bfloat16)
                tensor = tensor.clone().reshape(SHAPE).to(device="cuda")
                require(bool(torch.isfinite(tensor).all().item()))
                prepared.append(tensor)
            torch.cuda.synchronize()
        schedule = CAPTURE_SCHEDULE if capture else REPLAY_SCHEDULE
        for ordinal, (phase, index) in enumerate(schedule):
            case, stage = cases[index], "render"
            # Independent input per call; cloning/upload completion precedes runtime timing.
            replay = None if capture else prepared[index].clone()
            if replay is not None:
                torch.cuda.synchronize()
            progress("start")
            with prepared_noise(runtime.pipe, case, torch, replay) as observed:
                metrics, master, depth = runtime.render(case["prompt"], case["seed"])
            row = {
                "ordinal": ordinal,
                "phase": phase,
                "case_index": index,
                "metrics": metrics,
                "master": master,
                "depth": depth,
                "historical_images_exact": False,
                "initial_noise_exact": False,
                "latent_sha256": None,
                "latent_ids_sha256": None,
            }
            records.append(row)
            stage = "verification"
            require(
                metrics["seed"] == case["seed"]
                and metrics["sequence_bucket"] == case["expected_bucket"]
            )
            for name, data in (("master", master), ("depth", depth)):
                require(type(data) is bytes and len(data) > 0)
                require(hashlib.sha256(data).hexdigest() == metrics[f"{name}_sha256"])
            row["historical_images_exact"] = all(
                metrics[f"{name}_sha256"] == case[f"{name}_sha256"] for name in ("master", "depth")
            )
            stage = "noise_verification"
            artifact = encode_artifact(*observed[0], case, index, torch)
            row["latent_sha256"], row["latent_ids_sha256"] = (
                artifact["sha256"],
                artifact["latent_ids_sha256"],
            )
            expected = captured.get(index, artifact) if capture else artifacts[index]
            require(artifact == expected)
            row["initial_noise_exact"] = True
            if capture:
                require(row["historical_images_exact"])
                captured[index] = artifact
            progress("verified")
    except Exception as error:
        progress("failed", error)
        result = {"status": "failed", "failure_stage": stage, "records": records}
    else:
        result = {"status": "complete", "failure_stage": None, "records": records}
    result["artifacts"] = [captured[index] for index in sorted(captured)] if capture else artifacts
    return result


def run_capture(runtime, cases):
    return _run(runtime, cases, None)


def run_replay(runtime, cases, artifacts):
    return _run(runtime, cases, artifacts)
