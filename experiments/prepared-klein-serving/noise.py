"""Initial-noise receipts; original prepare_latents call and RNG remain unchanged."""

import hashlib
import inspect
import struct
import sys
from contextlib import contextmanager

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


def tensor_bytes(tensor, torch):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


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


def noise_receipt(observed, ordinal, case):
    import torch

    require(torch.__version__ == "2.8.0+cu128" and sys.byteorder == "little")
    require(len(observed) == 1 and type(ordinal) is int and 1 <= ordinal <= 12)
    noise, ids = observed[0]
    require(tuple(noise.shape) == PACKED_SHAPE and tuple(noise.stride()) == PACKED_STRIDE)
    require(noise.dtype == torch.bfloat16)
    require(tuple(ids.shape) == IDS_SHAPE and ids.dtype == torch.int64)
    # Called only after runtime.render returns; these copies/hashes remain inside worker time.
    data = tensor_bytes(noise.permute(0, 2, 1).reshape(SHAPE), torch)
    ids_hash = hashlib.sha256(tensor_bytes(ids, torch)).hexdigest()
    require(len(data) == NOISE_BYTES and ids_hash == IDS_SHA256)
    return {
        "ordinal": ordinal,
        "case_id": case["case_id"],
        "seed": case["seed"],
        "latent_sha256": hashlib.sha256(data).hexdigest(),
        "latent_ids_sha256": ids_hash,
    }
