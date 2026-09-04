"""Provider-independent Klein candidate; pinned, offline loading and bounded token buckets.

Not selected by the live router. Compiler artifacts are executable material: load
only artifacts produced by this runtime in our private, controlled cache.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

MODEL = "black-forest-labs/FLUX.2-klein-4B"
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REVISION = "b4769fd619394250528294b658587285526fab1c"
BUCKETS = (128, 256, 512)
PROFILE = {
    "contract": "klein-regional-default-v1",
    "model": MODEL,
    "model_revision": MODEL_REVISION,
    "depth_model": DEPTH_MODEL,
    "depth_revision": DEPTH_REVISION,
    "width": 1024,
    "height": 576,
    "steps": 4,
    "guidance": 1.0,
    "dtype": "bfloat16",
    "compile_mode": "default",
    "fullgraph": True,
    "buckets": list(BUCKETS),
}


def sequence_bucket(token_count: int) -> int:
    if token_count <= 0:
        raise ValueError("prompt must contain tokens")
    for bucket in BUCKETS:
        if token_count <= bucket:
            return bucket
    raise ValueError(f"prompt has {token_count} tokens; maximum is {BUCKETS[-1]}; not truncating")


def validate_cache(manifest: dict, artifact: bytes, identity: dict) -> None:
    if manifest["identity"] != identity:
        raise ValueError("compiler cache does not match this runtime")
    if manifest["sha256"] != hashlib.sha256(artifact).hexdigest():
        raise ValueError("compiler cache checksum mismatch")


class KleinSceneRuntime:
    def __init__(self, model_root: Path):
        import diffusers
        import torch
        import transformers
        import triton
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline

        started = time.perf_counter()
        expected_weights = {
            "klein": [MODEL, MODEL_REVISION],
            "depth": [DEPTH_MODEL, DEPTH_REVISION],
        }
        if json.loads((model_root / "identities.json").read_text()) != expected_weights:
            raise ValueError("baked weights identity mismatch")
        self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(
            str(model_root / "klein"),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.depth_pipe = pipeline(
            task="depth-estimation",
            model=AutoModelForDepthEstimation.from_pretrained(
                str(model_root / "depth"), torch_dtype=torch.float16, local_files_only=True
            ),
            image_processor=AutoImageProcessor.from_pretrained(
                str(model_root / "depth"), local_files_only=True
            ),
            device=0,
        )
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.identity = {
            **PROFILE,
            "torch": str(torch.__version__),
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def compile(self, cache_directory: Path | None = None) -> float:
        import torch

        started = time.perf_counter()
        if cache_directory is not None:
            manifest = json.loads((cache_directory / "manifest.json").read_text())
            artifact = (cache_directory / "artifacts.bin").read_bytes()
            validate_cache(manifest, artifact, self.identity)
            torch.compiler.load_cache_artifacts(artifact)
        self.pipe.transformer.compile_repeated_blocks(mode="default", fullgraph=True)
        return time.perf_counter() - started

    def save_cache(self, directory: Path) -> dict:
        import torch

        saved = torch.compiler.save_cache_artifacts()
        if saved is None:
            raise RuntimeError("compiler did not produce reusable artifacts")
        artifact, info = saved
        manifest = {
            "identity": self.identity,
            "sha256": hashlib.sha256(artifact).hexdigest(),
            "size_bytes": len(artifact),
            "cache_info": str(info),
        }
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "artifacts.bin").write_bytes(artifact)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    def render(self, prompt: str, seed: int) -> tuple[dict, bytes, bytes]:
        import torch

        if not prompt.strip() or len(prompt) > 4_000:
            raise ValueError("prompt requires 1–4000 characters")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an unsigned 32-bit integer")
        started = time.perf_counter()
        text = self.pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_count = len(self.pipe.tokenizer(text)["input_ids"])
        bucket = sequence_bucket(token_count)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            master = self.pipe(
                prompt=prompt,
                width=PROFILE["width"],
                height=PROFILE["height"],
                num_inference_steps=PROFILE["steps"],
                guidance_scale=PROFILE["guidance"],
                max_sequence_length=bucket,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize()
            image_seconds = time.perf_counter() - started
            depth_started = time.perf_counter()
            depth = self.depth_pipe(master)["depth"].convert("L").resize(master.size)
            torch.cuda.synchronize()
            depth_seconds = time.perf_counter() - depth_started
        encoding_started = time.perf_counter()
        master_buffer, depth_buffer = io.BytesIO(), io.BytesIO()
        master.convert("RGB").save(master_buffer, format="JPEG", quality=95, subsampling=0)
        depth.save(depth_buffer, format="JPEG", quality=85)
        master_bytes, depth_bytes = master_buffer.getvalue(), depth_buffer.getvalue()
        return (
            {
                "seed": seed,
                "token_count": token_count,
                "sequence_bucket": bucket,
                "image_seconds": image_seconds,
                "depth_seconds": depth_seconds,
                "encoding_seconds": time.perf_counter() - encoding_started,
                "total_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "master_sha256": hashlib.sha256(master_bytes).hexdigest(),
                "depth_sha256": hashlib.sha256(depth_bytes).hexdigest(),
            },
            master_bytes,
            depth_bytes,
        )
