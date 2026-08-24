"""Bounded staged Modal generation for live Bookforge scenes.

The same pinned classes support finite ``modal run`` jobs and authenticated SDK
lookup after ``modal deploy``. There is deliberately no public web endpoint. Both
classes scale from zero to one container and return to zero after 90 idle seconds. The longer
window keeps the explicitly prewarmed container alive while Jetson finishes local planning.
Each local entry point creates one bounded GPU call and exits::

    modal run deploy/modal_fast_scene.py::fast_scene_cli \
      --scene-id moon-gate-live-001 \
      --prompt "A moonlit watercolor fox beneath a cedar arch" \
      --output-dir artifacts/live-scenes/moon-gate-live-001

    modal run deploy/modal_fast_scene.py::motion_upgrade_cli \
      --scene-manifest-path \
        artifacts/live-scenes/moon-gate-live-001/scene.manifest.json
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "bookforge-fast-scene"
CACHE_DIR = "/cache"
GPU = "L4"
GPU_USD_PER_SECOND = 0.000222
FAST_TIMEOUT_SECONDS = 180
MOTION_TIMEOUT_SECONDS = 300
SCALEDOWN_WINDOW_SECONDS = 90
FAST_PREWARM_WIDTH = 896
FAST_PREWARM_HEIGHT = 512
MASTER_JPEG_QUALITY = 95
DEPTH_JPEG_QUALITY = 85
FAST_MODEL = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
FAST_MODEL_REVISION = "19683c58b7ea290e55cedd8950ae1d86ada7ef96"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_REVISION = "b4769fd619394250528294b658587285526fab1c"
DEPTH_DTYPE = "float16"
MOTION_MODEL = "Lightricks/LTX-Video"
MOTION_MODEL_REVISION = "a6d59ee37c13c58261aa79027d3e41cd41960925"
PROVIDER_NAME = "modal-finite"
DEFAULT_NEGATIVE = (
    "words, letters, captions, logo, watermark, interface, border, split screen, "
    "collage, photorealistic, plastic 3d render, distorted anatomy, duplicate character"
)
DEFAULT_MOTION_PROMPT = (
    "Locked storybook camera. Preserve every character, object, shape, and color. "
    "Add gentle breathing, blinking, drifting paper motes, and warm light. Subtle readable "
    "ambient motion only; no scene transition or new content."
)
DEFAULT_MOTION_NEGATIVE = (
    "camera shake, fast motion, hard cut, scene change, zoom, morphing, melting, flicker, "
    "jitter, inconsistent character, blurry, distorted, text, watermark"
)

runtime_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1")
    .uv_pip_install(
        "accelerate==1.4.0",
        "diffusers==0.39.0",
        "huggingface-hub==0.36.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "opencv-python==4.11.0.86",
        "Pillow==11.1.0",
        "protobuf==5.29.3",
        "safetensors==0.8.0",
        "sentencepiece==0.2.0",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.49.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": CACHE_DIR})
)

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("bookforge-model-cache", create_if_missing=True)

with runtime_image.imports():
    import diffusers
    import torch
    from PIL import Image


def _validate_identifier(value: str) -> None:
    if not value or len(value) > 96 or any(not (char.isalnum() or char in "-_") for char in value):
        raise ValueError("scene_id requires 1-96 letters, numbers, hyphens, or underscores")


def _validate_prompt(value: str, *, name: str) -> None:
    if not value.strip() or len(value) > 4_000:
        raise ValueError(f"{name} requires 1-4000 characters")


def _validate_dimensions(width: int, height: int, *, minimum: int) -> None:
    if not minimum <= width <= 1536 or not minimum <= height <= 1536:
        raise ValueError(f"dimensions must be between {minimum} and 1536")
    if width % 32 or height % 32:
        raise ValueError("dimensions must be divisible by 32")


def _validate_seed(seed: int) -> None:
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")


def _encode_scene_assets(master: Any, depth: Any) -> tuple[bytes, bytes, float]:
    packaging_started = time.perf_counter()

    def encode_master() -> bytes:
        master_buffer = io.BytesIO()
        master.convert("RGB").save(
            master_buffer,
            format="JPEG",
            quality=MASTER_JPEG_QUALITY,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        return master_buffer.getvalue()

    def encode_depth() -> bytes:
        depth_buffer = io.BytesIO()
        depth.convert("L").save(
            depth_buffer,
            format="JPEG",
            quality=DEPTH_JPEG_QUALITY,
            optimize=False,
            progressive=False,
        )
        return depth_buffer.getvalue()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="bookforge-pack") as pool:
        master_future = pool.submit(encode_master)
        depth_future = pool.submit(encode_depth)
        master_bytes = master_future.result()
        depth_bytes = depth_future.result()
    return master_bytes, depth_bytes, time.perf_counter() - packaging_started


@app.cls(
    image=runtime_image,
    gpu=GPU,
    timeout=FAST_TIMEOUT_SECONDS,
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes={CACHE_DIR: model_cache},
)
class FastSceneStudio:
    @modal.enter()
    def load(self) -> None:
        from transformers import pipeline

        load_started = time.perf_counter()
        self.image_pipe = diffusers.SanaSprintPipeline.from_pretrained(
            FAST_MODEL,
            revision=FAST_MODEL_REVISION,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.image_pipe.vae.to(torch.bfloat16)
        self.image_pipe.text_encoder.to(torch.bfloat16)
        self.image_pipe.set_progress_bar_config(disable=True)
        self.depth_pipe = pipeline(
            task="depth-estimation",
            model=DEPTH_MODEL,
            revision=DEPTH_MODEL_REVISION,
            dtype=torch.float16,
            device=0,
        )
        self.model_load_seconds = time.perf_counter() - load_started
        self.inference_warmup_seconds = 0.0
        self.inference_warmed = False
        self.loaded_at = time.monotonic()

    @modal.method()
    def prewarm(self) -> dict[str, Any]:
        if not self.inference_warmed:
            warmup_started = time.perf_counter()
            with torch.inference_mode():
                warmup_master = self.image_pipe(
                    prompt="bright layered paper theater with one simple lantern",
                    width=FAST_PREWARM_WIDTH,
                    height=FAST_PREWARM_HEIGHT,
                    guidance_scale=4.5,
                    num_inference_steps=2,
                    generator=torch.Generator(device="cuda").manual_seed(1),
                ).images[0]
                self.depth_pipe(warmup_master)
            self.inference_warmup_seconds = time.perf_counter() - warmup_started
            self.inference_warmed = True
        return {
            "model": FAST_MODEL,
            "model_revision": FAST_MODEL_REVISION,
            "depth_model": DEPTH_MODEL,
            "depth_model_revision": DEPTH_MODEL_REVISION,
            "depth_dtype": DEPTH_DTYPE,
            "model_load_seconds": self.model_load_seconds,
            "inference_warmup_seconds": self.inference_warmup_seconds,
            "inference_warmed": self.inference_warmed,
            "container_age_seconds": time.monotonic() - self.loaded_at,
        }

    @modal.method()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
    ) -> dict[str, Any]:
        _validate_prompt(prompt, name="prompt")
        _validate_prompt(negative_prompt, name="negative_prompt")
        _validate_seed(seed)
        _validate_dimensions(width, height, minimum=512)
        if not 1 <= steps <= 4:
            raise ValueError("fast-scene steps must be between 1 and 4")
        started = time.perf_counter()
        with torch.inference_mode():
            master = self.image_pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
        image_seconds = time.perf_counter() - started
        depth_started = time.perf_counter()
        with torch.inference_mode():
            depth = self.depth_pipe(master)["depth"].convert("L").resize(
                (width, height), Image.Resampling.LANCZOS
            )
        depth_seconds = time.perf_counter() - depth_started
        master_bytes, depth_bytes, packaging_seconds = _encode_scene_assets(master, depth)
        return {
            "master": master_bytes,
            "master_media_type": "image/jpeg",
            "depth": depth_bytes,
            "depth_media_type": "image/jpeg",
            "depth_dtype": DEPTH_DTYPE,
            "negative_prompt_supported": False,
            "master_jpeg_quality": MASTER_JPEG_QUALITY,
            "depth_jpeg_quality": DEPTH_JPEG_QUALITY,
            "image_seconds": image_seconds,
            "depth_seconds": depth_seconds,
            "packaging_seconds": packaging_seconds,
            "model_load_seconds": self.model_load_seconds,
            "container_age_seconds": time.monotonic() - self.loaded_at,
        }


@app.cls(
    image=runtime_image,
    gpu=GPU,
    timeout=MOTION_TIMEOUT_SECONDS,
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes={CACHE_DIR: model_cache},
)
class MotionUpgradeStudio:
    @modal.enter()
    def load(self) -> None:
        load_started = time.perf_counter()
        self.pipe = diffusers.LTXImageToVideoPipeline.from_pretrained(
            MOTION_MODEL,
            revision=MOTION_MODEL_REVISION,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.model_load_seconds = time.perf_counter() - load_started
        self.loaded_at = time.monotonic()

    @modal.method()
    def prewarm(self) -> dict[str, Any]:
        return {
            "model": MOTION_MODEL,
            "model_revision": MOTION_MODEL_REVISION,
            "model_load_seconds": self.model_load_seconds,
            "container_age_seconds": time.monotonic() - self.loaded_at,
        }

    @modal.method()
    def generate(
        self,
        master_bytes: bytes,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        generated_frames: int,
        fps: int,
        steps: int,
    ) -> dict[str, Any]:
        _validate_prompt(prompt, name="prompt")
        _validate_prompt(negative_prompt, name="negative_prompt")
        _validate_seed(seed)
        _validate_dimensions(width, height, minimum=384)
        if not 9 <= generated_frames <= 65 or (generated_frames - 1) % 8:
            raise ValueError("generated_frames must be 8n+1 and between 9 and 65")
        if not 12 <= fps <= 30:
            raise ValueError("fps must be between 12 and 30")
        if not 8 <= steps <= 40:
            raise ValueError("motion steps must be between 8 and 40")
        source = diffusers.utils.load_image(Image.open(io.BytesIO(master_bytes))).convert("RGB")
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.pipe(
                image=source,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=generated_frames,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).frames[0]
        loop_frames = result + list(reversed(result[:-1]))
        output_path = Path("/tmp/bookforge-live-motion.mp4")
        diffusers.utils.export_to_video(loop_frames, output_path, fps=fps)
        content = output_path.read_bytes()
        output_path.unlink(missing_ok=True)
        generation_seconds = time.perf_counter() - started
        torch.cuda.empty_cache()
        return {
            "content": content,
            "generation_seconds": generation_seconds,
            "frames": len(loop_frames),
            "fps": fps,
            "duration_ms": round(len(loop_frames) / fps * 1000),
            "model_load_seconds": self.model_load_seconds,
            "container_age_seconds": time.monotonic() - self.loaded_at,
        }


def _budget_types():
    repository_source = Path(__file__).resolve().parents[1] / "src"
    if str(repository_source) not in sys.path:
        sys.path.insert(0, str(repository_source))
    from bookforge.modal_budget import (
        require_modal_budget_reservation,
        reserve_modal_budget,
        settle_modal_budget,
    )
    from bookforge.visual_lab import GenerationRecord

    return (
        GenerationRecord,
        require_modal_budget_reservation,
        reserve_modal_budget,
        settle_modal_budget,
    )


def _guard_and_reserve(
    *,
    plan_file: str,
    ledger_path: str,
    experiment_id: str,
    timeout_seconds: int,
    maximum_gpu_usd: float,
    existing_reservation_id: str = "",
) -> str:
    _, require_reservation, reserve_modal_budget, _ = _budget_types()
    if existing_reservation_id:
        require_reservation(
            plan_path=Path(plan_file),
            ledger_path=Path(ledger_path),
            reservation_id=existing_reservation_id,
            expected_experiment_id=experiment_id,
        )
        return existing_reservation_id
    return reserve_modal_budget(
        plan_path=Path(plan_file),
        ledger_path=Path(ledger_path),
        experiment_id=experiment_id,
        gpu=GPU,
        timeout_seconds=timeout_seconds,
        maximum_gpu_usd=maximum_gpu_usd,
    )


def _record_budget(
    *,
    plan_file: str,
    ledger_path: str,
    reservation_id: str,
    experiment_id: str,
    stage: str,
    model: str,
    revision: str,
    seed: int,
    prompt: str,
    artifact_path: Path,
    sha256: str,
    remote_seconds: float,
    width: int,
    height: int,
    frames: int,
    fps: int,
) -> None:
    GenerationRecord, _, _, settle_modal_budget = _budget_types()
    settle_modal_budget(
        plan_path=Path(plan_file),
        ledger_path=Path(ledger_path),
        reservation_id=reservation_id,
        record=GenerationRecord(
            experiment_id=experiment_id,
            stage=stage,
            model=model,
            model_revision=revision,
            gpu=GPU,
            seed=seed,
            prompt=prompt,
            artifact_path=str(artifact_path),
            sha256=sha256,
            generation_seconds=remote_seconds,
            estimated_gpu_usd=remote_seconds * GPU_USD_PER_SECOND,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
        ),
    )


def _artifact_record(
    *,
    path: Path,
    root: Path,
    mime_type: str,
    width: int,
    height: int,
    duration_ms: int = 0,
    frames: int = 1,
    fps: int = 0,
) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "mime_type": mime_type,
        "width": width,
        "height": height,
        "duration_ms": duration_ms,
        "frames": frames,
        "fps": fps,
    }


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_source_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "1.0" or payload.get("provider") != PROVIDER_NAME:
        raise ValueError("motion source is not a finite Modal scene manifest")
    if "motion" in payload.get("artifacts", {}) or "motion" in payload.get("stages", {}):
        raise ValueError("scene already contains a motion upgrade")
    master = payload.get("artifacts", {}).get("master")
    if not isinstance(master, dict):
        raise ValueError("scene manifest has no master artifact")
    relative = Path(str(master.get("path", "")))
    root = path.parent.resolve()
    source = (root / relative).resolve()
    if relative.is_absolute() or not source.is_relative_to(root) or not source.is_file():
        raise ValueError("scene master path is invalid")
    if hashlib.sha256(source.read_bytes()).hexdigest() != master.get("sha256"):
        raise ValueError("scene master checksum mismatch")
    return payload


@app.local_entrypoint()
def fast_scene_cli(
    scene_id: str,
    prompt: str,
    output_dir: str,
    negative_prompt: str = DEFAULT_NEGATIVE,
    seed: int = 42,
    width: int = 1024,
    height: int = 576,
    steps: int = 2,
    guidance_scale: float = 4.5,
    plan_file: str = "experiments/live-scenes/modal-plan.json",
    ledger_path: str = "artifacts/live-scenes/modal-ledger.json",
    maximum_gpu_usd: float = 0.08,
    experiment_id: str = "",
    reservation_id: str = "",
) -> None:
    _validate_identifier(scene_id)
    _validate_prompt(prompt, name="prompt")
    _validate_prompt(negative_prompt, name="negative_prompt")
    _validate_seed(seed)
    _validate_dimensions(width, height, minimum=512)
    if not 1 <= steps <= 4:
        raise ValueError("fast-scene steps must be between 1 and 4")
    destination = Path(output_dir).resolve()
    manifest_path = destination / "scene.manifest.json"
    master_path = destination / "master.jpg"
    depth_path = destination / "depth.jpg"
    if any(path.exists() for path in (manifest_path, master_path, depth_path)):
        raise ValueError(f"scene output already exists: {destination}")
    identity = hashlib.sha256(
        json.dumps(
            {
                "scene_id": scene_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance_scale,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    computed_experiment_id = f"fast-scene:{scene_id}:{identity}"
    if experiment_id and experiment_id != computed_experiment_id:
        raise ValueError("external fast-scene experiment identity does not match the request")
    experiment_id = computed_experiment_id
    reservation_id = _guard_and_reserve(
        plan_file=plan_file,
        ledger_path=ledger_path,
        experiment_id=experiment_id,
        timeout_seconds=FAST_TIMEOUT_SECONDS,
        maximum_gpu_usd=maximum_gpu_usd,
        existing_reservation_id=reservation_id,
    )
    remote_started = time.perf_counter()
    result = FastSceneStudio().generate.remote(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
    )
    if result.get("master_media_type") != "image/jpeg":
        raise RuntimeError("fast scene returned an unsupported master format")
    if result.get("depth_media_type") != "image/jpeg":
        raise RuntimeError("fast scene returned an unsupported depth format")
    if result.get("negative_prompt_supported") is not False:
        raise RuntimeError("fast scene returned ambiguous negative-prompt provenance")
    if result.get("depth_dtype") != DEPTH_DTYPE:
        raise RuntimeError("fast scene returned ambiguous depth precision")
    remote_seconds = time.perf_counter() - remote_started
    estimated_gpu_usd = remote_seconds * GPU_USD_PER_SECOND
    if estimated_gpu_usd > maximum_gpu_usd + 1e-9:
        # Do not release the ledger reservation when the provider exceeds its
        # declared cap. This fails closed until billing is reconciled.
        raise RuntimeError(
            f"fast scene exceeded its ${maximum_gpu_usd:.6f} call cap: "
            f"${estimated_gpu_usd:.6f} estimated"
        )
    _atomic_write_bytes(master_path, result["master"])
    _atomic_write_bytes(depth_path, result["depth"])
    now = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": "1.0",
        "provider": PROVIDER_NAME,
        "scene_id": scene_id,
        "created_at": now,
        "updated_at": now,
        "request": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
        },
        "policy": {
            "finite_calls_only": True,
            "persistent_endpoint": False,
            "budget_plan": plan_file,
            "budget_ledger": ledger_path,
        },
        "stages": {
            "fast": {
                "model": FAST_MODEL,
                "model_revision": FAST_MODEL_REVISION,
                "additional_models": [
                    {"model": DEPTH_MODEL, "model_revision": DEPTH_MODEL_REVISION}
                ],
                "gpu": GPU,
                "finite_call": True,
                "hard_timeout_seconds": FAST_TIMEOUT_SECONDS,
                "remote_seconds": remote_seconds,
                "inference_seconds": result["image_seconds"] + result["depth_seconds"],
                "provider_overhead_seconds": max(
                    0.0,
                    remote_seconds - result["image_seconds"] - result["depth_seconds"],
                ),
                "image_seconds": result["image_seconds"],
                "depth_seconds": result["depth_seconds"],
                "packaging_seconds": result["packaging_seconds"],
                "master_jpeg_quality": result["master_jpeg_quality"],
                "depth_jpeg_quality": result["depth_jpeg_quality"],
                "depth_dtype": DEPTH_DTYPE,
                "negative_prompt_supported": False,
                "model_load_seconds": result.get("model_load_seconds", 0),
                "container_age_seconds": result.get("container_age_seconds", 0),
                "warm_state": "cold",
                "estimated_gpu_usd": estimated_gpu_usd,
                "steps": steps,
                "guidance_scale": guidance_scale,
            }
        },
        "artifacts": {
            "master": _artifact_record(
                path=master_path,
                root=destination,
                mime_type="image/jpeg",
                width=width,
                height=height,
            ),
            "depth": _artifact_record(
                path=depth_path,
                root=destination,
                mime_type="image/jpeg",
                width=width,
                height=height,
            ),
        },
    }
    _atomic_write_json(manifest_path, payload)
    _record_budget(
        plan_file=plan_file,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        experiment_id=experiment_id,
        stage="fast-scene",
        model=FAST_MODEL,
        revision=FAST_MODEL_REVISION,
        seed=seed,
        prompt=prompt,
        artifact_path=master_path,
        sha256=payload["artifacts"]["master"]["sha256"],
        remote_seconds=remote_seconds,
        width=width,
        height=height,
        frames=1,
        fps=0,
    )
    print(json.dumps(payload, sort_keys=True))


@app.local_entrypoint()
def motion_upgrade_cli(
    scene_manifest_path: str,
    prompt: str = DEFAULT_MOTION_PROMPT,
    negative_prompt: str = DEFAULT_MOTION_NEGATIVE,
    seed: int = 43,
    width: int = 800,
    height: int = 448,
    generated_frames: int = 41,
    fps: int = 24,
    steps: int = 24,
    plan_file: str = "experiments/live-scenes/modal-plan.json",
    ledger_path: str = "artifacts/live-scenes/modal-ledger.json",
    maximum_gpu_usd: float = 0.12,
    experiment_id: str = "",
    reservation_id: str = "",
) -> None:
    manifest_path = Path(scene_manifest_path).resolve()
    payload = _load_source_manifest(manifest_path)
    scene_id = str(payload["scene_id"])
    _validate_identifier(scene_id)
    _validate_prompt(prompt, name="prompt")
    _validate_prompt(negative_prompt, name="negative_prompt")
    _validate_seed(seed)
    _validate_dimensions(width, height, minimum=384)
    if not 9 <= generated_frames <= 65 or (generated_frames - 1) % 8:
        raise ValueError("generated_frames must be 8n+1 and between 9 and 65")
    if not 12 <= fps <= 30:
        raise ValueError("fps must be between 12 and 30")
    if not 8 <= steps <= 40:
        raise ValueError("motion steps must be between 8 and 40")
    destination = manifest_path.parent
    master_path = destination / str(payload["artifacts"]["master"]["path"])
    motion_path = destination / "motion.mp4"
    if motion_path.exists():
        raise ValueError(f"motion output already exists: {motion_path}")
    identity = hashlib.sha256(
        json.dumps(
            {
                "scene_id": scene_id,
                "master_sha256": payload["artifacts"]["master"]["sha256"],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "generated_frames": generated_frames,
                "fps": fps,
                "steps": steps,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    computed_experiment_id = f"live-motion:{scene_id}:{identity}"
    if experiment_id and experiment_id != computed_experiment_id:
        raise ValueError("external motion experiment identity does not match the request")
    experiment_id = computed_experiment_id
    reservation_id = _guard_and_reserve(
        plan_file=plan_file,
        ledger_path=ledger_path,
        experiment_id=experiment_id,
        timeout_seconds=MOTION_TIMEOUT_SECONDS,
        maximum_gpu_usd=maximum_gpu_usd,
        existing_reservation_id=reservation_id,
    )
    remote_started = time.perf_counter()
    result = MotionUpgradeStudio().generate.remote(
        master_bytes=master_path.read_bytes(),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        generated_frames=generated_frames,
        fps=fps,
        steps=steps,
    )
    remote_seconds = time.perf_counter() - remote_started
    estimated_gpu_usd = remote_seconds * GPU_USD_PER_SECOND
    if estimated_gpu_usd > maximum_gpu_usd + 1e-9:
        raise RuntimeError(
            f"motion upgrade exceeded its ${maximum_gpu_usd:.6f} call cap: "
            f"${estimated_gpu_usd:.6f} estimated"
        )
    _atomic_write_bytes(motion_path, result["content"])
    payload["updated_at"] = datetime.now(UTC).isoformat()
    payload["stages"]["motion"] = {
        "model": MOTION_MODEL,
        "model_revision": MOTION_MODEL_REVISION,
        "source_master_sha256": payload["artifacts"]["master"]["sha256"],
        "gpu": GPU,
        "finite_call": True,
        "hard_timeout_seconds": MOTION_TIMEOUT_SECONDS,
        "remote_seconds": remote_seconds,
        "inference_seconds": result["generation_seconds"],
        "provider_overhead_seconds": max(
            0.0,
            remote_seconds - result["generation_seconds"],
        ),
        "model_load_seconds": result.get("model_load_seconds", 0),
        "container_age_seconds": result.get("container_age_seconds", 0),
        "warm_state": "cold",
        "estimated_gpu_usd": estimated_gpu_usd,
        "steps": steps,
        "generated_frames": generated_frames,
        "loop_strategy": "exact-ping-pong",
    }
    payload["artifacts"]["motion"] = _artifact_record(
        path=motion_path,
        root=destination,
        mime_type="video/mp4",
        width=width,
        height=height,
        duration_ms=result["duration_ms"],
        frames=result["frames"],
        fps=result["fps"],
    )
    _atomic_write_json(manifest_path, payload)
    _record_budget(
        plan_file=plan_file,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        experiment_id=experiment_id,
        stage="live-motion",
        model=MOTION_MODEL,
        revision=MOTION_MODEL_REVISION,
        seed=seed,
        prompt=prompt,
        artifact_path=motion_path,
        sha256=payload["artifacts"]["motion"]["sha256"],
        remote_seconds=remote_seconds,
        width=width,
        height=height,
        frames=result["frames"],
        fps=result["fps"],
    )
    print(json.dumps(payload, sort_keys=True))
