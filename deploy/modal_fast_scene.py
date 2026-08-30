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
FAST_GPU = "L4"
FAST_GPU_USD_PER_SECOND = 0.000222
MOTION_GPU = "L4"
MOTION_GPU_USD_PER_SECOND = 0.000222
FAST_TIMEOUT_SECONDS = 180
MOTION_TIMEOUT_SECONDS = 300
SCALEDOWN_WINDOW_SECONDS = 90
FAST_PREWARM_WIDTH = 1024
FAST_PREWARM_HEIGHT = 576
MASTER_JPEG_QUALITY = 95
DEPTH_JPEG_QUALITY = 85
FAST_MODEL = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
FAST_MODEL_REVISION = "19683c58b7ea290e55cedd8950ae1d86ada7ef96"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_REVISION = "b4769fd619394250528294b658587285526fab1c"
DEPTH_DTYPE = "float16"
FIDELITY_MODEL = "IDEA-Research/grounding-dino-tiny"
FIDELITY_MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
FIDELITY_THRESHOLD = 0.25
FIDELITY_SHORTEST_EDGE = 512
FIDELITY_LONGEST_EDGE = 768
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


def _box_containment(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Intersection divided by the smaller box area, for nested-box suppression."""

    first_x0, first_y0, first_x1, first_y1 = first
    second_x0, second_y0, second_x1, second_y1 = second
    overlap_width = max(0.0, min(first_x1, second_x1) - max(first_x0, second_x0))
    overlap_height = max(0.0, min(first_y1, second_y1) - max(first_y0, second_y0))
    smaller_area = min(
        max(0.0, first_x1 - first_x0) * max(0.0, first_y1 - first_y0),
        max(0.0, second_x1 - second_x0) * max(0.0, second_y1 - second_y0),
    )
    return (overlap_width * overlap_height) / smaller_area if smaller_area else 0.0


def _box_intersection_fraction(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Intersection divided by the first box area."""

    first_x0, first_y0, first_x1, first_y1 = first
    second_x0, second_y0, second_x1, second_y1 = second
    overlap_width = max(0.0, min(first_x1, second_x1) - max(first_x0, second_x0))
    overlap_height = max(0.0, min(first_y1, second_y1) - max(first_y0, second_y0))
    first_area = max(0.0, first_x1 - first_x0) * max(0.0, first_y1 - first_y0)
    return (overlap_width * overlap_height) / first_area if first_area else 0.0


def _collapse_nested_detections(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if any(
            _box_containment(candidate["box"], prior["box"]) >= 0.8
            for prior in accepted
        ):
            continue
        accepted.append(candidate)
    return accepted


def _runtime_gpu_name() -> str:
    name = torch.cuda.get_device_name(0).upper()
    if "L40S" in name:
        return "L40S"
    if "L4" in name:
        return "L4"
    raise RuntimeError(f"unsupported BookForge GPU: {name}")


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
    gpu=FAST_GPU,
    timeout=FAST_TIMEOUT_SECONDS,
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    volumes={CACHE_DIR: model_cache},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class FastSceneStudio:
    @modal.enter(snap=True)
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
        # Grounding DINO is deliberately absent from the first-plate GPU
        # snapshot. Deferred-fidelity requests never need its weights; strict
        # inline requests load it only when they actually request the gate.
        self.fidelity_processor = None
        self.fidelity_model = None
        self.fidelity_model_load_seconds = 0.0
        self.model_load_seconds = time.perf_counter() - load_started
        self.gpu = _runtime_gpu_name()
        self.inference_warmup_seconds = 0.0
        self.inference_warmed = False
        self._warm_inference()
        self.loaded_at = time.monotonic()

    def _ensure_fidelity_model(self) -> float:
        if self.fidelity_processor is not None and self.fidelity_model is not None:
            return 0.0
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        started = time.perf_counter()
        self.fidelity_processor = AutoProcessor.from_pretrained(
            FIDELITY_MODEL,
            revision=FIDELITY_MODEL_REVISION,
        )
        self.fidelity_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            FIDELITY_MODEL,
            revision=FIDELITY_MODEL_REVISION,
        ).to("cuda")
        self.fidelity_model.eval()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.fidelity_model_load_seconds += elapsed
        return elapsed

    @modal.enter(snap=False)
    def restore(self) -> None:
        # Snapshot state includes the warmed CUDA pipelines. Reset only the
        # per-container clock after restore; request seeds remain explicit.
        self.loaded_at = time.monotonic()

    def _warm_inference(self) -> None:
        if self.inference_warmed:
            return
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
            self._warmup_master = warmup_master
        torch.cuda.synchronize()
        self.inference_warmup_seconds = time.perf_counter() - warmup_started
        self.inference_warmed = True

    @modal.method()
    def prewarm(self, include_fidelity: bool = True) -> dict[str, Any]:
        self._warm_inference()
        fidelity_warmup_seconds = 0.0
        if include_fidelity:
            fidelity_started = time.perf_counter()
            self._ensure_fidelity_model()
            self._evaluate_fidelity(
                self._warmup_master,
                subject_label="lantern",
                object_label="",
                expected_subject_count=1,
                require_overlap=False,
            )
            fidelity_warmup_seconds = time.perf_counter() - fidelity_started
        return {
            "model": FAST_MODEL,
            "model_revision": FAST_MODEL_REVISION,
            "depth_model": DEPTH_MODEL,
            "depth_model_revision": DEPTH_MODEL_REVISION,
            "depth_dtype": DEPTH_DTYPE,
            "fidelity_model": FIDELITY_MODEL,
            "fidelity_model_revision": FIDELITY_MODEL_REVISION,
            "fidelity_loaded": self.fidelity_model is not None,
            "fidelity_model_load_seconds": self.fidelity_model_load_seconds,
            "fidelity_warmup_seconds": fidelity_warmup_seconds,
            "model_load_seconds": self.model_load_seconds,
            "inference_warmup_seconds": self.inference_warmup_seconds,
            "inference_warmed": self.inference_warmed,
            "container_age_seconds": time.monotonic() - self.loaded_at,
            "gpu": self.gpu,
            "gpu_memory_snapshot": True,
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
        fidelity_label: str = "",
        fidelity_object_label: str = "",
        require_subject_object_overlap: bool = False,
        expected_subject_count: int = 1,
    ) -> dict[str, Any]:
        _validate_prompt(prompt, name="prompt")
        _validate_prompt(negative_prompt, name="negative_prompt")
        _validate_seed(seed)
        _validate_dimensions(width, height, minimum=512)
        if not 1 <= steps <= 4:
            raise ValueError("fast-scene steps must be between 1 and 4")
        if len(fidelity_label) > 80 or (fidelity_label and not fidelity_label.strip()):
            raise ValueError("fidelity_label must contain at most 80 characters")
        if len(fidelity_object_label) > 80 or (
            fidelity_object_label and not fidelity_object_label.strip()
        ):
            raise ValueError("fidelity_object_label must contain at most 80 characters")
        if require_subject_object_overlap and not fidelity_object_label:
            raise ValueError("subject/object overlap requires an object label")
        if not 1 <= expected_subject_count <= 4:
            raise ValueError("expected_subject_count must be between 1 and 4")
        master, image_seconds = self._generate_master(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
        )
        selected_seed = seed
        quality_attempts = 0
        quality_seconds = 0.0
        subject_count: int | None = None
        object_count: int | None = None
        subject_object_overlap: bool | None = None
        quality_scores: list[float] = []
        if fidelity_label:
            quality_seconds += self._ensure_fidelity_model()
            quality, detection_seconds = self._evaluate_fidelity(
                master,
                subject_label=fidelity_label,
                object_label=fidelity_object_label,
                expected_subject_count=expected_subject_count,
                require_overlap=require_subject_object_overlap,
            )
            quality_attempts = 1
            quality_seconds += detection_seconds
            subject_count = quality["subject_count"]
            object_count = quality["object_count"]
            subject_object_overlap = quality["subject_object_overlap"]
            quality_scores = quality["scores"]
            if not quality["passed"]:
                retry_seed = (seed + 1) % (2**32)
                retry_master, retry_image_seconds = self._generate_master(
                    prompt=prompt,
                    seed=retry_seed,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                )
                retry_quality, retry_detection_seconds = self._evaluate_fidelity(
                    retry_master,
                    subject_label=fidelity_label,
                    object_label=fidelity_object_label,
                    expected_subject_count=expected_subject_count,
                    require_overlap=require_subject_object_overlap,
                )
                image_seconds += retry_image_seconds
                quality_seconds += retry_detection_seconds
                quality_attempts = 2
                if retry_quality["passed"]:
                    master = retry_master
                    selected_seed = retry_seed
                    subject_count = retry_quality["subject_count"]
                    object_count = retry_quality["object_count"]
                    subject_object_overlap = retry_quality["subject_object_overlap"]
                    quality_scores = retry_quality["scores"]
                    quality = retry_quality
            if not quality["passed"]:
                raise RuntimeError(
                    "scene fidelity gate rejected both bounded candidates: "
                    f"subject={subject_count}, object={object_count}, "
                    f"overlap={subject_object_overlap}"
                )
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
            "quality_seconds": quality_seconds,
            "quality_attempts": quality_attempts,
            "quality_label": fidelity_label,
            "quality_object_label": fidelity_object_label,
            "quality_expected_count": expected_subject_count,
            "quality_subject_count": subject_count,
            "quality_object_count": object_count,
            "quality_subject_object_overlap": subject_object_overlap,
            "quality_scores": quality_scores,
            "quality_passed": quality["passed"] if fidelity_label else None,
            "fidelity_loaded": self.fidelity_model is not None,
            "fidelity_model_load_seconds": self.fidelity_model_load_seconds,
            "selected_seed": selected_seed,
            "packaging_seconds": packaging_seconds,
            "model_load_seconds": self.model_load_seconds,
            "container_age_seconds": time.monotonic() - self.loaded_at,
            "gpu": self.gpu,
        }

    @modal.method()
    def generate_preview(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        guidance_scale: float,
    ) -> dict[str, Any]:
        """Generate an explicitly provisional low-resolution plate without depth work."""

        _validate_prompt(prompt, name="prompt")
        _validate_seed(seed)
        _validate_dimensions(width, height, minimum=256)
        if width > 640 or height > 384:
            raise ValueError("preview dimensions cannot exceed 640x384")
        master, image_seconds = self._generate_master(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            steps=1,
            guidance_scale=guidance_scale,
        )
        packaging_started = time.perf_counter()
        master_buffer = io.BytesIO()
        master.convert("RGB").save(
            master_buffer,
            format="JPEG",
            quality=MASTER_JPEG_QUALITY,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        return {
            "master": master_buffer.getvalue(),
            "master_media_type": "image/jpeg",
            "negative_prompt_supported": False,
            "master_jpeg_quality": MASTER_JPEG_QUALITY,
            "image_seconds": image_seconds,
            "packaging_seconds": time.perf_counter() - packaging_started,
            "model_load_seconds": self.model_load_seconds,
            "container_age_seconds": time.monotonic() - self.loaded_at,
            "gpu": self.gpu,
        }

    def _generate_master(
        self,
        *,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
    ) -> tuple[Any, float]:
        # Diffusers defaults SCM's intermediate timestep to 1.3, then rejects
        # that value for every non-two-step request. Passing None is the
        # documented one/three/four-step path; two-step keeps the tuned 1.3.
        sprint_timing = {"intermediate_timesteps": None} if steps != 2 else {}
        started = time.perf_counter()
        with torch.inference_mode():
            master = self.image_pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                **sprint_timing,
            ).images[0]
        image_seconds = time.perf_counter() - started
        return master, image_seconds

    def _evaluate_fidelity(
        self,
        image: Any,
        *,
        subject_label: str,
        object_label: str,
        expected_subject_count: int,
        require_overlap: bool,
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        labels = [subject_label, *([object_label] if object_label else [])]
        inputs = self.fidelity_processor(
            images=image,
            text=[labels],
            return_tensors="pt",
            size={
                "shortest_edge": FIDELITY_SHORTEST_EDGE,
                "longest_edge": FIDELITY_LONGEST_EDGE,
            },
        ).to("cuda")
        with torch.inference_mode():
            outputs = self.fidelity_model(**inputs)
        result = self.fidelity_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=FIDELITY_THRESHOLD,
            text_threshold=FIDELITY_THRESHOLD,
            target_sizes=[(image.height, image.width)],
        )[0]
        grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
        for score, box, text_label in zip(
            result["scores"],
            result["boxes"].tolist(),
            result["text_labels"],
            strict=True,
        ):
            normalized = str(text_label).casefold().strip(" .")
            label = next(
                (candidate for candidate in labels if candidate.casefold() == normalized),
                None,
            )
            if label is None:
                continue
            grouped[label].append(
                {
                    "score": float(score),
                    "box": tuple(float(value) for value in box),
                }
            )
        for label in labels:
            grouped[label] = _collapse_nested_detections(grouped[label])
        subjects = grouped[subject_label]
        objects = grouped.get(object_label, [])
        overlap = (
            any(
                _box_intersection_fraction(subject["box"], item["box"]) >= 0.05
                for subject in subjects
                for item in objects
            )
            if object_label
            else None
        )
        passed = len(subjects) == expected_subject_count
        if object_label:
            passed = passed and len(objects) == 1
        if require_overlap:
            passed = passed and overlap is True
        torch.cuda.synchronize()
        return (
            {
                "passed": passed,
                "subject_count": len(subjects),
                "object_count": len(objects) if object_label else None,
                "subject_object_overlap": overlap,
                "scores": [item["score"] for item in [*subjects, *objects]],
            },
            time.perf_counter() - started,
        )


@app.cls(
    image=runtime_image,
    gpu=MOTION_GPU,
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
        self.gpu = _runtime_gpu_name()
        self.loaded_at = time.monotonic()

    @modal.method()
    def prewarm(self) -> dict[str, Any]:
        return {
            "model": MOTION_MODEL,
            "model_revision": MOTION_MODEL_REVISION,
            "model_load_seconds": self.model_load_seconds,
            "container_age_seconds": time.monotonic() - self.loaded_at,
            "gpu": self.gpu,
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
            "gpu": self.gpu,
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
    gpu: str,
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
        gpu=gpu,
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
    gpu: str,
    gpu_usd_per_second: float,
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
            gpu=gpu,
            seed=seed,
            prompt=prompt,
            artifact_path=str(artifact_path),
            sha256=sha256,
            generation_seconds=remote_seconds,
            estimated_gpu_usd=remote_seconds * gpu_usd_per_second,
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
    fidelity_label: str = "",
    fidelity_object_label: str = "",
    require_subject_object_overlap: bool = False,
    expected_subject_count: int = 1,
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
                "fidelity_label": fidelity_label,
                "fidelity_object_label": fidelity_object_label,
                "require_subject_object_overlap": require_subject_object_overlap,
                "expected_subject_count": expected_subject_count,
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
        gpu=FAST_GPU,
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
        fidelity_label=fidelity_label,
        fidelity_object_label=fidelity_object_label,
        require_subject_object_overlap=require_subject_object_overlap,
        expected_subject_count=expected_subject_count,
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
    estimated_gpu_usd = remote_seconds * FAST_GPU_USD_PER_SECOND
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
            "selected_seed": result.get("selected_seed", seed),
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
                    {"model": DEPTH_MODEL, "model_revision": DEPTH_MODEL_REVISION},
                    *(
                        [
                            {
                                "model": FIDELITY_MODEL,
                                "model_revision": FIDELITY_MODEL_REVISION,
                            }
                        ]
                        if fidelity_label
                        else []
                    ),
                ],
                "gpu": FAST_GPU,
                "finite_call": True,
                "hard_timeout_seconds": FAST_TIMEOUT_SECONDS,
                "remote_seconds": remote_seconds,
                "inference_seconds": (
                    result["image_seconds"]
                    + result["depth_seconds"]
                    + result.get("quality_seconds", 0)
                ),
                "provider_overhead_seconds": max(
                    0.0,
                    remote_seconds
                    - result["image_seconds"]
                    - result["depth_seconds"]
                    - result.get("quality_seconds", 0),
                ),
                "image_seconds": result["image_seconds"],
                "depth_seconds": result["depth_seconds"],
                "quality_seconds": result.get("quality_seconds", 0),
                "quality_attempts": result.get("quality_attempts", 0),
                "quality_label": result.get("quality_label", ""),
                "quality_object_label": result.get("quality_object_label", ""),
                "quality_expected_count": result.get("quality_expected_count"),
                "quality_subject_count": result.get("quality_subject_count"),
                "quality_object_count": result.get("quality_object_count"),
                "quality_subject_object_overlap": result.get(
                    "quality_subject_object_overlap"
                ),
                "quality_scores": result.get("quality_scores", []),
                "quality_passed": result.get("quality_passed"),
                "selected_seed": result.get("selected_seed", seed),
                "packaging_seconds": result["packaging_seconds"],
                "master_jpeg_quality": result["master_jpeg_quality"],
                "depth_jpeg_quality": result["depth_jpeg_quality"],
                "depth_dtype": DEPTH_DTYPE,
                "negative_prompt_supported": False,
                "model_load_seconds": result.get("model_load_seconds", 0),
                "fidelity_model_load_seconds": result.get(
                    "fidelity_model_load_seconds", 0
                ),
                "fidelity_loaded": result.get("fidelity_loaded", False),
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
        gpu=FAST_GPU,
        gpu_usd_per_second=FAST_GPU_USD_PER_SECOND,
    )
    print(json.dumps(payload, sort_keys=True))


@app.local_entrypoint()
def preview_scene_cli(
    scene_id: str,
    prompt: str,
    output_dir: str,
    seed: int = 42,
    width: int = 512,
    height: int = 288,
    guidance_scale: float = 4.5,
    plan_file: str = "experiments/live-scenes/modal-plan.json",
    ledger_path: str = "artifacts/live-scenes/modal-ledger.json",
    maximum_gpu_usd: float = 0.08,
    prewarm_first: bool = False,
    experiment_id: str = "",
    reservation_id: str = "",
) -> None:
    """Run one budgeted provisional-plate experiment; never a final scene."""

    _validate_identifier(scene_id)
    _validate_prompt(prompt, name="prompt")
    _validate_seed(seed)
    _validate_dimensions(width, height, minimum=256)
    if width > 640 or height > 384:
        raise ValueError("preview dimensions cannot exceed 640x384")
    destination = Path(output_dir).resolve()
    manifest_path = destination / "preview.manifest.json"
    master_path = destination / "preview.jpg"
    if manifest_path.exists() or master_path.exists():
        raise ValueError(f"preview output already exists: {destination}")
    identity = hashlib.sha256(
        json.dumps(
            {
                "scene_id": scene_id,
                "prompt": prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "guidance_scale": guidance_scale,
                "prewarm_first": prewarm_first,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    computed_experiment_id = f"preview-scene:{scene_id}:{identity}"
    if experiment_id and experiment_id != computed_experiment_id:
        raise ValueError("external preview experiment identity does not match the request")
    experiment_id = computed_experiment_id
    reservation_id = _guard_and_reserve(
        plan_file=plan_file,
        ledger_path=ledger_path,
        experiment_id=experiment_id,
        timeout_seconds=FAST_TIMEOUT_SECONDS,
        maximum_gpu_usd=maximum_gpu_usd,
        gpu=FAST_GPU,
        existing_reservation_id=reservation_id,
    )
    studio = FastSceneStudio()
    prewarm_remote_seconds = 0.0
    prewarm_receipt: dict[str, Any] = {}
    if prewarm_first:
        prewarm_started = time.perf_counter()
        prewarm_receipt = studio.prewarm.remote()
        prewarm_remote_seconds = time.perf_counter() - prewarm_started
    remote_started = time.perf_counter()
    result = studio.generate_preview.remote(
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
    )
    if result.get("master_media_type") != "image/jpeg":
        raise RuntimeError("preview scene returned an unsupported master format")
    if result.get("negative_prompt_supported") is not False:
        raise RuntimeError("preview scene returned ambiguous prompt provenance")
    remote_seconds = time.perf_counter() - remote_started
    billable_remote_seconds = prewarm_remote_seconds + remote_seconds
    estimated_gpu_usd = billable_remote_seconds * FAST_GPU_USD_PER_SECOND
    if estimated_gpu_usd > maximum_gpu_usd + 1e-9:
        raise RuntimeError(
            f"preview scene exceeded its ${maximum_gpu_usd:.6f} call cap: "
            f"${estimated_gpu_usd:.6f} estimated"
        )
    _atomic_write_bytes(master_path, result["master"])
    now = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": "1.0",
        "provider": PROVIDER_NAME,
        "scene_id": scene_id,
        "created_at": now,
        "updated_at": now,
        "request": {
            "prompt": prompt,
            "seed": seed,
            "width": width,
            "height": height,
        },
        "policy": {
            "finite_calls_only": True,
            "persistent_endpoint": False,
            "provisional_preview_only": True,
            "source_text_allowed": False,
            "prewarm_first": prewarm_first,
            "budget_plan": plan_file,
            "budget_ledger": ledger_path,
        },
        "stages": {
            "prewarm": {
                "enabled": prewarm_first,
                "remote_seconds": prewarm_remote_seconds,
                "model_load_seconds": prewarm_receipt.get("model_load_seconds", 0),
                "inference_warmup_seconds": prewarm_receipt.get(
                    "inference_warmup_seconds", 0
                ),
            },
            "preview": {
                "model": FAST_MODEL,
                "model_revision": FAST_MODEL_REVISION,
                "gpu": FAST_GPU,
                "finite_call": True,
                "hard_timeout_seconds": FAST_TIMEOUT_SECONDS,
                "remote_seconds": remote_seconds,
                "inference_seconds": result["image_seconds"],
                "provider_overhead_seconds": max(0.0, remote_seconds - result["image_seconds"]),
                "image_seconds": result["image_seconds"],
                "packaging_seconds": result["packaging_seconds"],
                "master_jpeg_quality": result["master_jpeg_quality"],
                "negative_prompt_supported": False,
                "model_load_seconds": result.get("model_load_seconds", 0),
                "container_age_seconds": result.get("container_age_seconds", 0),
                "estimated_gpu_usd": estimated_gpu_usd,
                "steps": 1,
                "guidance_scale": guidance_scale,
            }
        },
        "artifacts": {
            "preview": _artifact_record(
                path=master_path,
                root=destination,
                mime_type="image/jpeg",
                width=width,
                height=height,
            )
        },
    }
    _atomic_write_json(manifest_path, payload)
    _record_budget(
        plan_file=plan_file,
        ledger_path=ledger_path,
        reservation_id=reservation_id,
        experiment_id=experiment_id,
        stage="preview-scene",
        model=FAST_MODEL,
        revision=FAST_MODEL_REVISION,
        seed=seed,
        prompt=prompt,
        artifact_path=master_path,
        sha256=payload["artifacts"]["preview"]["sha256"],
        remote_seconds=billable_remote_seconds,
        width=width,
        height=height,
        frames=1,
        fps=0,
        gpu=FAST_GPU,
        gpu_usd_per_second=FAST_GPU_USD_PER_SECOND,
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
        gpu=MOTION_GPU,
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
    estimated_gpu_usd = remote_seconds * MOTION_GPU_USD_PER_SECOND
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
        "gpu": MOTION_GPU,
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
        gpu=MOTION_GPU,
        gpu_usd_per_second=MOTION_GPU_USD_PER_SECOND,
    )
    print(json.dumps(payload, sort_keys=True))
