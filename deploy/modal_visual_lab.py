"""Finite Modal visual R&D jobs for Bookforge.

This module intentionally exposes only ``modal run`` entrypoints. It does not
deploy a persistent endpoint, and each GPU class has a short idle window plus a
hard timeout.

Generate projection master candidates:

    modal run deploy/modal_visual_lab.py::master_batch_cli \
      --prompt-file experiments/visual-lab/moon-gate-prompts.json \
      --output-dir artifacts/visual-lab/masters

Animate one selected master into subtle ping-pong loops:

    modal run deploy/modal_visual_lab.py::motion_batch_cli \
      --image-path artifacts/visual-lab/masters/moon-gate-theater__20260822.png \
      --prompt-file experiments/visual-lab/moon-gate-motion-prompts.json \
      --output-dir artifacts/visual-lab/motion
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "bookforge-visual-lab"
CACHE_DIR = "/cache"
MINUTES = 60
SANA_GPU = "L4"
MOTION_GPU = "L4"
SANA_TIMEOUT_SECONDS = 3 * 60
MOTION_TIMEOUT_SECONDS = 4 * 60
SANA_MODEL = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
SANA_REVISION = "caa51e5ea874be07d3a9c7c2d0fd800570b18440"
LTX_MODEL = "Lightricks/LTX-Video"
LTX_REVISION = "a6d59ee37c13c58261aa79027d3e41cd41960925"
GPU_USD_PER_SECOND = {"L4": 0.000222}
DEFAULT_NEGATIVE = (
    "words, letters, captions, logo, watermark, interface, border, split screen, "
    "collage, photorealistic, plastic 3d render, clutter, muddy shadows, low contrast"
)
DEFAULT_MOTION_NEGATIVE = (
    "camera shake, fast motion, hard cut, scene change, zoom, morphing, melting, "
    "flicker, jitter, inconsistent character, blurry, distorted, text, watermark"
)

runtime_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1")
    .uv_pip_install(
        "accelerate==1.4.0",
        "beautifulsoup4==4.13.3",
        "diffusers==0.39.0",
        "huggingface-hub==0.36.0",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "opencv-python==4.11.0.86",
        "Pillow==11.1.0",
        "protobuf==5.29.3",
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


def _validate_dimensions(width: int, height: int) -> None:
    if not (512 <= width <= 1536 and 512 <= height <= 1536):
        raise ValueError("width and height must be between 512 and 1536")
    if width % 32 or height % 32:
        raise ValueError("width and height must be divisible by 32")


def _safe_slug(value: str) -> str:
    slug = "-".join("".join(char.lower() if char.isalnum() else " " for char in value).split())
    return slug[:72] or "scene"


@app.cls(
    image=runtime_image,
    gpu=SANA_GPU,
    timeout=SANA_TIMEOUT_SECONDS,
    scaledown_window=30,
    volumes={CACHE_DIR: model_cache},
)
class SanaStudio:
    @modal.enter()
    def load(self) -> None:
        self.pipe = diffusers.SanaPipeline.from_pretrained(
            SANA_MODEL,
            revision=SANA_REVISION,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.pipe.vae.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)

    @modal.method()
    def generate_batch(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not 1 <= len(jobs) <= 24:
            raise ValueError("a master batch must contain between 1 and 24 jobs")
        outputs: list[dict[str, Any]] = []
        for job in jobs:
            width = int(job.get("width", 1024))
            height = int(job.get("height", 576))
            _validate_dimensions(width, height)
            seed = int(job["seed"])
            started = time.perf_counter()
            image = self.pipe(
                prompt=str(job["prompt"]),
                negative_prompt=str(job.get("negative_prompt", DEFAULT_NEGATIVE)),
                width=width,
                height=height,
                guidance_scale=float(job.get("guidance_scale", 4.5)),
                num_inference_steps=int(job.get("steps", 20)),
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            content = buffer.getvalue()
            outputs.append(
                {
                    "id": str(job["id"]),
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "generation_seconds": time.perf_counter() - started,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "content": content,
                }
            )
        torch.cuda.empty_cache()
        return outputs


@app.cls(
    image=runtime_image,
    gpu=MOTION_GPU,
    timeout=MOTION_TIMEOUT_SECONDS,
    scaledown_window=30,
    volumes={CACHE_DIR: model_cache},
)
class MotionStudio:
    @modal.enter()
    def load(self) -> None:
        self.pipe = diffusers.LTXImageToVideoPipeline.from_pretrained(
            LTX_MODEL,
            revision=LTX_REVISION,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    @modal.method()
    def generate_batch(
        self,
        image_bytes: bytes,
        jobs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not 1 <= len(jobs) <= 12:
            raise ValueError("a motion batch must contain between 1 and 12 jobs")
        source = diffusers.utils.load_image(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        outputs: list[dict[str, Any]] = []
        for job in jobs:
            width = int(job.get("width", 768))
            height = int(job.get("height", 512))
            _validate_dimensions(width, height)
            seed = int(job["seed"])
            num_frames = int(job.get("frames", 49))
            fps = int(job.get("fps", 24))
            if not 9 <= num_frames <= 97 or (num_frames - 1) % 8:
                raise ValueError("frames must be 8n+1 and between 9 and 97")
            if not 12 <= fps <= 30:
                raise ValueError("fps must be between 12 and 30")
            started = time.perf_counter()
            result = self.pipe(
                image=source,
                prompt=str(job["prompt"]),
                negative_prompt=str(job.get("negative_prompt", DEFAULT_MOTION_NEGATIVE)),
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=int(job.get("steps", 40)),
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).frames[0]
            # Forward then backward makes the endpoint identical and prevents the
            # visible jump that ruins projected ambient loops.
            loop_frames = result + list(reversed(result[1:-1]))
            output_path = Path("/tmp") / f"{_safe_slug(str(job['id']))}-{seed}.mp4"
            diffusers.utils.export_to_video(loop_frames, output_path, fps=fps)
            content = output_path.read_bytes()
            output_path.unlink(missing_ok=True)
            outputs.append(
                {
                    "id": str(job["id"]),
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "frames": len(loop_frames),
                    "fps": fps,
                    "generation_seconds": time.perf_counter() - started,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "content": content,
                }
            )
        torch.cuda.empty_cache()
        return outputs


def _load_jobs(prompt_file: str, *, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(Path(prompt_file).read_text())
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("prompt file requires a non-empty jobs array")
    if len(jobs) > limit:
        raise ValueError(f"prompt file exceeds the {limit}-job hard limit")
    required = {"id", "prompt", "seed"}
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or not required.issubset(job):
            raise ValueError(f"job {index} requires id, prompt, and seed")
    identifiers = [str(job["id"]) for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("job IDs must be unique")
    return jobs


def _write_results(
    *,
    jobs: list[dict[str, Any]],
    results: list[dict[str, Any]],
    output_dir: str,
    suffix: str,
    stage: str,
    model: str,
    revision: str,
    gpu: str,
    remote_seconds: float,
) -> list[dict[str, Any]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    by_id = {str(job["id"]): job for job in jobs}
    allocated_seconds = remote_seconds / len(results)
    records: list[dict[str, Any]] = []
    for result in results:
        job = by_id[result["id"]]
        filename = f"{_safe_slug(result['id'])}__{result['seed']}{suffix}"
        path = destination / filename
        path.write_bytes(result.pop("content"))
        records.append(
            {
                "experiment_id": result["id"],
                "stage": stage,
                "model": model,
                "model_revision": revision,
                "gpu": gpu,
                "seed": result["seed"],
                "prompt": str(job["prompt"]),
                "artifact_path": str(path),
                "sha256": result["sha256"],
                "generation_seconds": result["generation_seconds"],
                "estimated_gpu_usd": allocated_seconds * GPU_USD_PER_SECOND[gpu],
                "width": result["width"],
                "height": result["height"],
                "frames": result.get("frames", 1),
                "fps": result.get("fps", 0),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "stage": stage,
        "model": model,
        "model_revision": revision,
        "gpu": gpu,
        "remote_seconds": remote_seconds,
        "estimated_gpu_usd": remote_seconds * GPU_USD_PER_SECOND[gpu],
        "records": records,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return records


def _open_budget(plan_file: str, ledger_path: str):
    from bookforge.visual_lab import BudgetEnvelope, VisualLabLedger

    plan = json.loads(Path(plan_file).read_text())
    envelope = BudgetEnvelope(**plan["budget"])
    ledger = Path(ledger_path)
    if ledger.exists():
        return VisualLabLedger.read(ledger, envelope=envelope)
    return VisualLabLedger(
        envelope=envelope,
        prior_estimated_usd=float(plan.get("ledger_baseline_usd", 0)),
    )


def _record_budget(ledger, ledger_path: str, records: list[dict[str, Any]]) -> None:
    from bookforge.visual_lab import GenerationRecord

    for record in records:
        ledger.add(
            GenerationRecord(
                **{
                    **record,
                    "experiment_id": f"{record['stage']}:{record['experiment_id']}",
                }
            )
        )
    ledger.write(Path(ledger_path))


def _reserve_budget(
    ledger,
    *,
    ledger_path: str,
    stage: str,
    prompt_file: str,
    gpu: str,
    timeout_seconds: int,
) -> str:
    source = Path(prompt_file).read_bytes()
    reservation_id = f"{stage}:{hashlib.sha256(source).hexdigest()[:16]}"
    ledger.reserve(
        gpu=gpu,
        maximum_seconds=timeout_seconds,
        reservation_id=reservation_id,
    )
    ledger.write(Path(ledger_path))
    return reservation_id


def _reject_recorded_jobs(ledger, *, stage: str, jobs: list[dict[str, Any]]) -> None:
    recorded = {record.experiment_id for record in ledger.records}
    duplicates = [str(job["id"]) for job in jobs if f"{stage}:{job['id']}" in recorded]
    if duplicates:
        raise ValueError(f"experiment IDs already recorded: {', '.join(duplicates)}")


@app.local_entrypoint()
def master_batch_cli(
    prompt_file: str,
    output_dir: str,
    plan_file: str = "experiments/visual-lab/plan.json",
    ledger_path: str = "artifacts/visual-lab/ledger.json",
) -> None:
    jobs = _load_jobs(prompt_file, limit=24)
    ledger = _open_budget(plan_file, ledger_path)
    _reject_recorded_jobs(ledger, stage="master", jobs=jobs)
    reservation_id = _reserve_budget(
        ledger,
        ledger_path=ledger_path,
        stage="master",
        prompt_file=prompt_file,
        gpu=SANA_GPU,
        timeout_seconds=SANA_TIMEOUT_SECONDS,
    )
    started = time.perf_counter()
    results = SanaStudio().generate_batch.remote(jobs)
    records = _write_results(
        jobs=jobs,
        results=results,
        output_dir=output_dir,
        suffix=".png",
        stage="master",
        model=SANA_MODEL,
        revision=SANA_REVISION,
        gpu=SANA_GPU,
        remote_seconds=time.perf_counter() - started,
    )
    ledger.release(reservation_id)
    _record_budget(ledger, ledger_path, records)


@app.local_entrypoint()
def motion_batch_cli(
    image_path: str,
    prompt_file: str,
    output_dir: str,
    plan_file: str = "experiments/visual-lab/plan.json",
    ledger_path: str = "artifacts/visual-lab/ledger.json",
) -> None:
    source_path = Path(image_path)
    if not source_path.is_file():
        raise ValueError(f"local image does not exist: {source_path}")
    jobs = _load_jobs(prompt_file, limit=12)
    ledger = _open_budget(plan_file, ledger_path)
    _reject_recorded_jobs(ledger, stage="motion", jobs=jobs)
    reservation_id = _reserve_budget(
        ledger,
        ledger_path=ledger_path,
        stage="motion",
        prompt_file=prompt_file,
        gpu=MOTION_GPU,
        timeout_seconds=MOTION_TIMEOUT_SECONDS,
    )
    started = time.perf_counter()
    results = MotionStudio().generate_batch.remote(source_path.read_bytes(), jobs)
    records = _write_results(
        jobs=jobs,
        results=results,
        output_dir=output_dir,
        suffix=".mp4",
        stage="motion",
        model=LTX_MODEL,
        revision=LTX_REVISION,
        gpu=MOTION_GPU,
        remote_seconds=time.perf_counter() - started,
    )
    ledger.release(reservation_id)
    _record_budget(ledger, ledger_path, records)
