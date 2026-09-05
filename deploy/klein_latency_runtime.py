"""Stage timing around the unchanged Klein image and depth operations."""

from __future__ import annotations

import hashlib
import io
import math
import time
from pathlib import Path

from klein_scene_runtime import PROFILE, KleinSceneRuntime, sequence_bucket


class LatencySceneRuntime(KleinSceneRuntime):
    def __init__(self, model_root: Path):
        super().__init__(model_root)
        self.instrumentation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self._completed_buckets: set[int] = set()

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
        if bucket not in (128, 256):
            raise ValueError("token bucket has not been qualified for the latency experiment")
        tokenization_seconds = time.perf_counter() - started
        bucket_was_warm = bucket in self._completed_buckets
        torch.cuda.reset_peak_memory_stats()
        cuda_image_seconds = None
        with torch.inference_mode():
            pipeline_started = time.perf_counter()
            try:
                first = torch.cuda.Event(enable_timing=True)
                last = torch.cuda.Event(enable_timing=True)
                first.record()
                events = first, last
            except (AttributeError, RuntimeError, NotImplementedError):
                events = None
            master = self.pipe(
                prompt=prompt,
                width=PROFILE["width"],
                height=PROFILE["height"],
                num_inference_steps=PROFILE["steps"],
                guidance_scale=PROFILE["guidance"],
                max_sequence_length=bucket,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            if events is not None:
                try:
                    events[1].record()
                except (RuntimeError, NotImplementedError):
                    events = None
            torch.cuda.synchronize()
            image_finished = time.perf_counter()
            pipeline_seconds = image_finished - pipeline_started
            image_seconds = image_finished - started
            if events is not None:
                try:
                    duration = events[0].elapsed_time(events[1]) / 1000
                    if math.isfinite(duration) and duration >= 0:
                        cuda_image_seconds = duration
                except (RuntimeError, NotImplementedError):
                    pass
            depth_started = time.perf_counter()
            depth = self.depth_pipe(master)["depth"].convert("L").resize(master.size)
            torch.cuda.synchronize()
            depth_seconds = time.perf_counter() - depth_started
        encoding_started = time.perf_counter()
        master_buffer, depth_buffer = io.BytesIO(), io.BytesIO()
        master.convert("RGB").save(master_buffer, format="JPEG", quality=95, subsampling=0)
        depth.save(depth_buffer, format="JPEG", quality=85)
        master_bytes, depth_bytes = master_buffer.getvalue(), depth_buffer.getvalue()
        metrics = {
            "seed": seed,
            "token_count": token_count,
            "sequence_bucket": bucket,
            "bucket_was_warm": bucket_was_warm,
            "tokenization_seconds": tokenization_seconds,
            "pipeline_seconds": pipeline_seconds,
            "cuda_image_seconds": cuda_image_seconds,
            "image_seconds": image_seconds,
            "depth_seconds": depth_seconds,
            "encoding_seconds": time.perf_counter() - encoding_started,
            "total_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "master_sha256": hashlib.sha256(master_bytes).hexdigest(),
            "depth_sha256": hashlib.sha256(depth_bytes).hexdigest(),
            "instrumentation_sha256": self.instrumentation_sha256,
        }
        self._completed_buckets.add(bucket)
        return metrics, master_bytes, depth_bytes

    def warmup(self) -> dict:
        started = time.perf_counter()
        reports = []
        for prompt, expected_bucket in (
            ("A watercolor boat on a lake.", 128),
            ("Watercolor illustration. " + "A small boat floats beside reeds. " * 20, 256),
        ):
            metrics, _, _ = self.render(prompt, 0)
            if metrics["sequence_bucket"] != expected_bucket:
                raise ValueError("warmup did not exercise the qualified token bucket")
            reports.append(metrics)
        return {
            "warmup_seconds": time.perf_counter() - started,
            "renders": reports,
            "instrumentation_sha256": self.instrumentation_sha256,
        }
