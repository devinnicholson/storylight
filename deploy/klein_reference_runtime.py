"""Isolated single-reference Klein path; existing compiler cache is not reference qualification."""

from __future__ import annotations

import hashlib
import io
import re
import time
from pathlib import Path

from klein_scene_runtime import PROFILE, KleinSceneRuntime, sequence_bucket
from PIL import Image, UnidentifiedImageError

MAX_REFERENCE_BYTES = 2_000_000
REFERENCE_SIZE = (1024, 576)


def _reference_image(content: bytes, expected_sha256: str):
    if not isinstance(content, bytes) or not 0 < len(content) <= MAX_REFERENCE_BYTES:
        raise ValueError("reference must be JPEG bytes within the size limit")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is None
        or hashlib.sha256(content).hexdigest() != expected_sha256
    ):
        raise ValueError("reference checksum differs")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "JPEG" or image.mode != "RGB" or image.size != REFERENCE_SIZE:
                raise ValueError("reference must be an RGB JPEG at 1024 by 576")
            image.load()
            # Drop metadata; conditioning uses decoded pixels only.
            return Image.frombytes("RGB", image.size, image.tobytes())
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("reference JPEG is invalid") from None


class KleinReferenceRuntime(KleinSceneRuntime):
    def render(
        self,
        prompt: str,
        seed: int,
        *,
        reference_jpeg: bytes | None = None,
        reference_sha256: str | None = None,
    ) -> tuple[dict, bytes, bytes]:
        if reference_jpeg is None:
            if reference_sha256 is not None:
                raise ValueError("reference checksum requires an image")
            metrics, master, depth = super().render(prompt, seed)
        else:
            reference = _reference_image(reference_jpeg, reference_sha256)
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4_000:
                raise ValueError("prompt requires 1–4000 characters")
            if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
                raise ValueError("seed must be an unsigned 32-bit integer")
            metrics, master, depth = self._render_reference(prompt, seed, reference)
        return (
            {
                **metrics,
                "reference_sha256": reference_sha256,
                "reference_runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reference_conditioned": reference_jpeg is not None,
                "reference_width": REFERENCE_SIZE[0] if reference_jpeg is not None else None,
                "reference_height": REFERENCE_SIZE[1] if reference_jpeg is not None else None,
            },
            master,
            depth,
        )

    def _render_reference(self, prompt: str, seed: int, reference) -> tuple[dict, bytes, bytes]:
        import torch

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
                image=reference,
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
