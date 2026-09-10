"""Finite same-GPU renderer comparison; never deploys or changes production.

Run with ``modal run experiments/renderer-fidelity/modal_compare.py --output-dir ...``.
One L4 call, at most 900 seconds, no retries, fixed synthetic prompts only.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

MODELS = {
    "sana": (
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        "19683c58b7ea290e55cedd8950ae1d86ada7ef96",
        2,
        4.5,
    ),
    "klein": (
        "black-forest-labs/FLUX.2-klein-4B",
        "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        4,
        1.0,
    ),
}
PROMPTS = [
    "A single golden paper boat floats on a calm indigo pond beneath a crescent moon.",
    "One silver fox stands to the left of a glowing golden lantern among tall cedar trees.",
    "A single owl perches on a tree branch above a calm sea. "
    "A lighthouse is far to the left of the owl.",
    "One red fox carries a small golden lantern in its mouth while walking through a snowy forest.",
    "Exactly two red paper boats float side by side on a calm blue pond.",
    "One child stands on a wooden bridge holding an open green book in both hands. "
    "No other people.",
]
STYLE = "Luminous watercolor paper theater. "
SUFFIX = " Full-bleed illustration, no text."
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "diffusers==0.39.0",
        "transformers==4.57.1",
        "accelerate==1.10.1",
        "huggingface-hub==0.36.0",
        "safetensors==0.8.0",
        "Pillow==11.1.0",
        "sentencepiece==0.2.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
app = modal.App("storylight-renderer-fidelity-comparison")


@app.function(
    image=image,
    gpu="L4",
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    cpu=4,
    memory=65536,
)
def compare(text_length_experiment: bool = False):
    import gc

    import diffusers
    import torch

    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "diffusers": diffusers.__version__,
        "width": 1024,
        "height": 576,
        "automatic_retries": 0,
        "samples": [],
        "loads": [],
        "experiment": "text_length" if text_length_experiment else "model_comparison",
    }
    assets = {}
    for name, (model_id, revision, steps, guidance) in MODELS.items():
        if text_length_experiment and name != "klein":
            continue
        started = time.perf_counter()
        pipeline_class = (
            diffusers.SanaSprintPipeline if name == "sana" else diffusers.Flux2KleinPipeline
        )
        pipe = pipeline_class.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        pipe.set_progress_bar_config(disable=True)
        torch.cuda.synchronize()
        report["loads"].append(
            {
                "model": name,
                "id": model_id,
                "revision": revision,
                "download_and_load_seconds": time.perf_counter() - started,
            }
        )
        cases = [
            (index, repeat, length)
            for repeat in range(2 if text_length_experiment else 1)
            for index in range(len(PROMPTS))
            for length in (
                ([512, 128] if (index + repeat) % 2 == 0 else [128, 512])
                if text_length_experiment
                else [512]
            )
        ]
        for index, repeat, length in cases:
            seed = 20260903 + index + 100 * repeat
            brief = PROMPTS[index]
            prompt = STYLE + brief + SUFFIX
            options = {}
            token_count = None
            if name == "klein":
                text = pipe.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                token_count = len(pipe.tokenizer(text)["input_ids"])
                if token_count > length:
                    raise ValueError("sequence experiment would truncate a prompt")
                options["max_sequence_length"] = length
            started = time.perf_counter()
            with torch.inference_mode():
                master = pipe(
                    prompt=prompt,
                    width=1024,
                    height=576,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=torch.Generator("cuda").manual_seed(seed),
                    **options,
                ).images[0]
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - started
            buffer = io.BytesIO()
            master.convert("RGB").save(buffer, format="JPEG", quality=95, subsampling=0)
            content = buffer.getvalue()
            filename = (
                f"{name}-{index}-r{repeat}-seq{length}.jpg"
                if text_length_experiment
                else f"{name}-{index}.jpg"
            )
            assets[filename] = content
            row = {
                "model": name,
                "prompt": prompt,
                "seed": seed,
                "steps": steps,
                "guidance_scale": guidance,
                "inference_seconds": inference_seconds,
                "first_inference": index == 0 and repeat == 0,
                "max_sequence_length": length if name == "klein" else None,
                "token_count": token_count,
                "repeat": repeat,
                "filename": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            report["samples"].append(row)
            print(json.dumps(row), flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    return report, assets


@app.local_entrypoint()
def main(output_dir: str, text_length_experiment: bool = False):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    call = compare.spawn(text_length_experiment)
    print(f"Recoverable function call: {call.object_id}", flush=True)
    report, assets = call.get()
    report["function_call_id"] = call.object_id
    report["whole_call_seconds"] = time.perf_counter() - started
    report["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for filename, content in assets.items():
        (destination / filename).write_bytes(content)
    (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output_dir": str(destination), "samples": len(report["samples"])}))
