"""Finite L4 comparison of eager and compiled Klein, including depth and encoding.

Uses the pinned renderer-comparison image. No deployment, retries, or live stories.
"""

import hashlib
import io
import json
import time
from pathlib import Path

import modal
from modal_compare import MODELS, PROMPTS, STYLE, SUFFIX
from modal_compare import image as base_image

app = modal.App("bookforge-klein-compile-comparison")
image = base_image.add_local_file(
    Path(__file__).with_name("modal_compare.py"), "/root/modal_compare.py"
)
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REVISION = "b4769fd619394250528294b658587285526fab1c"


@app.function(
    image=image,
    gpu="L4",
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    cpu=8,
    memory=65536,
)
def compare():
    import diffusers
    import torch
    from transformers import pipeline

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "diffusers": diffusers.__version__,
        "automatic_retries": 0,
        "width": 1024,
        "height": 576,
        "max_sequence_length": 128,
        "compile": {"mode": "reduce-overhead", "fullgraph": True},
        "depth_model": DEPTH_MODEL,
        "depth_revision": DEPTH_REVISION,
        "samples": [],
        "warmups": [],
    }
    assets = {}
    model_id, revision, steps, guidance = MODELS["klein"]
    report.update(model=model_id, revision=revision, steps=steps, guidance=guidance)
    started = time.perf_counter()
    pipe = diffusers.Flux2KleinPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    depth_pipe = pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
        revision=DEPTH_REVISION,
        dtype=torch.float16,
        device=0,
    )
    torch.cuda.synchronize()
    report["download_and_load_seconds"] = time.perf_counter() - started
    eager = pipe.transformer
    compiled = torch.compile(eager, mode="reduce-overhead", fullgraph=True)

    def render(profile, prompt, seed):
        pipe.transformer = eager if profile == "eager" else compiled
        text = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_count = len(pipe.tokenizer(text)["input_ids"])
        if token_count > 128:
            raise ValueError("prompt would be truncated")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            master = pipe(
                prompt=prompt,
                width=1024,
                height=576,
                num_inference_steps=steps,
                guidance_scale=guidance,
                max_sequence_length=128,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize()
            image_seconds = time.perf_counter() - started
            depth_started = time.perf_counter()
            depth = depth_pipe(master)["depth"].convert("L").resize(master.size)
            torch.cuda.synchronize()
            depth_seconds = time.perf_counter() - depth_started
        encoded_started = time.perf_counter()
        image_buffer, depth_buffer = io.BytesIO(), io.BytesIO()
        master.convert("RGB").save(image_buffer, format="JPEG", quality=95, subsampling=0)
        depth.save(depth_buffer, format="JPEG", quality=85)
        encoded_seconds = time.perf_counter() - encoded_started
        master_bytes, depth_bytes = image_buffer.getvalue(), depth_buffer.getvalue()
        return (
            {
                "profile": profile,
                "seed": seed,
                "prompt": prompt,
                "token_count": token_count,
                "image_seconds": image_seconds,
                "depth_seconds": depth_seconds,
                "encoding_seconds": encoded_seconds,
                "total_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "master_sha256": hashlib.sha256(master_bytes).hexdigest(),
                "depth_sha256": hashlib.sha256(depth_bytes).hexdigest(),
            },
            master_bytes,
            depth_bytes,
        )

    try:
        for profile in ("eager", "compiled"):
            row, _, _ = render(profile, "A small amber lantern beside a quiet river.", 17)
            report["warmups"].append(row)
            print(json.dumps({"warmup": row}), flush=True)
        for index, brief in enumerate(PROMPTS):
            profiles = ("eager", "compiled") if index % 2 == 0 else ("compiled", "eager")
            for profile in profiles:
                row, master_bytes, depth_bytes = render(
                    profile, STYLE + brief + SUFFIX, 20260903 + index
                )
                stem = f"{profile}-{index}"
                assets[f"{stem}.jpg"] = master_bytes
                assets[f"{stem}-depth.jpg"] = depth_bytes
                row.update(case=index, master_file=f"{stem}.jpg", depth_file=f"{stem}-depth.jpg")
                report["samples"].append(row)
                print(json.dumps(row), flush=True)
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "detail": str(error)[:2000]}
        print(json.dumps(report["failure"]), flush=True)
    return report, assets


@app.local_entrypoint()
def main(output_dir: str):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    call = compare.spawn()
    print(f"Recoverable function call: {call.object_id}", flush=True)
    report, assets = call.get()
    report.update(
        function_call_id=call.object_id,
        whole_call_seconds=time.perf_counter() - started,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    for filename, content in assets.items():
        (destination / filename).write_bytes(content)
    (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output_dir": str(destination), "samples": len(report["samples"])}))
