"""Bounded compiled-Klein snapshot probe; never changes live renderer routing.

Deploy this isolated app, then run this file with --app-id and --output-dir.
The local runner stops that exact app even when the measurement fails.
"""

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import modal
from modal_compare import MODELS, PROMPTS, STYLE, SUFFIX
from modal_compare import image as base_image

APP_NAME = "bookforge-klein-compiled-snapshot-probe"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REVISION = "b4769fd619394250528294b658587285526fab1c"
image = base_image.env({"TORCHINDUCTOR_COMPILE_THREADS": "1"}).add_local_file(
    Path(__file__).with_name("modal_compare.py"), "/root/modal_compare.py"
)
app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="L4",
    cpu=8,
    memory=65536,
    timeout=60,
    startup_timeout=420,
    retries=0,
    min_containers=0,
    max_containers=1,
    scaledown_window=5,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class CompiledKlein:
    @modal.enter(snap=True)
    def load(self):
        import diffusers
        import torch
        from transformers import pipeline

        started = time.perf_counter()
        model, revision, self.steps, self.guidance = MODELS["klein"]
        self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(
            model, revision=revision, torch_dtype=torch.bfloat16
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.depth_pipe = pipeline(
            "depth-estimation",
            model=DEPTH_MODEL,
            revision=DEPTH_REVISION,
            dtype=torch.float16,
            device=0,
        )
        self.pipe.transformer = torch.compile(
            self.pipe.transformer, mode="reduce-overhead", fullgraph=True
        )
        self.initialization = {
            "snapshot_id": uuid.uuid4().hex,
            "load_seconds": time.perf_counter() - started,
            "model": model,
            "revision": revision,
            "depth_model": DEPTH_MODEL,
            "depth_revision": DEPTH_REVISION,
            "torch": str(torch.__version__),
            "diffusers": diffusers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "warmups": [],
        }
        for index in range(2):
            row, _, _ = self._render(0, 17 + index)
            self.initialization["warmups"].append(row)
            print(json.dumps({"snapshot_warmup": row}), flush=True)
        print(json.dumps({"snapshot_ready": self.initialization}), flush=True)

    @modal.enter(snap=False)
    def restore(self):
        self.restore_id = uuid.uuid4().hex
        self.request_count = 0

    def _render(self, case, seed):
        import torch

        prompt = STYLE + PROMPTS[case] + SUFFIX
        text = self.pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = len(self.pipe.tokenizer(text)["input_ids"])
        if tokens > 128:
            raise ValueError("snapshot probe would truncate a prompt")
        started = time.perf_counter()
        with torch.inference_mode():
            master = self.pipe(
                prompt=prompt,
                width=1024,
                height=576,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                max_sequence_length=128,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            torch.cuda.synchronize()
            image_seconds = time.perf_counter() - started
            depth_started = time.perf_counter()
            depth = self.depth_pipe(master)["depth"].convert("L").resize(master.size)
            torch.cuda.synchronize()
            depth_seconds = time.perf_counter() - depth_started
        master_buffer, depth_buffer = io.BytesIO(), io.BytesIO()
        master.convert("RGB").save(master_buffer, format="JPEG", quality=95, subsampling=0)
        depth.save(depth_buffer, format="JPEG", quality=85)
        master_bytes, depth_bytes = master_buffer.getvalue(), depth_buffer.getvalue()
        row = {
            "case": case,
            "seed": seed,
            "prompt": prompt,
            "token_count": tokens,
            "image_seconds": image_seconds,
            "depth_seconds": depth_seconds,
            "total_seconds": time.perf_counter() - started,
            "master_sha256": hashlib.sha256(master_bytes).hexdigest(),
            "depth_sha256": hashlib.sha256(depth_bytes).hexdigest(),
        }
        return row, master_bytes, depth_bytes

    @modal.method()
    def render(self, case: int, seed: int):
        row, master, depth = self._render(case, seed)
        self.request_count += 1
        row.update(
            restore_id=self.restore_id,
            request_count=self.request_count,
            initialization=self.initialization,
        )
        return row, master, depth


def run_probe(app_id: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "app_id": app_id,
        "app_name": APP_NAME,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "automatic_retries": 0,
        "samples": [],
        "scale_to_zero": [],
    }
    method = modal.Cls.from_name(APP_NAME, "CompiledKlein")().render
    started = time.perf_counter()
    try:
        for cycle in range(3):
            if cycle:
                idle_started = time.perf_counter()
                while True:
                    stats = method.get_current_stats()
                    if stats.num_total_runners == 0 and stats.backlog == 0:
                        report["scale_to_zero"].append(
                            {
                                "before_cycle": cycle,
                                "wait_seconds": time.perf_counter() - idle_started,
                                "num_total_runners": stats.num_total_runners,
                            }
                        )
                        break
                    if time.perf_counter() - idle_started > 90:
                        raise TimeoutError("renderer did not scale to zero within 90 seconds")
                    time.sleep(5)
            for request, case in enumerate((0, 5, 0)):
                remaining = 900 - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("snapshot probe exceeded its 900-second client bound")
                call_started = time.perf_counter()
                call = method.spawn(case, 20260903 + case)
                print(
                    json.dumps(
                        {"cycle": cycle, "request": request, "function_call_id": call.object_id}
                    ),
                    flush=True,
                )
                try:
                    row, master, depth = call.get(timeout=min(480, remaining))
                except Exception:
                    call.cancel(terminate_containers=True)
                    raise
                stem = f"cycle-{cycle}-request-{request}"
                row.update(
                    cycle=cycle,
                    request=request,
                    function_call_id=call.object_id,
                    client_seconds=time.perf_counter() - call_started,
                    master_file=f"{stem}.jpg",
                    depth_file=f"{stem}-depth.jpg",
                )
                (output_dir / row["master_file"]).write_bytes(master)
                (output_dir / row["depth_file"]).write_bytes(depth)
                report["samples"].append(row)
                (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "detail": str(error)[:2000]}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        stopped = subprocess.run(
            [sys.executable, "-m", "modal", "app", "stop", "--yes", app_id],
            timeout=30,
            check=False,
        )
        report["stop_exit_code"] = stopped.returncode
        (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_probe(args.app_id, args.output_dir)
