"""Modal GPU scene generation for Storylight.

Run a real generation from the repository root with:

    modal run deploy/modal_scene_foundry.py::generate_cli \
      --prompt "A luminous storybook forest" \
      --output-path /tmp/master.png \
      --depth-output-path /tmp/depth.png
"""

from __future__ import annotations

import io
from pathlib import Path

import modal

APP_NAME = "storylight-scene-foundry"
CACHE_DIR = "/cache"
MINUTES = 60
IMAGE_MODEL = "stabilityai/sdxl-turbo"
IMAGE_REVISION = "71153311d3dbb46851df1931d3ca6e939de83304"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REVISION = "b4769fd619394250528294b658587285526fab1c"

runtime_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "huggingface-hub==0.36.0",
        "Pillow==11.1.0",
        "safetensors==0.4.5",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.47.1",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_CACHE": CACHE_DIR})
)

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("storylight-model-cache", create_if_missing=True)

with runtime_image.imports():
    import torch
    from diffusers import AutoPipelineForText2Image
    from PIL import Image
    from transformers import pipeline


@app.cls(
    image=runtime_image,
    gpu="T4",
    timeout=20 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={CACHE_DIR: model_cache},
)
class SceneModel:
    @modal.enter()
    def load(self) -> None:
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL,
            revision=IMAGE_REVISION,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")
        self.depth_pipe = pipeline(
            task="depth-estimation",
            model=DEPTH_MODEL,
            revision=DEPTH_REVISION,
            device=0,
        )

    @modal.method()
    def generate_scene(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
    ) -> tuple[bytes, bytes]:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        master = self.image_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        depth = self.depth_pipe(master)["depth"].convert("L").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        master_buffer = io.BytesIO()
        depth_buffer = io.BytesIO()
        master.save(master_buffer, format="PNG", optimize=True)
        depth.save(depth_buffer, format="PNG", optimize=True)
        torch.cuda.empty_cache()
        return master_buffer.getvalue(), depth_buffer.getvalue()


@app.local_entrypoint()
def generate_cli(
    prompt: str,
    output_path: str,
    depth_output_path: str,
    negative_prompt: str = "text, labels, watermark, interface, frame, collage",
    seed: int = 42,
    width: int = 1024,
    height: int = 576,
    steps: int = 4,
) -> None:
    master_bytes, depth_bytes = SceneModel().generate_scene.remote(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        steps=steps,
    )
    master_path = Path(output_path)
    depth_path = Path(depth_output_path)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_bytes(master_bytes)
    depth_path.write_bytes(depth_bytes)
