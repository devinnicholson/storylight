"""Opt-in Klein candidate: authenticated SDK only, one L4, bounded calls, idle scale-to-zero."""

import sys
import time
from pathlib import Path

import modal

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
from klein_scene_runtime import KleinSceneRuntime  # noqa: E402

# Reuse the exact already-qualified weight image, including its CPU-only bake layer.
EXPERIMENTS = DEPLOY.parent / "experiments" / "renderer-fidelity"
if modal.is_local():
    sys.path.insert(0, str(EXPERIMENTS))
    from klein_restart import image
else:
    image = None

app = modal.App("storylight-klein-candidate")
cache = modal.Volume.from_name("storylight-klein-compile-cache-v1")
CACHE_ID = "f305950a0fbb4ecf89acfb80a3990351"


@app.cls(
    image=image,
    gpu="L4",
    cpu=(8, 8),
    memory=(65536, 65536),
    timeout=180,
    startup_timeout=120,
    retries=0,
    min_containers=0,
    max_containers=1,
    scaledown_window=90,
    volumes={"/compiled": cache},
)
@modal.concurrent(max_inputs=1)
class KleinSceneStudio:
    @modal.enter()
    def load(self):
        started = time.perf_counter()
        self.runtime = KleinSceneRuntime(Path("/models"))
        self.cache_seconds = self.runtime.compile(Path("/compiled") / CACHE_ID)
        self.startup_seconds = time.perf_counter() - started
        self.renders = 0

    @modal.method()
    def prewarm(self):
        started = time.perf_counter()
        # These fixed synthetic inputs cover the two accepted production token buckets.
        for prompt in (
            "A watercolor boat on a lake.",
            "Watercolor illustration. " + "A small boat floats beside reeds. " * 20,
        ):
            self.runtime.render(prompt, 0)
            self.renders += 1
        return {
            "model_load_seconds": self.runtime.load_seconds,
            "startup_seconds": self.startup_seconds,
            "cache_setup_seconds": self.cache_seconds,
            "warmup_seconds": time.perf_counter() - started,
        }

    @modal.method()
    def generate(self, prompt: str, seed: int):
        text = self.runtime.pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(self.runtime.pipe.tokenizer(text)["input_ids"]) > 256:
            raise ValueError(
                "Klein preview accepts at most 256 tokens; not truncating an unqualified shape"
            )
        metrics, master, depth = self.runtime.render(prompt, seed)
        warm = self.renders > 0
        self.renders += 1
        return {
            "metrics": metrics,
            "master": master,
            "depth": depth,
            "identity": self.runtime.identity,
            "warm_state": "warm" if warm else "cold",
            "model_load_seconds": self.runtime.load_seconds,
            "startup_seconds": self.startup_seconds,
            "cache_setup_seconds": self.cache_seconds,
        }
