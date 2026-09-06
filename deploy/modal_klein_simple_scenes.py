"""Finite simple-scene comparison on one authenticated, non-snapshot Klein worker."""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import modal

DEPLOY = Path(__file__).resolve().parent
sys.path.insert(0, str(DEPLOY))
EXPERIMENT = "simple-scenes-20260905-a"
APP = "bookforge-klein-simple-scenes"
IMAGE_ID = "im-WtXer8GjRPdgMqWAAUSMwJ"
CACHE_ID = "f305950a0fbb4ecf89acfb80a3990351"
FIXTURE_SHA256 = "fc0a6ab8585aa7c71111fac50e02edd6787b6d436feb6393696429e4c8d29bba"
MANIFEST = Path("/root/simple-scenes-manifest.json")
FIXTURE = Path("/root/simple-scene-controls.json")

if modal.is_local():
    image = (
        modal.Image.from_id(IMAGE_ID)
        .add_local_file(DEPLOY / "klein_scene_runtime.py", "/root/klein_scene_runtime.py")
        .add_local_file(DEPLOY / "klein_reference_runtime.py", "/root/klein_reference_runtime.py")
        .add_local_file(
            DEPLOY.parent / "experiments/renderer-fidelity/simple-scene-controls-v1.json",
            str(FIXTURE),
        )
        .add_local_file(
            DEPLOY.parent / "benchmarks/simple-scenes-2026-09-05/manifest.json", str(MANIFEST)
        )
    )
else:
    image = None

app = modal.App(APP)
claims = modal.Dict.from_name(f"bookforge-{EXPERIMENT}-claims", create_if_missing=True)
cache = modal.Volume.from_name("bookforge-klein-compile-cache-v1")


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _expiry(manifest):
    expires = manifest.get("expires_at")
    if type(expires) is not int or not time.time() < expires <= time.time() + 7200:
        raise ValueError("simple-scene authorization expired")


def _read(path):
    with path.open("rb") as stream:
        raw = stream.read(32769)
    if len(raw) > 32768:
        raise ValueError("simple-scene input exceeds bound")
    return json.loads(raw), _sha(raw)


def _configuration():
    manifest, proof = _read(MANIFEST)
    fixture, fixture_sha = _read(FIXTURE)
    if not isinstance(manifest, dict):
        raise ValueError("simple-scene manifest differs")
    _expiry(manifest)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("experiment_id") != EXPERIMENT
        or manifest.get("status") != "authorized"
        or type(manifest.get("maximum_operations")) is not int
        or manifest["maximum_operations"] != 12
        or fixture_sha != FIXTURE_SHA256
        or manifest.get("fixture_sha256") != FIXTURE_SHA256
    ):
        raise ValueError("simple-scene manifest differs")
    for key, path in {
        "runtime_sha256": DEPLOY / "klein_scene_runtime.py",
        "reference_runtime_sha256": DEPLOY / "klein_reference_runtime.py",
        "deployment_sha256": Path(__file__),
    }.items():
        if manifest.get(key) != _sha(path.read_bytes()):
            raise ValueError("simple-scene source differs")
    expected = []
    for index, control in enumerate(fixture["controls"]):
        variants = (
            ("text_next", "reference_next") if index % 2 == 0 else ("reference_next", "text_next")
        )
        base_id = _sha(f"{EXPERIMENT}:{control['id']}:base".encode())[:32]
        for variant in ("base", *variants):
            prompt = control["before_prompt" if variant == "base" else "after_prompt"]
            expected.append(
                {
                    "ordinal": len(expected),
                    "request_id": _sha(f"{EXPERIMENT}:{control['id']}:{variant}".encode())[:32],
                    "control_id": control["id"],
                    "variant": variant,
                    "seed": control["seed"],
                    "prompt_sha256": _sha(prompt.encode()),
                    "reference_request_id": base_id if variant == "reference_next" else None,
                }
            )
    if json.dumps(manifest.get("operations"), sort_keys=True) != json.dumps(
        expected, sort_keys=True
    ):
        raise ValueError("simple-scene operations differ")
    if (
        os.environ.get("MODAL_CLOUD_PROVIDER")
        not in {"CLOUD_PROVIDER_AWS", "CLOUD_PROVIDER_GCP", "CLOUD_PROVIDER_OCI"}
        or not re.fullmatch(r"us-[a-z0-9-]{1,48}", os.environ.get("MODAL_REGION", ""))
        or not os.environ.get("MODAL_TASK_ID")
    ):
        raise ValueError("simple-scene placement differs")
    return manifest, fixture, proof


def _event(phase, state, **fields):
    print(
        json.dumps({"simple_scene_stage": {"phase": phase, "state": state, **fields}}), flush=True
    )


@app.cls(
    image=image,
    gpu="L4",
    cpu=(8, 8),
    memory=(65536, 65536),
    startup_timeout=120,
    timeout=60,
    retries=0,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=90,
    region="us",
    routing_region="us-east",
    volumes={"/compiled": cache},
    include_source=True,
)
@modal.concurrent(max_inputs=1)
class SimpleSceneRenderer:
    @modal.enter()
    def load(self):
        self.manifest, self.fixture, self.manifest_sha256 = _configuration()
        self.container_sha256 = _sha(os.environ["MODAL_TASK_ID"].encode())
        if not claims.put("initialization", self.container_sha256, skip_if_exists=True):
            raise ValueError("simple-scene initialization allowance exhausted")
        _event("initialization", "start", container_sha256=self.container_sha256)
        started = time.perf_counter()
        from klein_reference_runtime import KleinReferenceRuntime

        self.runtime = KleinReferenceRuntime(Path("/models"))
        if self.runtime.identity != self.manifest["expected_identity"]:
            raise ValueError("simple-scene runtime identity differs")
        cache_seconds = self.runtime.compile(Path("/compiled") / CACHE_ID)
        self.next_ordinal = 0
        self.base_hashes = {}
        _expiry(self.manifest)
        _event(
            "initialization",
            "end",
            container_sha256=self.container_sha256,
            startup_seconds=time.perf_counter() - started,
            model_load_seconds=self.runtime.load_seconds,
            cache_setup_seconds=cache_seconds,
        )

    @modal.method()
    def render(self, request_id, reference_jpeg=None, reference_sha256=None):
        _expiry(self.manifest)
        operations = self.manifest["operations"]
        if self.next_ordinal >= len(operations):
            raise ValueError("simple-scene operation allowance exhausted")
        operation = operations[self.next_ordinal]
        if request_id != operation["request_id"]:
            raise ValueError("simple-scene request order differs")
        reference_id = operation["reference_request_id"]
        if reference_id is None:
            if reference_jpeg is not None or reference_sha256 is not None:
                raise ValueError("simple-scene reference is unexpected")
        elif (
            reference_id not in self.base_hashes
            or reference_sha256 != self.base_hashes[reference_id]
            or not isinstance(reference_jpeg, bytes)
            or not 0 < len(reference_jpeg) <= 2_000_000
            or _sha(reference_jpeg) != reference_sha256
        ):
            raise ValueError("simple-scene reference differs from its first state")
        if not claims.put(f"request:{request_id}", True, skip_if_exists=True):
            raise ValueError("simple-scene request already attempted")
        self.next_ordinal += 1
        control = next(
            row for row in self.fixture["controls"] if row["id"] == operation["control_id"]
        )
        prompt = control["before_prompt" if operation["variant"] == "base" else "after_prompt"]
        text = self.runtime.pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(self.runtime.pipe.tokenizer(text)["input_ids"]) > 256:
            raise ValueError("simple-scene prompt exceeds qualified text bucket")
        _event("render", "start", ordinal=operation["ordinal"], variant=operation["variant"])
        try:
            metrics, master, depth = self.runtime.render(
                prompt,
                operation["seed"],
                reference_jpeg=reference_jpeg,
                reference_sha256=reference_sha256,
            )
        except Exception:
            _event("render", "failed", ordinal=operation["ordinal"])
            raise RuntimeError("simple-scene render failed") from None
        if operation["variant"] == "base":
            self.base_hashes[request_id] = _sha(master)
        _event("render", "end", ordinal=operation["ordinal"], variant=operation["variant"])
        return {
            "request_id": request_id,
            "identity": self.runtime.identity,
            "location": {
                "cloud": os.environ["MODAL_CLOUD_PROVIDER"],
                "compute_region": os.environ["MODAL_REGION"],
                "container_sha256": self.container_sha256,
            },
            "metrics": metrics,
            "master": master,
            "depth": depth,
        }
