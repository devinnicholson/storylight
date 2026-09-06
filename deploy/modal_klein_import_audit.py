"""One CPU-only audit of framework imports; this does not qualify snapshot restoration."""

import hashlib
import json
import re
import resource
import time
from pathlib import Path

import modal

IMAGE_ID = "im-WtXer8GjRPdgMqWAAUSMwJ"
EXPERIMENT = "klein-import-audit-20260905-a"
MANIFEST = Path("/root/import-audit-manifest.json")
VERSIONS = {
    "torch": "2.8.0+cu128",
    "diffusers": "0.39.0",
    "transformers": "4.57.1",
    "triton": "3.4.0",
}

if modal.is_local():
    image = modal.Image.from_id(IMAGE_ID).add_local_file(
        Path(__file__).resolve().parents[1]
        / "benchmarks/renderer-import-audit-2026-09-05/manifest.json",
        str(MANIFEST),
    )
else:
    image = None

app = modal.App("bookforge-klein-import-audit")
claims = modal.Dict.from_name(f"bookforge-{EXPERIMENT}-claims", create_if_missing=True)


def _audit_imports():
    started = time.perf_counter()
    torch_seconds = 0.0
    attempts, patches = [], []
    packages = dict.fromkeys(VERSIONS, "unavailable")
    completed = guards_installed = False

    def guard(name):
        def reject(*args, **kwargs):
            if name not in attempts:
                attempts.append(name)
            raise RuntimeError("CUDA access forbidden during CPU import audit")

        return reject

    try:
        torch_started = time.perf_counter()
        try:
            import torch
        finally:
            torch_seconds = time.perf_counter() - torch_started
        packages["torch"] = (
            VERSIONS["torch"] if str(torch.__version__) == VERSIONS["torch"] else "unexpected"
        )
        for prefix, owner, methods in (
            (
                "torch.cuda",
                torch.cuda,
                (
                    "is_available",
                    "_lazy_init",
                    "device_count",
                    "get_device_capability",
                    "get_device_properties",
                    "current_device",
                    "init",
                ),
            ),
            ("torch._C", torch._C, ("_cuda_getDeviceCount", "_cuda_init")),
        ):
            for method in methods:
                original = getattr(owner, method)
                patches.append((owner, method, original))
                setattr(owner, method, guard(f"{prefix}.{method}"))
        guards_installed = True

        import diffusers
        import transformers
        import triton
        from transformers import (  # noqa: F401
            AutoImageProcessor,
            AutoModelForDepthEstimation,
            pipeline,
        )

        for name, module in (
            ("diffusers", diffusers),
            ("transformers", transformers),
            ("triton", triton),
        ):
            packages[name] = (
                VERSIONS[name] if str(module.__version__) == VERSIONS[name] else "unexpected"
            )
        completed = all(packages[name] == version for name, version in VERSIONS.items())
    except Exception:
        # Framework exceptions may include paths or unbounded text; the bounded evidence is enough.
        pass
    finally:
        for owner, method, original in reversed(patches):
            setattr(owner, method, original)
    return {
        "imports_seconds": time.perf_counter() - started,
        "torch_import_seconds": torch_seconds,
        "attempts": attempts,
        "import_completed": completed,
        "guards_installed": guards_installed,
        "packages": packages,
        "process_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "pre_torch_cuda_audit": "unknown",
    }


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(8192, 16384),
    startup_timeout=120,
    timeout=120,
    single_use_containers=True,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    cloud="aws",
    region="us-east",
    include_source=True,
)
def audit_imports(request_id: str):
    with MANIFEST.open("rb") as stream:
        raw = stream.read(8193)
    if len(raw) > 8192:
        raise ValueError("import audit manifest exceeds bound")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("import audit authorization differs")
    expires = manifest.get("expires_at")
    if (
        set(manifest)
        != {
            "schema_version",
            "status",
            "experiment_id",
            "expires_at",
            "image_id",
            "deployment_sha256",
            "request_id",
        }
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["status"] != "authorized"
        or manifest["experiment_id"] != EXPERIMENT
        or manifest["image_id"] != IMAGE_ID
        or type(expires) is not int
        or not time.time() < expires <= time.time() + 7200
        or not isinstance(request_id, str)
        or not re.fullmatch(r"[a-f0-9]{32}", request_id)
        or manifest["request_id"] != request_id
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest["deployment_sha256"]
    ):
        raise ValueError("import audit authorization differs")
    if not claims.put(request_id, True, skip_if_exists=True):
        raise ValueError("import audit already claimed")
    return _audit_imports()
