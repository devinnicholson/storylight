"""Supervise one isolated native-GCP qualification service and delete it afterward."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_gcp_klein as benchmark  # noqa: E402

PROJECT = "your-gcp-project"
REGION = "us-central1"
ACCOUNT = f"bookforge-renderer@{PROJECT}.iam.gserviceaccount.com"
IMAGE_PREFIX = f"{REGION}-docker.pkg.dev/{PROJECT}/bookforge/klein-qualification@sha256:"
LIFETIME_SECONDS = 600
CLEANUP_SECONDS = 60


def write(path: Path, value: dict | list) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def execute(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    if timeout <= 0:
        raise TimeoutError("supervised deadline reached")
    process = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def gcloud(args: list[str], timeout: float = 30) -> dict | list:
    result = execute(["gcloud", *args, "--project", PROJECT, "--format=json", "--quiet"], timeout)
    if result.returncode:
        raise RuntimeError("GCP control-plane command failed")
    return json.loads(result.stdout or "{}")


def deployment_args(service: str, image: str) -> list[str]:
    return [
        "run",
        "deploy",
        service,
        "--region",
        REGION,
        "--image",
        image,
        "--service-account",
        ACCOUNT,
        "--gpu=1",
        "--gpu-type=nvidia-rtx-pro-6000",
        "--no-gpu-zonal-redundancy",
        "--cpu=20",
        "--memory=80Gi",
        "--concurrency=1",
        "--min=0",
        "--min-instances=0",
        "--max=1",
        "--max-instances=1",
        "--timeout=180s",
        "--no-deploy-health-check",
        "--no-allow-unauthenticated",
        "--no-cpu-throttling",
        "--no-cpu-boost",
        "--revision-suffix=trial",
        "--startup-probe=tcpSocket.port=8080,periodSeconds=1,timeoutSeconds=1,failureThreshold=60",
        "--labels=app=bookforge,component=klein-qualification",
    ]


def verify_deployment(value: dict, service: str, image: str) -> tuple[str, str]:
    template = value["spec"]["template"]
    spec = template["spec"]
    annotations = template["metadata"]["annotations"]
    container = spec["containers"][0]
    revision = f"{service}-trial"
    checks = [
        value["metadata"]["name"] == service,
        value["metadata"].get("annotations", {}).get("run.googleapis.com/maxScale") == "1",
        value["metadata"].get("annotations", {}).get("run.googleapis.com/minScale", "0") == "0",
        annotations.get("autoscaling.knative.dev/maxScale") == "1",
        annotations.get("autoscaling.knative.dev/minScale", "0") == "0",
        annotations.get("run.googleapis.com/gpu-zonal-redundancy-disabled") == "true",
        annotations.get("run.googleapis.com/startup-cpu-boost") == "false",
        annotations.get("run.googleapis.com/cpu-throttling") == "false",
        spec["containerConcurrency"] == 1,
        spec["timeoutSeconds"] == 180,
        spec["serviceAccountName"] == ACCOUNT,
        spec["nodeSelector"]["run.googleapis.com/accelerator"] == "nvidia-rtx-pro-6000",
        len(spec["containers"]) == 1,
        container["image"] == image,
        container["resources"]["limits"] == {"cpu": "20", "memory": "80Gi", "nvidia.com/gpu": "1"},
        value["status"]["latestReadyRevisionName"] == revision,
        len(value["status"]["traffic"]) == 1,
        value["status"]["traffic"][0].get("revisionName") == revision,
        value["status"]["traffic"][0].get("percent") == 100,
        not value["status"]["traffic"][0].get("tag"),
    ]
    if not all(checks):
        raise ValueError("deployed qualification scope differs from the frozen scope")
    return value["status"]["url"], revision


def run(
    image: str,
    manifest_path: Path,
    output: Path,
    *,
    service: str | None = None,
    runtime_source: Path | None = None,
    worker_source: Path | None = None,
    export_compiler_cache: bool = False,
) -> None:
    if runtime_source is not None:
        runtime_source = runtime_source.resolve()
    if worker_source is not None:
        worker_source = worker_source.resolve()
    if export_compiler_cache and worker_source is None:
        raise ValueError("compiler export requires an explicit reviewed worker")
    manifest = benchmark.protocol.decode_json(manifest_path.read_bytes())
    benchmark.validate_manifest(
        manifest,
        active=True,
        runtime_source=runtime_source,
        worker_source=worker_source,
    )
    for key, path in {
        "worker": worker_source or ROOT / "deploy/gcp_klein_worker/app.py",
        "runtime": runtime_source or ROOT / "deploy/klein_scene_runtime.py",
        "weights": ROOT / "deploy/gcp_klein_worker/klein_weights.py",
    }.items():
        if manifest["sources"][key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("qualification source differs from the manifest")
    service = service or manifest["experiment_id"]
    if not re.fullmatch(r"bookforge-klein-qualification-20260906-[a-z]", service):
        raise ValueError("qualification service name is outside this experiment")
    if not re.fullmatch(re.escape(IMAGE_PREFIX) + r"[0-9a-f]{64}", image):
        raise ValueError("qualification requires its own immutable GCP image")
    if not time.time() + LIFETIME_SECONDS < manifest["expires_at"]:
        raise ValueError("manifest expires before the supervised experiment can finish")
    output.mkdir(parents=True, exist_ok=False)
    existing = gcloud(["run", "services", "list", "--region", REGION])
    if any(row["metadata"]["name"] == service for row in existing):
        raise ValueError("refusing to replace an existing service or retry a service name")
    write(
        output / "scope.json",
        {
            "project": PROJECT,
            "region": REGION,
            "service": service,
            "experiment_id": manifest["experiment_id"],
            "image": image,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "maximum_service_seconds": LIFETIME_SECONDS,
            "cleanup_reserve_seconds": CLEANUP_SECONDS,
            "deployment_timeout_seconds": 240,
            "iam_propagation_wait_seconds": 120,
            "maximum_gpu_instances_configured": 1,
            "two_slot_resource_lifecycle_allowance_usd": 1.1684904,
            "cost_is_not_a_hard_platform_cap": True,
            "production_changed": False,
            "export_compiler_cache": export_compiler_cache,
        },
    )
    started = time.monotonic()

    def bounded(args, timeout=30):
        remaining = LIFETIME_SECONDS - (time.monotonic() - started)
        return gcloud(args, min(timeout, remaining))

    attempted = False
    deployment_completed = False
    failed = None
    try:
        attempted = True
        bounded(deployment_args(service, image), 240)
        deployment_completed = True
        observed = bounded(["run", "services", "describe", service, "--region", REGION])
        write(output / "deployment.json", observed)
        url, revision = verify_deployment(observed, service, image)
        bounded(
            [
                "run",
                "services",
                "add-iam-policy-binding",
                service,
                "--region",
                REGION,
                "--member",
                f"serviceAccount:{ACCOUNT}",
                "--role=roles/run.invoker",
            ]
        )
        policy = bounded(["run", "services", "get-iam-policy", service, "--region", REGION])
        write(output / "iam.json", policy)
        if any(
            member in {"allUsers", "allAuthenticatedUsers"}
            for binding in policy.get("bindings", [])
            for member in binding["members"]
        ):
            raise ValueError("qualification service is not private")
        if not any(
            binding.get("role") == "roles/run.invoker"
            and f"serviceAccount:{ACCOUNT}" in binding.get("members", [])
            and not binding.get("condition")
            for binding in policy.get("bindings", [])
        ):
            raise ValueError("qualification invoker grant is missing")
        # A returned IAM policy does not mean the invocation frontend has received it.
        for _ in range(4):
            if LIFETIME_SECONDS - (time.monotonic() - started) <= 30:
                raise TimeoutError("insufficient supervised time for IAM propagation")
            time.sleep(30)
        remaining = LIFETIME_SECONDS - (time.monotonic() - started)
        if remaining <= 30:
            raise TimeoutError("insufficient supervised time for inference")
        command = [
            sys.executable,
            str(ROOT / "scripts/benchmark_gcp_klein.py"),
            "--run",
            "--manifest",
            str(manifest_path),
            "--proof-manifest-sha256",
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "--endpoint",
            url,
            "--service",
            service,
            "--revision",
            revision,
            "--output",
            str(output / "renders"),
        ]
        if runtime_source is not None:
            command.extend(["--runtime-source", str(runtime_source)])
        if worker_source is not None:
            command.extend(["--worker-source", str(worker_source)])
        result = execute(command, remaining)
        write(output / "client-exit.json", {"returncode": result.returncode})
        if result.returncode:
            raise RuntimeError("qualification client did not complete")
        summary = benchmark.protocol.decode_json((output / "renders/summary.json").read_bytes())
        if not (
            summary.get("completed_cases") == summary.get("declared_cases") == 10
            and summary.get("worker_count") == 1
            and summary.get("decision") == "ungraded"
            and len(summary.get("cases", [])) == 10
            and all(row.get("status") == "ok" for row in summary["cases"])
        ):
            raise RuntimeError("qualification workload was rejected")
        if export_compiler_cache:
            remaining = LIFETIME_SECONDS - (time.monotonic() - started)
            if remaining <= 5:
                raise TimeoutError("insufficient supervised time for compiler export")
            command = [
                sys.executable,
                str(ROOT / "experiments/renderer-native-cache/export_cache.py"),
                "--manifest",
                str(manifest_path),
                "--proof-manifest-sha256",
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "--renders",
                str(output / "renders"),
                "--endpoint",
                url,
                "--service",
                service,
                "--revision",
                revision,
                "--worker-source",
                str(worker_source),
                "--output",
                str(output / "compiler-cache"),
            ]
            if runtime_source is not None:
                command.extend(["--runtime-source", str(runtime_source)])
            result = execute(command, min(60, remaining))
            write(output / "export-exit.json", {"returncode": result.returncode})
            if result.returncode:
                raise RuntimeError("compiler cache export did not complete")
    except BaseException as error:
        failed = type(error).__name__
        raise
    finally:
        deleted = False
        absent = False
        cleanup_error = None
        cleanup_deadline = started + LIFETIME_SECONDS + CLEANUP_SECONDS

        def cleanup(args, timeout):
            remaining = cleanup_deadline - time.monotonic() - 3
            return gcloud(args, min(timeout, remaining))

        if attempted:
            try:
                cleanup(["run", "services", "delete", service, "--region", REGION], 20)
            except Exception as error:
                cleanup_error = type(error).__name__
            try:
                remaining_services = cleanup(["run", "services", "list", "--region", REGION], 10)
                absent = all(row["metadata"]["name"] != service for row in remaining_services)
                deleted = absent and deployment_completed
            except Exception as error:
                cleanup_error = type(error).__name__
        write(
            output / "closure.json",
            {
                "service_deleted_and_absent": deleted,
                "service_absent_at_observation": absent,
                "creation_terminal_observed": deployment_completed,
                "late_creation_cleanup_required": not deployment_completed,
                "failure": failed,
                "cleanup_error": cleanup_error,
                "supervisor_seconds": time.monotonic() - started,
                "billing_reconciliation_required": True,
                "absence_is_not_proof_of_zero_outstanding_charges": True,
            },
        )
        if not deleted:
            raise RuntimeError("qualification service deletion could not be verified")


def main() -> None:
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service", help="Fresh service name for an explicit deployment attempt")
    parser.add_argument(
        "--runtime-source",
        type=Path,
        help="Explicit reviewed runtime source matching both manifest hash pins",
    )
    parser.add_argument(
        "--worker-source", type=Path, help="Reviewed worker matching the manifest pin"
    )
    parser.add_argument("--export-compiler-cache", action="store_true")
    args = parser.parse_args()
    run(
        args.image,
        args.manifest.resolve(),
        args.output.resolve(),
        service=args.service,
        runtime_source=args.runtime_source,
        worker_source=args.worker_source,
        export_compiler_cache=args.export_compiler_cache,
    )


if __name__ == "__main__":
    main()
