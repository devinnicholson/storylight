from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (ROOT / "infra/gcp/k8s/anticipatory.yaml").read_text()
DEPLOY = (ROOT / "infra/gcp/gke/deploy-anticipatory.sh").read_text()
SUSPEND = (ROOT / "infra/gcp/gke/suspend-anticipatory.sh").read_text()
API_DEPLOY = (ROOT / "infra/gcp/gke/deploy-anticipatory-api.sh").read_text()
INFRASTRUCTURE_EVIDENCE = ROOT / "benchmarks/anticipatory-gke-infrastructure-2026-09-03.json"


def test_nemotron_is_an_independently_scalable_pinned_l4_workload() -> None:
    assert "name: bookforge-nemotron" in MANIFEST
    assert "replicas: 0" in MANIFEST
    assert "cloud.google.com/gke-accelerator: nvidia-l4" in MANIFEST
    assert "memory: 32Gi" in MANIFEST
    assert "memory: 41Gi" in MANIFEST
    assert "NIM_MAX_MODEL_LEN" in MANIFEST
    assert 'value: "2048"' in MANIFEST
    assert "bookforge-nim-cache" in MANIFEST
    assert "storage: 80Gi" in MANIFEST
    assert "BOOKFORGE_GPU_MAX_RUNTIME_SECONDS" in MANIFEST
    assert 'resources: ["deployments/scale"]' in MANIFEST
    assert 'resourceNames: ["bookforge-nemotron"]' in MANIFEST
    assert "automountServiceAccountToken: false" in MANIFEST


def test_cpu_api_uses_private_nim_dns_and_stays_deployable_while_gpu_is_off() -> None:
    assert "http://bookforge-nemotron.bookforge.svc.cluster.local:8000" in MANIFEST
    api = MANIFEST.split("name: bookforge-anticipatory", 2)[-1]
    assert "readinessProbe:" in api
    assert "path: /health" in api
    assert "path: /ready" not in api


def test_network_policy_preserves_workload_identity_and_private_nim_traffic() -> None:
    assert "cidr: 169.254.169.254/32" in MANIFEST
    assert "cidr: 169.254.169.252/32" in MANIFEST
    assert "port: 988" in MANIFEST
    assert "app.kubernetes.io/name: bookforge-nemotron" in MANIFEST


def test_failure_and_suspend_paths_scale_only_the_gpu_deployment() -> None:
    assert 'GPU_DEPLOYMENT="bookforge-nemotron"' in DEPLOY
    assert 'API_DEPLOYMENT="bookforge-anticipatory"' in DEPLOY
    assert "deployment/${GPU_DEPLOYMENT}" in DEPLOY
    assert "deployment/bookforge-nemotron --replicas=0" in SUSPEND


def test_cpu_api_release_never_applies_or_scales_the_nim_manifest() -> None:
    assert "kubectl apply" not in API_DEPLOY
    assert 'kubectl -n "${NAMESPACE}" scale' not in API_DEPLOY
    assert 'set image "deployment/${API_DEPLOYMENT}"' in API_DEPLOY
    assert "GPU_REPLICAS_BEFORE" in API_DEPLOY
    assert "GPU_REPLICAS_AFTER" in API_DEPLOY


def test_infrastructure_evidence_binds_sources_benchmarks_and_gpu_shutdown() -> None:
    evidence = json.loads(INFRASTRUCTURE_EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == "bookforge-anticipatory-gke-infrastructure-v1"
    assert evidence["post_run_gpu_state"]["regional_l4_quota_usage"] == 0
    assert evidence["nemotron"]["post_run_desired_replicas"] == 0
    assert evidence["gpu_watchdog"] == {
        "service_account": "bookforge-gpu-watchdog",
        "can_patch_nemotron_scale": True,
        "can_patch_api_scale": False,
        "identity_is_mounted_only_into_watchdog_sidecar": True,
        "actual_projected_token_scale_zero_to_zero_verified": True,
    }
    for record in evidence["source_files"].values():
        content = (ROOT / record["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
    for record in evidence["benchmarks"]:
        content = (ROOT / record["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
        report = json.loads(content)
        assert report["gates"]["passed"] is record["passed"]
