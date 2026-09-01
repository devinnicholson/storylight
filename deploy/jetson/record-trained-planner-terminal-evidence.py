#!/usr/bin/env python3
"""Atomically record terminal planner deployment evidence on a Jetson."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from pathlib import Path

SHA = re.compile(r"[a-f0-9]{64}\Z")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,63}\Z")
CANDIDATE_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,95}\Z")
USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}\Z")
EVIDENCE_ROOT = Path("/var/lib/bookforge/trained-planner/evidence")


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, expected_sha256: str) -> dict[str, object]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
            raise ValueError(f"checksum-bound JSON is missing or unbounded: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"checksum-bound JSON is missing or changed: {path}")

    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON repeats key: {key}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def parse_environment(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def user_systemctl(user: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    account = pwd.getpwnam(user)
    return subprocess.run(
        [
            "runuser",
            "-u",
            user,
            "--",
            "env",
            f"XDG_RUNTIME_DIR=/run/user/{account.pw_uid}",
            f"DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{account.pw_uid}/bus",
            "systemctl",
            "--user",
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


def request_json(
    url: str, payload: dict[str, object] | None = None, *, timeout: int = 15
) -> dict[str, object]:
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"content-type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ValueError(f"endpoint returned a non-object: {url}")
    return value


def lineage(
    gate: dict[str, object],
    manifest: dict[str, object],
    *,
    candidate_manifest_sha256: str,
    gate_artifact_sha256: str,
    baseline_engine_sha256: str,
    active_engine_sha256: str,
    outcome: str,
) -> dict[str, str]:
    candidate_identity = gate.get("candidate_identity")
    baseline_identity = gate.get("baseline_identity")
    values = {
        "run_id": gate.get("run_id"),
        "training_run_id": gate.get("training_run_id"),
        "candidate_id": manifest.get("candidate_id"),
    }
    if (
        not isinstance(values["run_id"], str)
        or RUN_ID.fullmatch(values["run_id"]) is None
        or not isinstance(values["training_run_id"], str)
        or RUN_ID.fullmatch(values["training_run_id"]) is None
        or not isinstance(values["candidate_id"], str)
        or CANDIDATE_ID.fullmatch(values["candidate_id"]) is None
    ):
        raise ValueError("terminal lineage contains an invalid identifier")
    required_status = "passed" if outcome == "promoted" else "rejected"
    config_sha256 = gate.get("config_sha256")
    dataset_manifest_sha256 = gate.get("dataset_manifest_sha256")
    if (
        gate.get("schema_version") != "1.0"
        or gate.get("stage") != "gate"
        or gate.get("producer") != "bookforge-fidelity-gate-builder"
        or gate.get("status") != required_status
        or not isinstance(config_sha256, str)
        or SHA.fullmatch(config_sha256) is None
        or not isinstance(dataset_manifest_sha256, str)
        or SHA.fullmatch(dataset_manifest_sha256) is None
        or gate.get("candidate_manifest_sha256") != candidate_manifest_sha256
        or manifest.get("training_run_id") != values["training_run_id"]
        or manifest.get("source_config_sha256") != config_sha256
        or manifest.get("source_dataset_manifest_sha256")
        != dataset_manifest_sha256
        or not isinstance(candidate_identity, dict)
        or candidate_identity.get("candidate_id") != values["candidate_id"]
        or candidate_identity.get("candidate_manifest_sha256") != candidate_manifest_sha256
        or not isinstance(baseline_identity, dict)
        or not isinstance(baseline_identity.get("candidate_id"), str)
        or not baseline_identity["candidate_id"].startswith("accepted-baseline-")
        or not isinstance(baseline_identity.get("candidate_manifest_sha256"), str)
        or SHA.fullmatch(baseline_identity["candidate_manifest_sha256"]) is None
        or baseline_identity.get("engine_sha256") != baseline_engine_sha256
        or baseline_identity.get("model_revision") != f"sha256:{baseline_engine_sha256}"
    ):
        raise ValueError("gate, candidate, and accepted baseline lineage do not match")
    candidate_engine = candidate_identity.get("engine_sha256")
    if not isinstance(candidate_engine, str) or SHA.fullmatch(candidate_engine) is None:
        raise ValueError("gate candidate engine identity is invalid")
    candidate_revision = f"sha256:{candidate_engine}"
    if (
        candidate_identity.get("model_revision") != candidate_revision
        or manifest.get("engine_sha256") != candidate_engine
        or manifest.get("model_revision") != candidate_revision
    ):
        raise ValueError("gate and manifest candidate engine identities do not match")
    expected_active = candidate_engine if outcome == "promoted" else baseline_engine_sha256
    if active_engine_sha256 != expected_active:
        raise ValueError("active engine does not match the terminal outcome")
    return {
        **values,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "gate_artifact_sha256": gate_artifact_sha256,
        "baseline_engine_sha256": baseline_engine_sha256,
        "active_engine_sha256": active_engine_sha256,
    }


def validated_output_path(path: Path) -> Path:
    if not path.is_absolute() or path.parent != EVIDENCE_ROOT or path.name in {"", ".", ".."}:
        raise ValueError("terminal output must be one immediate child of the evidence root")
    return path


def expected_approval_token(
    *,
    outcome: str,
    user: str,
    candidate_id: str,
    candidate_manifest_sha256: str,
    gate_artifact_sha256: str,
    baseline_engine_sha256: str,
    output_directory: Path,
) -> str:
    output = validated_output_path(output_directory)
    if outcome == "promoted":
        return (
            "PROMOTE_BOOKFORGE_TRAINED_PLANNER:"
            f"{user}:{candidate_id}:{candidate_manifest_sha256}:"
            f"{gate_artifact_sha256}:{output}"
        )
    if outcome != "retained":
        raise ValueError("terminal outcome is invalid")
    action = hashlib.sha256()
    for value in (
        user,
        gate_artifact_sha256,
        candidate_manifest_sha256,
        baseline_engine_sha256,
        str(output),
    ):
        action.update(value.encode())
        action.update(b"\0")
    return f"RETAIN_BOOKFORGE_ACCEPTED_BASELINE:{action.hexdigest()}"


def documents(
    *,
    outcome: str,
    lineage_values: dict[str, str],
    approval_sha256: str,
    maximum_output_tokens: int,
    backup_config_sha256: str | None,
) -> dict[str, dict[str, object]]:
    common = {
        "run_id": lineage_values["run_id"],
        "training_run_id": lineage_values["training_run_id"],
        "candidate_id": lineage_values["candidate_id"],
        "candidate_manifest_sha256": lineage_values["candidate_manifest_sha256"],
        "gate_artifact_sha256": lineage_values["gate_artifact_sha256"],
    }
    active = lineage_values["active_engine_sha256"]
    receipt = {
        "schema_version": "story-fidelity-terminal-receipt-v1",
        "producer": (
            "bookforge-trained-planner-promotion"
            if outcome == "promoted"
            else "bookforge-baseline-retention"
        ),
        "status": "succeeded",
        "outcome": outcome,
        **common,
        "active_engine_sha256": active,
        "accepted_baseline_engine_sha256": lineage_values["baseline_engine_sha256"],
        "one_purpose_approval_sha256": approval_sha256,
    }
    health = {
        "schema_version": "story-fidelity-post-action-health-v1",
        "producer": "bookforge-terminal-health-recorder",
        "status": "passed",
        "outcome": outcome,
        **common,
        "active_engine_sha256": active,
        "active_model_revision": f"sha256:{active}",
        "checks": {
            "planner_ready": True,
            "api_ready": True,
            "controller_ready": True,
            "kiosk_active": True,
            "live_scene_passed": True,
            "output_tokens_bounded": True,
            "swap_disabled": True,
        },
        "observations": {"maximum_output_tokens": maximum_output_tokens},
    }
    result = {"terminal-receipt.json": receipt, "post-action-health.json": health}
    if outcome == "promoted":
        if backup_config_sha256 is None or SHA.fullmatch(backup_config_sha256) is None:
            raise ValueError("promotion has no checksum-bound rollback config")
        result["rollback-state.json"] = {
            "schema_version": "story-fidelity-rollback-state-v1",
            "producer": "bookforge-trained-planner-rollback-state",
            "status": "ready",
            **common,
            "accepted_baseline_engine_sha256": lineage_values["baseline_engine_sha256"],
            "backup_config_sha256": backup_config_sha256,
            "restorable": True,
        }
    return result


def publish(output: Path, values: dict[str, dict[str, object]]) -> dict[str, str]:
    output = validated_output_path(output)
    root_info = EVIDENCE_ROOT.lstat()
    if (
        EVIDENCE_ROOT.is_symlink()
        or not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != 0
        or root_info.st_gid != 0
        or root_info.st_mode & 0o077
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("terminal output is not a new path under the trusted evidence root")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        os.chown(temporary, 0, 0)
        os.chmod(temporary, 0o700)
        digests: dict[str, str] = {}
        for name, document in values.items():
            target = temporary / name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            descriptor = os.open(target, flags, 0o600)
            try:
                payload = canonical(document)
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
                os.fchown(descriptor, 0, 0)
                os.fchmod(descriptor, 0o600)
                digests[name] = hashlib.sha256(payload).hexdigest()
            finally:
                os.close(descriptor)
        os.replace(temporary, output)
        return digests
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", choices=("promoted", "retained"), required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--gate-artifact", type=Path, required=True)
    parser.add_argument("--gate-artifact-sha256", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--active-engine", type=Path, required=True)
    parser.add_argument("--active-engine-sha256", required=True)
    parser.add_argument("--baseline-engine-sha256", required=True)
    parser.add_argument("--backup-config", type=Path)
    parser.add_argument(
        "--active-state",
        type=Path,
        default=Path("/var/lib/bookforge/trained-planner/active.env"),
    )
    parser.add_argument("--one-purpose-approval-token", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for label, value in (
        ("gate artifact", args.gate_artifact_sha256),
        ("candidate manifest", args.candidate_manifest_sha256),
        ("active engine", args.active_engine_sha256),
        ("baseline engine", args.baseline_engine_sha256),
    ):
        if SHA.fullmatch(value) is None:
            raise SystemExit(f"{label} SHA-256 is invalid")
    if os.geteuid() != 0:
        raise SystemExit("terminal evidence recording must run as root")
    if USER.fullmatch(args.user) is None:
        raise SystemExit("--user is invalid")
    pwd.getpwnam(args.user)
    gate = load_json(args.gate_artifact, args.gate_artifact_sha256)
    manifest = load_json(args.candidate_manifest, args.candidate_manifest_sha256)
    engine_info = args.active_engine.lstat()
    if (
        args.active_engine.is_symlink()
        or not stat.S_ISREG(engine_info.st_mode)
        or engine_info.st_uid != 0
        or engine_info.st_gid != 0
        or engine_info.st_mode & 0o222
        or sha256(args.active_engine) != args.active_engine_sha256
    ):
        raise SystemExit("active engine is not the approved immutable identity")
    lineage_values = lineage(
        gate,
        manifest,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        gate_artifact_sha256=args.gate_artifact_sha256,
        baseline_engine_sha256=args.baseline_engine_sha256,
        active_engine_sha256=args.active_engine_sha256,
        outcome=args.outcome,
    )
    output = validated_output_path(args.output_directory)
    expected_token = expected_approval_token(
        outcome=args.outcome,
        user=args.user,
        candidate_id=lineage_values["candidate_id"],
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        gate_artifact_sha256=args.gate_artifact_sha256,
        baseline_engine_sha256=args.baseline_engine_sha256,
        output_directory=output,
    )
    if not hmac.compare_digest(args.one_purpose_approval_token, expected_token):
        raise SystemExit("terminal evidence approval token does not bind this exact action")
    model = request_json("http://127.0.0.1:11435/v1/models")
    api = request_json("http://127.0.0.1:8080/readyz")
    controller = request_json("http://127.0.0.1:8081/healthz")
    scene = request_json(
        "http://127.0.0.1:8080/v1/live-scene-planner/prepare",
        {
            "text": "A copper fox raises a blue lantern above a quiet paper observatory.",
            "visual_style": "layered paper theater",
            "seed": 20260901,
            "session_id": "bookforge-terminal-evidence",
        },
    )
    tokens = scene.get("output_tokens")
    checks_ok = (
        bool(model)
        and api.get("ready") is True
        and controller.get("ready") is True
        and scene.get("revision") == f"sha256:{args.active_engine_sha256}"
        and isinstance(tokens, int)
        and not isinstance(tokens, bool)
        and 1 <= tokens <= 64
        and user_systemctl(args.user, "is-active", "bookforge-kiosk.service").stdout.strip()
        == "active"
        and len(Path("/proc/swaps").read_text().splitlines()) == 1
    )
    if not checks_ok:
        raise SystemExit("post-action planner, API, controller, kiosk, or swap probe failed")
    backup_sha = None
    if args.outcome == "promoted":
        if args.backup_config is None or args.backup_config.is_symlink():
            raise SystemExit("promotion rollback config is missing or unsafe")
        backup_info = args.backup_config.lstat()
        state_info = args.active_state.lstat()
        if (
            not stat.S_ISREG(backup_info.st_mode)
            or backup_info.st_uid != 0
            or backup_info.st_gid != 0
            or backup_info.st_mode & 0o077
            or args.active_state.is_symlink()
            or not stat.S_ISREG(state_info.st_mode)
            or state_info.st_uid != 0
            or state_info.st_gid != 0
            or state_info.st_mode & 0o077
        ):
            raise SystemExit("promotion rollback files are not safe root-owned state")
        backup_sha = sha256(args.backup_config)
        backup_environment = parse_environment(args.backup_config)
        active_state = parse_environment(args.active_state)
        if (
            backup_environment.get("BOOKFORGE_LIVE_SCENE_PLANNER_BACKEND") != "tensorrt_slots"
            or backup_environment.get("BOOKFORGE_LIVE_SCENE_PLANNER_BASE_URL")
            != "http://127.0.0.1:11435"
            or backup_environment.get("BOOKFORGE_LIVE_SCENE_PLANNER_MODEL_REVISION")
            != f"sha256:{args.baseline_engine_sha256}"
            or active_state.get("PHASE") != "ACTIVE"
            or active_state.get("CANDIDATE_ID") != lineage_values["candidate_id"]
            or active_state.get("MANIFEST_SHA256") != args.candidate_manifest_sha256
            or active_state.get("GATE_EVIDENCE_SHA256") != args.gate_artifact_sha256
            or active_state.get("ENGINE_SHA256") != args.active_engine_sha256
            or active_state.get("ACCEPTED_ENGINE_SHA256") != args.baseline_engine_sha256
            or active_state.get("BACKUP_FILE") != str(args.backup_config)
            or active_state.get("BACKUP_SHA256") != backup_sha
        ):
            raise SystemExit("durable active state does not prove this exact rollback path")
    values = documents(
        outcome=args.outcome,
        lineage_values=lineage_values,
        approval_sha256=hashlib.sha256(expected_token.encode()).hexdigest(),
        maximum_output_tokens=tokens,
        backup_config_sha256=backup_sha,
    )
    digests = publish(output, values)
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "producer": "bookforge-terminal-evidence-recorder",
                "status": "published",
                "output_directory": str(output),
                "artifacts": {
                    name: {"path": str(output / name), "sha256": digest}
                    for name, digest in sorted(digests.items())
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
