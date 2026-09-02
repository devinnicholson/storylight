"""Recover or regenerate the immutable accepted-engine development baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from bookforge.fidelity_benchmark import (
    CandidateIdentity,
    FidelitySummary,
    candidate_identity_from_manifest,
    population_contract_from_manifest,
)
from bookforge.fidelity_manifest import validate_manifest
from bookforge.fidelity_schema import DatasetSplit

from .integrity import canonical_json_bytes

APPROVAL_ENVIRONMENT = "BOOKFORGE_BASELINE_DEVELOPMENT_APPROVAL"
_SHA256 = frozenset("0123456789abcdef")
_REPORT_FIELDS = {
    "schema_version",
    "split",
    "candidate_identity",
    "dataset_manifest_sha256",
    "custody_receipt_sha256",
    "privacy",
    "summary",
}
_HEADLINE_FIELDS = frozenset(
    {
        "schema_valid_rate",
        "semantic_atom_recall",
        "exact_example_pass_rate",
        "privacy_pass_rate",
    }
)
_BASELINE_SIGNATURES_BY_DATASET_MANIFEST_SHA256: Mapping[
    str, Mapping[str, float]
] = MappingProxyType(
    {
        # story-fidelity-v1: preserve the pre-training accepted-engine measurement.
        "e717eb38c44fceeeae3a2bc88981767c316ca1339198ce1077b893252afeb1de": (
            MappingProxyType(
                {
                    "schema_valid_rate": 1.0,
                    "semantic_atom_recall": 0.587109375,
                    "exact_example_pass_rate": 0.0,
                    "privacy_pass_rate": 0.74609375,
                }
            )
        ),
        # story-fidelity-v2: accepted Jetson engine measured before candidate
        # prediction or evaluation on 2026-09-02.
        "fd3317ef440a04c9adc41825dda2b58c03a52ae0829bd422750522b4e11d428e": (
            MappingProxyType(
                {
                    "schema_valid_rate": 1.0,
                    "semantic_atom_recall": 0.537109375,
                    "exact_example_pass_rate": 0.0,
                    "privacy_pass_rate": 1.0,
                }
            )
        ),
    }
)


class BaselineDevelopmentError(ValueError):
    """Accepted baseline evidence is incomplete, mutable, or from another lineage."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _headline_signature(dataset_manifest_sha256: str) -> Mapping[str, float]:
    signature = _BASELINE_SIGNATURES_BY_DATASET_MANIFEST_SHA256.get(
        dataset_manifest_sha256
    )
    if signature is None:
        raise BaselineDevelopmentError(
            "no frozen baseline signature is registered for this dataset manifest"
        )
    if set(signature) != _HEADLINE_FIELDS or any(
        type(value) is not float or not 0.0 <= value <= 1.0
        for value in signature.values()
    ):
        raise BaselineDevelopmentError("registered baseline headline signature is malformed")
    return signature


def _regular(path: Path, label: str, *, nonempty: bool = True) -> os.stat_result:
    if path.is_symlink():
        raise BaselineDevelopmentError(f"{label} may not be a symbolic link")
    try:
        metadata = path.stat()
    except FileNotFoundError as error:
        raise BaselineDevelopmentError(f"{label} is missing") from error
    if not stat.S_ISREG(metadata.st_mode) or (nonempty and metadata.st_size == 0):
        raise BaselineDevelopmentError(f"{label} must be a non-empty regular file")
    return metadata


def _approved_json(path: Path, expected_sha256: str, label: str) -> dict[str, object]:
    _regular(path, label)
    if not _is_sha256(expected_sha256) or _sha256(path) != expected_sha256:
        raise BaselineDevelopmentError(f"{label} differs from its approved SHA-256")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BaselineDevelopmentError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise BaselineDevelopmentError(f"{label} must contain one JSON object")
    return document


def validate_accepted_identity(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    engine_path: Path,
    engine_sha256: str,
    dataset_manifest_sha256: str,
) -> tuple[CandidateIdentity, dict[str, object]]:
    """Verify the accepted manifest against the engine bytes still serving on Jetson."""

    engine_stat = _regular(engine_path, "accepted engine")
    if not _is_sha256(engine_sha256) or _sha256(engine_path) != engine_sha256:
        raise BaselineDevelopmentError("accepted engine differs from its approved SHA-256")
    document = _approved_json(manifest_path, manifest_sha256, "accepted-engine manifest")
    try:
        identity = candidate_identity_from_manifest(document, manifest_sha256=manifest_sha256)
    except ValueError as error:
        raise BaselineDevelopmentError(str(error)) from error
    engine_row = {
        "path": "engines/llm/llm.engine",
        "bytes": engine_stat.st_size,
        "sha256": engine_sha256,
    }
    if (
        document.get("schema_version") != "1.0"
        or document.get("result") != "complete"
        or document.get("identity_type") != "accepted-baseline"
        or not identity.candidate_id.startswith("accepted-baseline-")
        or identity.engine_sha256 != engine_sha256
        or document.get("source_dataset_manifest_sha256") != dataset_manifest_sha256
        or document.get("engine_path") != engine_row["path"]
        or document.get("file_count") != 1
        or document.get("total_bytes") != engine_stat.st_size
        or document.get("files") != [engine_row]
    ):
        raise BaselineDevelopmentError("accepted-engine manifest identity or file binding changed")
    return identity, document


def validate_baseline_report(
    path: Path,
    *,
    expected_sha256: str,
    identity: CandidateIdentity,
    dataset_manifest_path: Path,
    dataset_manifest_sha256: str,
) -> FidelitySummary:
    """Validate a privacy-safe report against the frozen accepted-engine signature."""

    document = _approved_json(path, expected_sha256, "baseline development report")
    if (
        set(document) != _REPORT_FIELDS
        or document.get("schema_version") != "story-fidelity-evaluation-v1"
        or document.get("split") != "development"
        or document.get("candidate_identity") != asdict(identity)
        or document.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or document.get("custody_receipt_sha256") is not None
        or document.get("privacy")
        != {"passages_recorded": False, "outputs_recorded": False}
    ):
        raise BaselineDevelopmentError("baseline development report identity or privacy changed")
    raw_summary = document.get("summary")
    if not isinstance(raw_summary, dict):
        raise BaselineDevelopmentError("baseline development report has no summary")
    try:
        summary = FidelitySummary(**raw_summary)
    except TypeError as error:
        raise BaselineDevelopmentError("baseline development summary is malformed") from error
    population = population_contract_from_manifest(
        dataset_manifest_path,
        expected_manifest_sha256=dataset_manifest_sha256,
        split=DatasetSplit.DEVELOPMENT,
    )
    if (
        summary.surface != "raw"
        or summary.split != "development"
        or summary.records != population.records
        or summary.record_ids_sha256 != population.record_ids_sha256
        or summary.counterfactual_pairs != population.pairs
        or dict(summary.category_record_counts) != dict(population.category_record_counts)
        or set(summary.category_pass_rates) != set(population.category_record_counts)
    ):
        raise BaselineDevelopmentError("baseline development report uses another population")
    signature = _headline_signature(dataset_manifest_sha256)
    if any(getattr(summary, name) != value for name, value in signature.items()):
        raise BaselineDevelopmentError("baseline headline metrics differ from the frozen baseline")
    return summary


def validate_baseline_bundle(
    root: Path,
    *,
    expected_completion_sha256: str,
    dataset_manifest_path: Path,
    dataset_manifest_sha256: str,
    development_records_sha256: str,
    accepted_manifest_sha256: str,
    accepted_engine_sha256: str,
) -> dict[str, object]:
    """Verify a materialized baseline directory using its trusted completion digest."""

    if root.is_symlink() or not root.is_dir():
        raise BaselineDevelopmentError("baseline bundle must be a regular directory")
    completion_path = root / "completion.json"
    completion = _approved_json(
        completion_path,
        expected_completion_sha256,
        "baseline completion",
    )
    fields = {
        "schema_version",
        "status",
        "stage",
        "mode",
        "candidate_identity",
        "dataset_manifest_sha256",
        "development_records_sha256",
        "source_binding_sha256",
        "runtime_attestation_sha256",
        "report_sha256",
        "privacy",
        "files",
    }
    identity = completion.get("candidate_identity")
    if (
        set(completion) != fields
        or completion.get("schema_version") != "1.0"
        or completion.get("status") != "succeeded"
        or completion.get("stage") != "accepted-baseline-development"
        or completion.get("mode") not in {"recover", "regenerate"}
        or not isinstance(identity, dict)
        or set(identity) != {
            "candidate_id",
            "candidate_manifest_sha256",
            "engine_sha256",
            "model_revision",
        }
        or not str(identity.get("candidate_id", "")).startswith("accepted-baseline-")
        or identity.get("candidate_manifest_sha256") != accepted_manifest_sha256
        or identity.get("engine_sha256") != accepted_engine_sha256
        or identity.get("model_revision") != f"sha256:{accepted_engine_sha256}"
        or completion.get("dataset_manifest_sha256") != dataset_manifest_sha256
        or completion.get("development_records_sha256") != development_records_sha256
        or not _is_sha256(completion.get("source_binding_sha256"))
        or (
            completion.get("runtime_attestation_sha256") is not None
            and not _is_sha256(completion.get("runtime_attestation_sha256"))
        )
        or (
            completion.get("mode") == "regenerate"
            and completion.get("runtime_attestation_sha256") is None
        )
        or (
            completion.get("mode") == "recover"
            and (
                completion.get("runtime_attestation_sha256") is not None
                or completion.get("source_binding_sha256")
                != completion.get("report_sha256")
            )
        )
        or not _is_sha256(completion.get("report_sha256"))
        or completion.get("privacy")
        != {"passages_recorded": False, "outputs_recorded": False}
    ):
        raise BaselineDevelopmentError("baseline completion identity or lineage changed")
    report = root / "baseline-development.json"
    expected_row = {
        "path": report.name,
        "bytes": report.stat().st_size if report.is_file() else -1,
        "sha256": completion["report_sha256"],
    }
    if completion.get("files") != [expected_row]:
        raise BaselineDevelopmentError("baseline completion file table changed")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != completion_path
    }
    if actual != {report.name}:
        raise BaselineDevelopmentError("baseline bundle has undeclared or missing files")
    candidate_identity = CandidateIdentity(
        candidate_id=str(identity.get("candidate_id")),
        candidate_manifest_sha256=accepted_manifest_sha256,
        engine_sha256=accepted_engine_sha256,
        model_revision=f"sha256:{accepted_engine_sha256}",
    )
    validate_baseline_report(
        report,
        expected_sha256=str(completion["report_sha256"]),
        identity=candidate_identity,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    return completion


def approval_token(
    *,
    mode: str,
    accepted_manifest_sha256: str,
    accepted_engine_sha256: str,
    dataset_manifest_sha256: str,
    development_records_sha256: str,
    source_binding_sha256: str,
) -> str:
    return (
        f"APPROVE_ACCEPTED_BASELINE_DEVELOPMENT:{mode}:{accepted_manifest_sha256}:"
        f"{accepted_engine_sha256}:{dataset_manifest_sha256}:"
        f"{development_records_sha256}:{source_binding_sha256}"
    )


def _write_once(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _endpoint_binding(base_url: str, model: str, timeout_seconds: float) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 11435
        or parsed.path not in {"", "/"}
    ):
        raise BaselineDevelopmentError(
            "baseline regeneration requires the accepted endpoint at 127.0.0.1:11435"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BaselineDevelopmentError("baseline regeneration endpoint contains unsupported fields")
    if model != "llm" or timeout_seconds <= 0:
        raise BaselineDevelopmentError(
            "baseline endpoint must use model llm and a positive timeout"
        )
    return hashlib.sha256(
        canonical_json_bytes(
            {"base_url": base_url.rstrip("/"), "model": model, "timeout_seconds": timeout_seconds}
        )
    ).hexdigest()


def _accepted_endpoint_attestation(base_url: str, engine_path: Path) -> str:
    """Bind the loopback server process to the accepted engine directory."""

    _endpoint_binding(base_url, "llm", 12.0)
    unit = "bookforge-tensorrt-planner.service"
    active = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        check=False,
        timeout=10,
    )
    if active.returncode != 0:
        raise BaselineDevelopmentError("accepted TensorRT service is not active")
    pid_result = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=MainPID", "--value"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    try:
        pid = int(pid_result.stdout.strip())
    except ValueError as error:
        raise BaselineDevelopmentError("accepted TensorRT service has no valid main PID") from error
    process = Path("/proc") / str(pid)
    if pid <= 1 or not process.is_dir() or process.stat().st_uid != os.getuid():
        raise BaselineDevelopmentError("accepted TensorRT service process identity changed")
    command = [
        value.decode("utf-8")
        for value in (process / "cmdline").read_bytes().split(b"\0")
        if value
    ]

    def option(name: str) -> str:
        try:
            return command[command.index(name) + 1]
        except (ValueError, IndexError) as error:
            raise BaselineDevelopmentError(
                f"accepted TensorRT service command has no {name} binding"
            ) from error

    if (
        Path(option("--model")).resolve() != engine_path.parent.resolve()
        or option("--host") != "127.0.0.1"
        or option("--port") != "11435"
    ):
        raise BaselineDevelopmentError("accepted TensorRT service command changed")
    return hashlib.sha256(
        canonical_json_bytes(
            {"unit": unit, "pid": pid, "uid": os.getuid(), "command": command}
        )
    ).hexdigest()


def materialize_baseline(
    *,
    dataset_manifest_path: Path,
    dataset_manifest_sha256: str,
    development_records_path: Path,
    development_records_sha256: str,
    accepted_manifest_path: Path,
    accepted_manifest_sha256: str,
    accepted_engine_path: Path,
    accepted_engine_sha256: str,
    output_directory: Path,
    source_report_path: Path | None,
    source_report_sha256: str | None,
    base_url: str,
    model: str,
    timeout_seconds: float,
    runner=subprocess.run,
    endpoint_probe=_accepted_endpoint_attestation,
) -> dict[str, object]:
    """Copy trusted evidence or regenerate it, publishing completion last."""

    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"refusing to overwrite baseline evidence: {output_directory}")
    _regular(dataset_manifest_path, "dataset manifest")
    _regular(development_records_path, "development records")
    if (
        not _is_sha256(dataset_manifest_sha256)
        or _sha256(dataset_manifest_path) != dataset_manifest_sha256
    ):
        raise BaselineDevelopmentError("dataset manifest differs from its approved SHA-256")
    manifest = validate_manifest(dataset_manifest_path)
    development = manifest.splits[DatasetSplit.DEVELOPMENT]
    if (
        not _is_sha256(development_records_sha256)
        or _sha256(development_records_path) != development_records_sha256
        or development.sha256 != development_records_sha256
        or development_records_path.resolve()
        != (dataset_manifest_path.parent / str(development.path)).resolve()
    ):
        raise BaselineDevelopmentError("development records differ from the manifest population")
    identity, _ = validate_accepted_identity(
        manifest_path=accepted_manifest_path,
        manifest_sha256=accepted_manifest_sha256,
        engine_path=accepted_engine_path,
        engine_sha256=accepted_engine_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )

    mode = "recover" if source_report_path is not None else "regenerate"
    if mode == "recover":
        if source_report_sha256 is None:
            raise BaselineDevelopmentError("recovery requires the source report SHA-256")
        validate_baseline_report(
            source_report_path,
            expected_sha256=source_report_sha256,
            identity=identity,
            dataset_manifest_path=dataset_manifest_path,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
        source_binding_sha256 = source_report_sha256
    else:
        if source_report_sha256 is not None:
            raise BaselineDevelopmentError("regeneration must not accept a source report SHA-256")
        source_binding_sha256 = _endpoint_binding(base_url, model, timeout_seconds)
    token = approval_token(
        mode=mode,
        accepted_manifest_sha256=accepted_manifest_sha256,
        accepted_engine_sha256=accepted_engine_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        development_records_sha256=development_records_sha256,
        source_binding_sha256=source_binding_sha256,
    )
    if os.environ.get(APPROVAL_ENVIRONMENT) != token:
        raise RuntimeError("exact baseline development approval token is required")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.partial-", dir=output_directory.parent)
    )
    try:
        report = temporary / "baseline-development.json"
        runtime_attestation_sha256: str | None = None
        if source_report_path is not None:
            _write_once(report, source_report_path.read_bytes())
        else:
            runtime_attestation_sha256 = endpoint_probe(base_url, accepted_engine_path)
            runner(
                [
                    sys.executable,
                    "-m",
                    "bookforge.fidelity_endpoint_evaluation",
                    "--records",
                    str(development_records_path),
                    "--manifest",
                    str(dataset_manifest_path),
                    "--manifest-sha256",
                    dataset_manifest_sha256,
                    "--split",
                    "development",
                    "--candidate-manifest",
                    str(accepted_manifest_path),
                    "--candidate-manifest-sha256",
                    accepted_manifest_sha256,
                    "--serving-engine-sha256",
                    accepted_engine_sha256,
                    "--base-url",
                    base_url.rstrip("/"),
                    "--model",
                    model,
                    "--timeout-seconds",
                    str(timeout_seconds),
                    "--output",
                    str(report),
                    "--execute",
                ],
                check=True,
                timeout=max(900, int(development.records * timeout_seconds + 300)),
            )
            os.chmod(report, 0o400)
        report_sha256 = _sha256(report)
        validate_baseline_report(
            report,
            expected_sha256=report_sha256,
            identity=identity,
            dataset_manifest_path=dataset_manifest_path,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
        if mode == "recover" and report_sha256 != source_report_sha256:
            raise BaselineDevelopmentError("source baseline report changed during recovery")
        if mode == "regenerate" and endpoint_probe(
            base_url, accepted_engine_path
        ) != runtime_attestation_sha256:
            raise BaselineDevelopmentError("accepted TensorRT service changed during regeneration")
        if (
            accepted_engine_path.is_symlink()
            or _sha256(accepted_engine_path) != accepted_engine_sha256
        ):
            raise BaselineDevelopmentError(
                "accepted engine changed during baseline materialization"
            )
        if (
            accepted_manifest_path.is_symlink()
            or _sha256(accepted_manifest_path) != accepted_manifest_sha256
        ):
            raise BaselineDevelopmentError(
                "accepted manifest changed during baseline materialization"
            )
        if (
            _sha256(dataset_manifest_path) != dataset_manifest_sha256
            or _sha256(development_records_path) != development_records_sha256
        ):
            raise BaselineDevelopmentError("development population changed during materialization")
        completion: dict[str, object] = {
            "schema_version": "1.0",
            "status": "succeeded",
            "stage": "accepted-baseline-development",
            "mode": mode,
            "candidate_identity": asdict(identity),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "development_records_sha256": development_records_sha256,
            "source_binding_sha256": source_binding_sha256,
            "runtime_attestation_sha256": runtime_attestation_sha256,
            "report_sha256": report_sha256,
            "privacy": {"passages_recorded": False, "outputs_recorded": False},
            "files": [
                {
                    "path": report.name,
                    "bytes": report.stat().st_size,
                    "sha256": report_sha256,
                }
            ],
        }
        completion_path = temporary / "completion.json"
        _write_once(completion_path, canonical_json_bytes(completion))
        completion_sha256 = _sha256(completion_path)
        os.chmod(temporary, 0o500)
        os.replace(temporary, output_directory)
        parent = os.open(output_directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        with suppress(OSError):
            os.chmod(temporary, 0o700)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {**completion, "completion_sha256": completion_sha256}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--development-records", type=Path, required=True)
    parser.add_argument("--development-records-sha256", required=True)
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--accepted-manifest-sha256", required=True)
    parser.add_argument("--accepted-engine", type=Path, required=True)
    parser.add_argument("--accepted-engine-sha256", required=True)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--source-report-sha256")
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default="llm")
    parser.add_argument("--timeout-seconds", type=float, default=12)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    mode = "recover" if args.source_report is not None else "regenerate"
    if (args.source_report is None) != (args.source_report_sha256 is None):
        raise SystemExit("--source-report and --source-report-sha256 must be supplied together")
    source_binding_sha256 = (
        args.source_report_sha256
        if args.source_report is not None
        else _endpoint_binding(args.base_url, args.model, args.timeout_seconds)
    )
    token = approval_token(
        mode=mode,
        accepted_manifest_sha256=args.accepted_manifest_sha256,
        accepted_engine_sha256=args.accepted_engine_sha256,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        development_records_sha256=args.development_records_sha256,
        source_binding_sha256=source_binding_sha256,
    )
    plan = {
        "schema_version": "1.0",
        "mode": mode,
        "output_directory": str(args.output_directory),
        "source_binding_sha256": source_binding_sha256,
        "approval_token": token,
        "remote_mutation": False,
        "paid_resources": False,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    result = materialize_baseline(
        dataset_manifest_path=args.dataset_manifest,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        development_records_path=args.development_records,
        development_records_sha256=args.development_records_sha256,
        accepted_manifest_path=args.accepted_manifest,
        accepted_manifest_sha256=args.accepted_manifest_sha256,
        accepted_engine_path=args.accepted_engine,
        accepted_engine_sha256=args.accepted_engine_sha256,
        output_directory=args.output_directory,
        source_report_path=args.source_report,
        source_report_sha256=args.source_report_sha256,
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
