from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from bookforge.fidelity_lineage import stable_run_id

RUN_SCHEMA_VERSION = "1.0"
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,63}")
_CANDIDATE_ID = re.compile(r"[a-z0-9][a-z0-9-]{2,95}")

_STAGE_PRODUCERS = {
    "roundtrip": "bookforge-jax-roundtrip-recorder",
    "train": "bookforge-jax-training-recorder",
    "candidate-eval": "bookforge-candidate-evaluation-recorder",
    "hf-export": "bookforge-hf-release-recorder",
    "int4-export": "bookforge-int4-export-recorder",
    "jetson-shadow": "bookforge-jetson-shadow-recorder",
    "promotion": "bookforge-trained-planner-promotion",
    "retain-baseline": "bookforge-baseline-retention",
    "reconcile": "bookforge-fidelity-reconciler",
}

_STAGE_EVIDENCE = {
    "roundtrip": {
        "conversion_run",
        "conversion_completion",
        "hf_to_maxtext_completion",
        "maxtext_to_hf_completion",
        "roundtrip",
        "exported_checkpoint_manifest",
        "smoke_training_run",
        "smoke_training_completion",
        "smoke_adapter_manifest",
    },
    "train": {
        "remote_completion",
        "training_run",
        "training_completion",
        "adapter_manifest",
        "package_manifest",
        "runtime_lock",
        "base_snapshot_completion",
        "base_checkpoint_manifest",
        "tokenizer_manifest",
    },
    "candidate-eval": {
        "development_summary",
        "dataset_manifest",
        "merged_checkpoint_manifest",
    },
    "hf-export": {
        "release_manifest",
        "training_run",
        "training_completion",
        "roundtrip",
        "development_summary",
    },
    "int4-export": {
        "source_release_manifest",
        "export_manifest",
        "calibration_provenance",
    },
    "jetson-shadow": {"runtime", "candidate_manifest", "hidden_summary"},
    "promotion": {"promotion_receipt", "post_promotion_health", "rollback_state"},
    "retain-baseline": {"retention_receipt", "accepted_engine_health"},
    "reconcile": {"cost_ledger", "paid_resource_inventory", "deployment_receipt"},
}

_COMMON_STAGE_FIELDS = {
    "schema_version",
    "stage",
    "producer",
    "run_id",
    "config_sha256",
    "dataset_manifest_sha256",
    "status",
    "inputs",
}
_STAGE_FIELDS = {
    "roundtrip": _COMMON_STAGE_FIELDS | {"roundtrip_status", "evidence_sha256"},
    "train": _COMMON_STAGE_FIELDS
    | {"training_run_id", "backend", "automatic_retries", "evidence_sha256"},
    "candidate-eval": _COMMON_STAGE_FIELDS
    | {
        "training_run_id",
        "candidate_id",
        "development_eligibility",
        "evidence_sha256",
    },
    "hf-export": _COMMON_STAGE_FIELDS
    | {"training_run_id", "candidate_id", "release_manifest_sha256", "evidence_sha256"},
    "int4-export": _COMMON_STAGE_FIELDS
    | {
        "training_run_id",
        "candidate_id",
        "source_release_manifest_sha256",
        "export_manifest_sha256",
        "evidence_sha256",
    },
    "jetson-shadow": _COMMON_STAGE_FIELDS
    | {
        "training_run_id",
        "candidate_id",
        "shadow_status",
        "candidate_manifest_sha256",
        "candidate_identity",
        "evidence_sha256",
    },
    "promotion": _COMMON_STAGE_FIELDS
    | {
        "training_run_id",
        "candidate_id",
        "candidate_manifest_sha256",
        "gate_artifact_sha256",
        "evidence_sha256",
    },
    "retain-baseline": _COMMON_STAGE_FIELDS
    | {
        "training_run_id",
        "candidate_id",
        "candidate_manifest_sha256",
        "gate_artifact_sha256",
        "evidence_sha256",
    },
    "reconcile": _COMMON_STAGE_FIELDS
    | {
        "training_run_id",
        "candidate_id",
        "gate_artifact_sha256",
        "deployment_artifact_sha256",
        "deployment_outcome",
        "gross_cost_reconciled",
        "active_paid_resources",
        "evidence_sha256",
    },
}


class FidelityRunError(RuntimeError):
    pass


class StageStatus(StrEnum):
    PENDING = "pending"
    BLOCKED = "blocked"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    dependencies: tuple[str, ...]
    approval_environment: str | None = None
    approval_value: str | None = None

    @property
    def requires_approval(self) -> bool:
        return self.approval_environment is not None


STAGES = (
    StageSpec("dataset", ()),
    StageSpec("cpu-smoke", ("dataset",)),
    StageSpec("compatibility-package", ("cpu-smoke",)),
    StageSpec("baseline", ("compatibility-package",)),
    StageSpec(
        "roundtrip",
        ("baseline",),
        "BOOKFORGE_JAX_ROUNDTRIP",
        "I_APPROVE_THIS_BOUNDED_ROUNDTRIP",
    ),
    StageSpec(
        "train",
        ("roundtrip",),
        "BOOKFORGE_JAX_TRAIN",
        "I_APPROVE_THIS_BOUNDED_TPU_JOB",
    ),
    StageSpec("candidate-eval", ("train",)),
    StageSpec("hf-export", ("candidate-eval",)),
    StageSpec(
        "int4-export",
        ("hf-export",),
        "BOOKFORGE_JAX_INT4_EXPORT",
        "I_APPROVE_THIS_BOUNDED_GPU_EXPORT",
    ),
    StageSpec(
        "jetson-shadow",
        ("int4-export",),
        "BOOKFORGE_JAX_JETSON_SHADOW",
        "I_APPROVE_THIS_REVERSIBLE_SHADOW_TEST",
    ),
    StageSpec("gate", ("jetson-shadow",)),
    StageSpec(
        "promotion",
        ("gate",),
        "BOOKFORGE_JAX_PROMOTE",
        "I_APPROVE_THIS_VERIFIED_ENGINE",
    ),
    StageSpec("retain-baseline", ("gate",)),
    StageSpec("reconcile", ("gate",)),
)
_STAGE_BY_NAME = {stage.name: stage for stage in STAGES}


@dataclass(slots=True)
class StageRecord:
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    artifact_sha256: str | None = None
    artifact_bytes: int | None = None
    artifact_status: str | None = None
    candidate_manifest_sha256: str | None = None
    gate_artifact_sha256: str | None = None
    detail: str | None = None


@dataclass(slots=True)
class FidelityRun:
    run_id: str
    config_sha256: str
    dataset_manifest_sha256: str
    baseline_commit: str
    baseline_engine_sha256: str
    created_at: str
    stages: dict[str, StageRecord]
    training_run_id: str | None = None
    candidate_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        config_sha256: str,
        dataset_manifest_sha256: str,
        baseline_commit: str,
        baseline_engine_sha256: str,
    ) -> FidelityRun:
        _validate_run_id(run_id)
        for label, value in (
            ("config SHA-256", config_sha256),
            ("dataset manifest SHA-256", dataset_manifest_sha256),
            ("baseline engine SHA-256", baseline_engine_sha256),
        ):
            _validate_sha256(value, label=label)
        if not baseline_commit.strip():
            raise ValueError("baseline commit must not be empty")
        return cls(
            run_id=run_id,
            config_sha256=config_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            baseline_commit=baseline_commit,
            baseline_engine_sha256=baseline_engine_sha256,
            created_at=_timestamp(),
            stages={stage.name: StageRecord() for stage in STAGES},
        )

    @classmethod
    def read(cls, path: Path) -> FidelityRun:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FidelityRunError(f"could not read fidelity run state: {error}") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != RUN_SCHEMA_VERSION:
            raise FidelityRunError("unsupported fidelity run state schema")
        _validate_run_id(payload.get("run_id"))
        _validate_sha256(payload.get("config_sha256"), label="config SHA-256")
        _validate_sha256(
            payload.get("dataset_manifest_sha256"),
            label="dataset manifest SHA-256",
        )
        _validate_sha256(
            payload.get("baseline_engine_sha256"),
            label="baseline engine SHA-256",
        )
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, dict) or set(raw_stages) != set(_STAGE_BY_NAME):
            raise FidelityRunError("fidelity run state does not contain the exact stage set")
        stages: dict[str, StageRecord] = {}
        try:
            for name, raw_record in raw_stages.items():
                stages[name] = StageRecord(
                    status=StageStatus(raw_record["status"]),
                    started_at=raw_record.get("started_at"),
                    finished_at=raw_record.get("finished_at"),
                    artifact_sha256=raw_record.get("artifact_sha256"),
                    artifact_bytes=raw_record.get("artifact_bytes"),
                    artifact_status=raw_record.get("artifact_status"),
                    candidate_manifest_sha256=raw_record.get("candidate_manifest_sha256"),
                    gate_artifact_sha256=raw_record.get("gate_artifact_sha256"),
                    detail=raw_record.get("detail"),
                )
        except (KeyError, TypeError, ValueError) as error:
            raise FidelityRunError("fidelity run state contains an invalid stage record") from error
        run = cls(
            run_id=payload["run_id"],
            config_sha256=payload["config_sha256"],
            dataset_manifest_sha256=payload["dataset_manifest_sha256"],
            baseline_commit=str(payload.get("baseline_commit", "")),
            baseline_engine_sha256=payload["baseline_engine_sha256"],
            created_at=str(payload.get("created_at", "")),
            stages=stages,
            training_run_id=payload.get("training_run_id"),
            candidate_id=payload.get("candidate_id"),
        )
        run.validate()
        return run

    def validate(self) -> None:
        if self.training_run_id is not None:
            _validate_training_run_id(self, self.training_run_id)
        if self.candidate_id is not None:
            _validate_candidate_id(self.candidate_id)
        completed: set[str] = set()
        active = 0
        for spec in STAGES:
            record = self.stages[spec.name]
            if record.status is StageStatus.COMPLETED:
                if not set(spec.dependencies).issubset(completed):
                    raise FidelityRunError(
                        f"completed stage {spec.name!r} has incomplete dependencies"
                    )
                if record.artifact_sha256 is None or record.artifact_bytes is None:
                    raise FidelityRunError(
                        f"completed stage {spec.name!r} has no artifact evidence"
                    )
                _validate_sha256(
                    record.artifact_sha256,
                    label=f"{spec.name} artifact SHA-256",
                )
                if record.artifact_bytes < 1:
                    raise FidelityRunError(
                        f"completed stage {spec.name!r} has an invalid artifact size"
                    )
                if record.artifact_status is None:
                    raise FidelityRunError(
                        f"completed stage {spec.name!r} has no typed artifact status"
                    )
                completed.add(spec.name)
            elif record.status is StageStatus.SKIPPED:
                if spec.name not in {"promotion", "retain-baseline"}:
                    raise FidelityRunError(f"stage {spec.name!r} may not be skipped")
                if record.detail not in {"gate-passed", "gate-rejected"}:
                    raise FidelityRunError(f"stage {spec.name!r} has no gate outcome")
            elif record.status is StageStatus.RUNNING:
                active += 1
        promotion = self.stages["promotion"]
        retained = self.stages["retain-baseline"]
        if self.stages["reconcile"].status is StageStatus.COMPLETED:
            outcomes = {promotion.status, retained.status}
            if outcomes != {StageStatus.COMPLETED, StageStatus.SKIPPED}:
                raise FidelityRunError(
                    "reconciliation requires exactly one completed deployment outcome"
                )
        if active > 1:
            raise FidelityRunError("only one fidelity stage may be running")
        if self.stages["train"].status is StageStatus.COMPLETED:
            if self.training_run_id is None:
                raise FidelityRunError("completed training has no immutable training run ID")
        elif self.training_run_id is not None:
            raise FidelityRunError("training run ID exists before training completed")
        if self.stages["candidate-eval"].status is StageStatus.COMPLETED:
            if self.candidate_id is None:
                raise FidelityRunError("completed candidate evaluation has no candidate ID")
        elif self.candidate_id is not None:
            raise FidelityRunError("candidate ID exists before candidate evaluation completed")

    def next_stage(self) -> StageSpec | None:
        for spec in STAGES:
            status = self.stages[spec.name].status
            if status in {StageStatus.PENDING, StageStatus.BLOCKED}:
                return spec
            if status is StageStatus.SKIPPED:
                continue
            if status in {StageStatus.RUNNING, StageStatus.FAILED}:
                return spec
        return None

    def begin(self, stage_name: str, *, environment: dict[str, str] | None = None) -> None:
        spec = _require_stage(stage_name)
        record = self.stages[stage_name]
        if record.status not in {StageStatus.PENDING, StageStatus.BLOCKED}:
            raise FidelityRunError(
                f"stage {stage_name!r} cannot begin from {record.status.value!r}"
            )
        if any(item.status is StageStatus.RUNNING for item in self.stages.values()):
            raise FidelityRunError("another fidelity stage is already running")
        incomplete = [
            name
            for name in spec.dependencies
            if self.stages[name].status is not StageStatus.COMPLETED
        ]
        if incomplete:
            raise FidelityRunError(
                f"stage {stage_name!r} has incomplete dependencies: {', '.join(incomplete)}"
            )
        if stage_name == "reconcile":
            outcomes = {
                self.stages["promotion"].status,
                self.stages["retain-baseline"].status,
            }
            if outcomes != {StageStatus.COMPLETED, StageStatus.SKIPPED}:
                raise FidelityRunError("reconciliation requires one completed deployment outcome")
        values = os.environ if environment is None else environment
        if spec.requires_approval and values.get(spec.approval_environment or "") != (
            spec.approval_value
        ):
            record.status = StageStatus.BLOCKED
            record.detail = f"requires explicit {spec.approval_environment} approval"
            raise FidelityRunError(record.detail)
        record.status = StageStatus.RUNNING
        record.started_at = _timestamp()
        record.finished_at = None
        record.artifact_sha256 = None
        record.artifact_bytes = None
        record.artifact_status = None
        record.candidate_manifest_sha256 = None
        record.gate_artifact_sha256 = None
        record.detail = None

    def complete(self, stage_name: str, *, artifact: Path) -> None:
        record = self.stages[_require_stage(stage_name).name]
        if record.status is not StageStatus.RUNNING:
            raise FidelityRunError(f"stage {stage_name!r} is not running")
        if not artifact.is_file() or artifact.is_symlink():
            raise FidelityRunError("stage artifact must be a regular file")
        document = _validated_stage_artifact(self, stage_name, artifact)
        artifact_sha256 = _file_sha256(artifact)
        status = str(document["status"])
        candidate_manifest_sha256 = document.get("candidate_manifest_sha256")
        gate_artifact_sha256 = document.get("gate_artifact_sha256")
        if stage_name == "train":
            self.training_run_id = str(document["training_run_id"])
        elif stage_name == "candidate-eval":
            self.candidate_id = str(document["candidate_id"])
        record.status = StageStatus.COMPLETED
        record.finished_at = _timestamp()
        record.artifact_sha256 = artifact_sha256
        record.artifact_bytes = artifact.stat().st_size
        record.artifact_status = status
        record.candidate_manifest_sha256 = (
            str(candidate_manifest_sha256) if candidate_manifest_sha256 else None
        )
        record.gate_artifact_sha256 = str(gate_artifact_sha256) if gate_artifact_sha256 else None
        record.detail = None
        if stage_name == "gate":
            skipped_name = "retain-baseline" if status == "passed" else "promotion"
            skipped = self.stages[skipped_name]
            skipped.status = StageStatus.SKIPPED
            skipped.finished_at = _timestamp()
            skipped.detail = f"gate-{status}"
        self.validate()

    def fail(self, stage_name: str, *, detail: str) -> None:
        record = self.stages[_require_stage(stage_name).name]
        if record.status is not StageStatus.RUNNING:
            raise FidelityRunError(f"stage {stage_name!r} is not running")
        detail = detail.strip()
        if not detail or len(detail) > 500:
            raise ValueError("failure detail must contain between 1 and 500 characters")
        record.status = StageStatus.FAILED
        record.finished_at = _timestamp()
        record.detail = detail

    def write(self, path: Path) -> None:
        self.validate()
        payload = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "baseline_commit": self.baseline_commit,
            "baseline_engine_sha256": self.baseline_engine_sha256,
            "created_at": self.created_at,
            "training_run_id": self.training_run_id,
            "candidate_id": self.candidate_id,
            "stages": {
                name: {**asdict(record), "status": record.status.value}
                for name, record in self.stages.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)


@contextmanager
def locked_fidelity_run(path: Path) -> Iterator[FidelityRun]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        run = FidelityRun.read(path)
        try:
            yield run
        except BaseException:
            run.write(path)
            raise
        else:
            run.write(path)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"file was not found: {path}")
    return _file_sha256(path)


def stage_plan() -> list[dict[str, object]]:
    return [
        {
            "name": stage.name,
            "dependencies": list(stage.dependencies),
            "requires_approval": stage.requires_approval,
            "approval_environment": stage.approval_environment,
        }
        for stage in STAGES
    ]


def typed_stage_document(
    run: FidelityRun,
    stage_name: str,
    *,
    status: str,
    fields: dict[str, object],
    evidence_sha256: dict[str, str],
) -> dict[str, object]:
    """Build one stage wrapper from already verified, immutable source evidence.

    The caller is responsible for validating each source document before calling this
    function. Keeping the shared lineage fields here prevents individual recorders from
    accidentally substituting a campaign, dependency, configuration, or dataset hash.
    """

    spec = _require_stage(stage_name)
    try:
        producer = _STAGE_PRODUCERS[stage_name]
    except KeyError as error:
        raise ValueError(f"stage has no typed recorder contract: {stage_name}") from error
    if "evidence_sha256" in fields or set(fields) & _COMMON_STAGE_FIELDS:
        raise ValueError("typed stage fields overlap recorder-owned fields")
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "stage": stage_name,
        "producer": producer,
        "run_id": run.run_id,
        "config_sha256": run.config_sha256,
        "dataset_manifest_sha256": run.dataset_manifest_sha256,
        "status": status,
        "inputs": {
            dependency: run.stages[dependency].artifact_sha256
            for dependency in spec.dependencies
        },
        **fields,
        "evidence_sha256": dict(evidence_sha256),
    }


def validate_stage_artifact(
    run: FidelityRun,
    stage_name: str,
    path: Path,
) -> dict[str, object]:
    """Validate a producer-owned stage artifact without mutating run state."""

    if not path.is_file() or path.is_symlink():
        raise FidelityRunError("stage artifact must be a regular file")
    return _validated_stage_artifact(run, stage_name, path)


def _require_stage(name: str) -> StageSpec:
    try:
        return _STAGE_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"unknown fidelity stage: {name}") from error


def _validated_stage_artifact(run: FidelityRun, stage_name: str, path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FidelityRunError("stage artifact must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise FidelityRunError("stage artifact must contain a JSON object")
    expected = {
        "schema_version": RUN_SCHEMA_VERSION,
        "stage": stage_name,
        "run_id": run.run_id,
        "config_sha256": run.config_sha256,
        "dataset_manifest_sha256": run.dataset_manifest_sha256,
    }
    for name, value in expected.items():
        if document.get(name) != value:
            raise FidelityRunError(f"stage artifact has invalid {name}")

    dependencies = _STAGE_BY_NAME[stage_name].dependencies
    expected_inputs = {
        dependency: run.stages[dependency].artifact_sha256 for dependency in dependencies
    }
    if document.get("inputs") != expected_inputs:
        raise FidelityRunError("stage artifact is not bound to its dependency evidence")

    if stage_name in {"dataset", "cpu-smoke", "compatibility-package"}:
        _validate_local_preflight_artifact(run, stage_name, document)
    elif stage_name == "baseline":
        _validate_baseline_artifact(run, document)
    elif stage_name in _STAGE_PRODUCERS:
        _validate_typed_stage_artifact(run, stage_name, document)

    status = document.get("status")
    allowed_status = {
        "gate": {"passed", "rejected"},
        "promotion": {"promoted"},
        "retain-baseline": {"retained"},
    }.get(stage_name, {"succeeded"})
    if status not in allowed_status:
        raise FidelityRunError(f"stage artifact has invalid status for {stage_name}")

    if stage_name in {"gate", "promotion", "retain-baseline"}:
        _validate_sha256(
            document.get("candidate_manifest_sha256"),
            label="candidate manifest SHA-256",
        )
    if stage_name == "gate":
        if document.get("producer") != "bookforge-fidelity-gate-builder":
            raise FidelityRunError("gate artifact has an invalid producer")
        if set(document) != {
            "schema_version",
            "stage",
            "producer",
            "run_id",
            "config_sha256",
            "dataset_manifest_sha256",
            "status",
            "inputs",
            "training_run_id",
            "candidate_manifest_sha256",
            "candidate_identity",
            "baseline_identity",
            "hidden_custody_receipt_sha256",
            "evidence_sha256",
            "decision",
        }:
            raise FidelityRunError("gate artifact has an unexpected top-level contract")
        if document.get("training_run_id") != run.training_run_id:
            raise FidelityRunError("gate artifact changed the training run ID")
        evidence = document.get("evidence_sha256")
        expected_evidence = {
            "baseline_development_summary",
            "baseline_hidden_summary",
            "candidate_development_summary",
            "candidate_hidden_summary",
            "candidate_manifest",
            "contest",
            "human_review",
            "runtime",
            "dataset_manifest",
        }
        if not isinstance(evidence, dict) or set(evidence) != expected_evidence:
            raise FidelityRunError("gate artifact has incomplete evidence bindings")
        for name, digest in evidence.items():
            _validate_sha256(digest, label=f"gate {name} SHA-256")
        if evidence["dataset_manifest"] != run.dataset_manifest_sha256:
            raise FidelityRunError("gate artifact changed the approved dataset manifest")
        if evidence["candidate_manifest"] != document["candidate_manifest_sha256"]:
            raise FidelityRunError("gate artifact changed the candidate manifest")
        shadow = run.stages["jetson-shadow"]
        if document["candidate_manifest_sha256"] != shadow.candidate_manifest_sha256:
            raise FidelityRunError("gate artifact differs from the shadowed candidate")
        candidate_identity = document.get("candidate_identity")
        baseline_identity = document.get("baseline_identity")
        _validate_candidate_identity(
            candidate_identity,
            expected_candidate_id=run.candidate_id,
            expected_manifest_sha256=shadow.candidate_manifest_sha256,
            label="gate candidate",
        )
        _validate_candidate_identity(
            baseline_identity,
            expected_engine_sha256=run.baseline_engine_sha256,
            label="gate baseline",
        )
        _validate_sha256(
            document.get("hidden_custody_receipt_sha256"),
            label="hidden custody receipt SHA-256",
        )
        decision = document.get("decision")
        if (
            not isinstance(decision, dict)
            or set(decision) != {"passed", "reasons", "checks"}
            or type(decision.get("passed")) is not bool
        ):
            raise FidelityRunError("gate artifact has no typed promotion decision")
        reasons = decision.get("reasons")
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) and reason for reason in reasons
        ):
            raise FidelityRunError("gate artifact has invalid decision reasons")
        checks = decision.get("checks")
        if (
            not isinstance(checks, dict)
            or not checks
            or any(type(value) is not bool for value in checks.values())
        ):
            raise FidelityRunError("gate artifact has invalid decision checks")
        passed = status == "passed"
        if decision["passed"] is not passed or (passed and reasons) or (not passed and not reasons):
            raise FidelityRunError("gate status and promotion decision disagree")
        if passed and not all(checks.values()):
            raise FidelityRunError("passed gate contains a failed decision check")
    if stage_name in {"promotion", "retain-baseline", "reconcile"}:
        gate = run.stages["gate"]
        if gate.status is not StageStatus.COMPLETED or gate.artifact_sha256 is None:
            raise FidelityRunError("deployment outcome requires completed gate evidence")
        if document.get("gate_artifact_sha256") != gate.artifact_sha256:
            raise FidelityRunError("deployment outcome is not bound to gate evidence")
        if stage_name != "reconcile" and (
            document.get("candidate_manifest_sha256") != gate.candidate_manifest_sha256
        ):
            raise FidelityRunError("deployment outcome changed the gated candidate")
    if stage_name == "promotion" and run.stages["gate"].artifact_status != "passed":
        raise FidelityRunError("promotion requires a passed gate")
    if stage_name == "retain-baseline" and run.stages["gate"].artifact_status != "rejected":
        raise FidelityRunError("baseline retention requires a rejected gate")
    if stage_name == "reconcile":
        promoted = run.stages["promotion"].status is StageStatus.COMPLETED
        expected_outcome = "promoted" if promoted else "retained"
        if document.get("deployment_outcome") != expected_outcome:
            raise FidelityRunError("reconciliation outcome does not match deployment state")
        if document.get("gross_cost_reconciled") is not True:
            raise FidelityRunError("reconciliation must confirm gross cost")
        if document.get("active_paid_resources") != 0:
            raise FidelityRunError("reconciliation must confirm zero paid resources")
    return document


def _validate_typed_stage_artifact(
    run: FidelityRun,
    stage_name: str,
    document: dict[str, object],
) -> None:
    if document.get("producer") != _STAGE_PRODUCERS[stage_name]:
        raise FidelityRunError(f"{stage_name} artifact has an invalid producer")
    if set(document) != _STAGE_FIELDS[stage_name]:
        raise FidelityRunError(f"{stage_name} artifact has an unexpected top-level contract")
    _validate_evidence_bindings(
        document,
        expected=_STAGE_EVIDENCE[stage_name],
        label=stage_name,
    )

    if stage_name == "roundtrip":
        if document.get("roundtrip_status") != "passed":
            raise FidelityRunError("roundtrip artifact did not pass the numerical seam")
        return

    if stage_name == "train":
        _validate_training_run_id(run, document.get("training_run_id"))
        if document.get("backend") not in {"vertex-tpu-v6e", "modal-l40s"}:
            raise FidelityRunError("training artifact has an invalid finite backend")
        if document.get("automatic_retries") != 0:
            raise FidelityRunError("training artifact did not disable automatic retries")
        return

    if document.get("training_run_id") != run.training_run_id:
        raise FidelityRunError(f"{stage_name} artifact changed the training run ID")

    if stage_name == "candidate-eval":
        _validate_candidate_id(document.get("candidate_id"))
        if document["evidence_sha256"]["dataset_manifest"] != run.dataset_manifest_sha256:
            raise FidelityRunError("candidate evaluation changed the dataset manifest")
        if document.get("development_eligibility") != {
            "passed": True,
            "hidden_evaluated": False,
        }:
            raise FidelityRunError("candidate evaluation is not development-only and passing")
        return

    if document.get("candidate_id") != run.candidate_id:
        raise FidelityRunError(f"{stage_name} artifact changed the candidate ID")

    if stage_name == "hf-export":
        digest = document.get("release_manifest_sha256")
        _validate_sha256(digest, label="merged-HF release manifest SHA-256")
        if document["evidence_sha256"]["release_manifest"] != digest:
            raise FidelityRunError("HF export evidence changed the release manifest")
    elif stage_name == "int4-export":
        source = document.get("source_release_manifest_sha256")
        exported = document.get("export_manifest_sha256")
        _validate_sha256(source, label="source merged-HF release manifest SHA-256")
        _validate_sha256(exported, label="INT4 export manifest SHA-256")
        if (
            document["evidence_sha256"]["source_release_manifest"] != source
            or document["evidence_sha256"]["export_manifest"] != exported
        ):
            raise FidelityRunError("INT4 export evidence changed its release lineage")
    elif stage_name == "jetson-shadow":
        if document.get("shadow_status") not in {"passed", "rejected"}:
            raise FidelityRunError("Jetson shadow artifact has no terminal shadow status")
        manifest_sha256 = document.get("candidate_manifest_sha256")
        _validate_sha256(manifest_sha256, label="shadowed candidate manifest SHA-256")
        if document["evidence_sha256"]["candidate_manifest"] != manifest_sha256:
            raise FidelityRunError("Jetson shadow evidence changed the candidate manifest")
        _validate_candidate_identity(
            document.get("candidate_identity"),
            expected_candidate_id=run.candidate_id,
            expected_manifest_sha256=str(manifest_sha256),
            label="shadow candidate",
        )
    elif stage_name in {"promotion", "retain-baseline"}:
        if (
            document.get("candidate_manifest_sha256")
            != run.stages["gate"].candidate_manifest_sha256
        ):
            raise FidelityRunError(f"{stage_name} changed the gated candidate manifest")
    elif stage_name == "reconcile":
        outcome_stage = (
            "promotion"
            if run.stages["promotion"].status is StageStatus.COMPLETED
            else "retain-baseline"
        )
        if document.get("deployment_artifact_sha256") != run.stages[outcome_stage].artifact_sha256:
            raise FidelityRunError("reconciliation changed the deployment receipt")


def _validate_evidence_bindings(
    document: dict[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    evidence = document.get("evidence_sha256")
    if not isinstance(evidence, dict) or set(evidence) != expected:
        raise FidelityRunError(f"{label} artifact has incomplete evidence bindings")
    for name, digest in evidence.items():
        _validate_sha256(digest, label=f"{label} {name} SHA-256")


def _validate_candidate_identity(
    value: object,
    *,
    expected_candidate_id: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_engine_sha256: str | None = None,
    label: str,
) -> None:
    expected_fields = {
        "candidate_id",
        "candidate_manifest_sha256",
        "engine_sha256",
        "model_revision",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise FidelityRunError(f"{label} identity is incomplete")
    _validate_candidate_id(value.get("candidate_id"))
    manifest_sha256 = value.get("candidate_manifest_sha256")
    engine_sha256 = value.get("engine_sha256")
    _validate_sha256(manifest_sha256, label=f"{label} manifest SHA-256")
    _validate_sha256(engine_sha256, label=f"{label} engine SHA-256")
    if value.get("model_revision") != f"sha256:{engine_sha256}":
        raise FidelityRunError(f"{label} model revision is not its engine checksum")
    if expected_candidate_id is not None and value["candidate_id"] != expected_candidate_id:
        raise FidelityRunError(f"{label} identity changed the candidate ID")
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise FidelityRunError(f"{label} identity changed the candidate manifest")
    if expected_engine_sha256 is not None and engine_sha256 != expected_engine_sha256:
        raise FidelityRunError(f"{label} identity changed the engine")


def _validate_local_preflight_artifact(
    run: FidelityRun,
    stage_name: str,
    document: dict[str, object],
) -> None:
    if document.get("producer") != "bookforge-local-preflight":
        raise FidelityRunError("local preflight artifact has an invalid producer")
    evidence = document.get("evidence")
    if not isinstance(evidence, dict):
        raise FidelityRunError("local preflight artifact has no evidence")
    if evidence.get("dataset_manifest_sha256") != run.dataset_manifest_sha256:
        raise FidelityRunError("local preflight artifact changed the dataset manifest")
    if stage_name == "dataset":
        split_records = evidence.get("split_records")
        if split_records != {"train": 4096, "development": 512, "hidden": 512}:
            raise FidelityRunError("dataset artifact has invalid split counts")
        for name in (
            "record_schema_sha256",
            "generator_source_sha256",
            "generator_config_sha256",
        ):
            _validate_sha256(evidence.get(name), label=f"dataset {name}")
    elif stage_name == "cpu-smoke":
        _validate_sha256(evidence.get("fixture_sha256"), label="CPU smoke fixture SHA-256")
        summary = evidence.get("summary")
        if (
            not isinstance(summary, dict)
            or summary.get("records") != 32
            or summary.get("schema_valid_rate") != 1.0
            or summary.get("privacy_pass_rate") != 1.0
            or summary.get("exact_example_pass_rate") != 1.0
        ):
            raise FidelityRunError("CPU smoke artifact is not an exact 32-record pass")
    else:
        if (
            evidence.get("config_sha256") != run.config_sha256
            or evidence.get("formatted_records") != 32
        ):
            raise FidelityRunError("compatibility artifact changed its configuration or records")
        _validate_sha256(
            evidence.get("formatted_smoke_sha256"),
            label="formatted smoke SHA-256",
        )
        if not isinstance(evidence.get("roundtrip_contract"), dict):
            raise FidelityRunError("compatibility artifact has no roundtrip contract")


def _validate_baseline_artifact(run: FidelityRun, document: dict[str, object]) -> None:
    if document.get("producer") != "bookforge-baseline-recorder":
        raise FidelityRunError("baseline artifact has an invalid producer")
    identity = document.get("baseline_identity")
    if not isinstance(identity, dict):
        raise FidelityRunError("baseline artifact has no engine identity")
    if (
        identity.get("engine_sha256") != run.baseline_engine_sha256
        or identity.get("model_revision") != f"sha256:{run.baseline_engine_sha256}"
    ):
        raise FidelityRunError("baseline artifact does not identify the accepted engine")
    _validate_sha256(
        document.get("hidden_custody_receipt_sha256"),
        label="baseline hidden custody receipt SHA-256",
    )
    evidence = document.get("evidence_sha256")
    expected = {
        "baseline_manifest",
        "development_summary",
        "hidden_summary",
        "dataset_manifest",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected:
        raise FidelityRunError("baseline artifact has incomplete evidence bindings")
    for name, digest in evidence.items():
        _validate_sha256(digest, label=f"baseline {name} SHA-256")
    if evidence["dataset_manifest"] != run.dataset_manifest_sha256:
        raise FidelityRunError("baseline artifact changed the dataset manifest")


def _validate_run_id(value: object) -> None:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ValueError("run ID must contain 3-64 lowercase letters, digits, or hyphens")


def _validate_training_run_id(run: FidelityRun, value: object) -> None:
    expected = stable_run_id(
        stage="lora-train",
        config_sha256=run.config_sha256,
        dataset_manifest_sha256=run.dataset_manifest_sha256,
    )
    if value != expected:
        raise FidelityRunError("training run ID is not the stable approved lineage ID")


def _validate_candidate_id(value: object) -> None:
    if not isinstance(value, str) or _CANDIDATE_ID.fullmatch(value) is None:
        raise FidelityRunError("candidate ID must be one immutable lowercase slug")


def _validate_sha256(value: object, *, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
