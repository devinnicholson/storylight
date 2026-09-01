"""Evaluate a private fidelity split directly against a loopback model endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from bookforge.fidelity_benchmark import (
    CandidateIdentity,
    FidelitySummary,
    PopulationContract,
    candidate_identity_from_manifest,
    population_contract_from_manifest,
    summarize_evaluations,
)
from bookforge.fidelity_evaluation import SurfaceEvaluation, concept_vocabulary, evaluate_surface
from bookforge.fidelity_manifest import (
    FidelityDatasetManifest,
    sha256_path,
    validate_manifest,
)
from bookforge.fidelity_schema import DatasetSplit, FidelityRecord
from bookforge.tensorrt_slot_client import _slot_messages

PredictionFunction = Callable[[str], str]
HIDDEN_EVALUATION_STATE_ROOT = Path("/var/lib/bookforge/fidelity-hidden-evaluation")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _sha256(path: Path) -> str:
    return sha256_path(path)


def _secure_regular(path: Path, label: str, *, exact_mode: int | None = None) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symbolic link")
    try:
        metadata = path.stat()
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if exact_mode is not None and stat.S_IMODE(metadata.st_mode) != exact_mode:
        raise ValueError(f"{label} must have mode {exact_mode:04o}")


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    _secure_regular(path, label)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return document


def _load_candidate_identity(
    path: Path, *, expected_sha256: str, serving_engine_sha256: str
) -> tuple[CandidateIdentity, Mapping[str, object]]:
    if _SHA256.fullmatch(expected_sha256) is None or _sha256(path) != expected_sha256:
        raise ValueError("candidate manifest SHA-256 differs from the approved value")
    document = _load_json_object(path, "candidate manifest")
    identity = candidate_identity_from_manifest(document, manifest_sha256=expected_sha256)
    if identity.engine_sha256 != serving_engine_sha256:
        raise ValueError("serving engine digest differs from the candidate manifest")
    return identity, document


def _load_records(path: Path) -> list[FidelityRecord]:
    records: list[FidelityRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                records.append(FidelityRecord.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"invalid fidelity record at {path}:{line_number}") from error
    return records


def _assert_population(summary: FidelitySummary, contract: PopulationContract) -> None:
    if (
        summary.split != contract.split
        or summary.records != contract.records
        or summary.record_ids_sha256 != contract.record_ids_sha256
        or summary.counterfactual_pairs != contract.pairs
        or dict(summary.category_record_counts) != dict(contract.category_record_counts)
    ):
        raise ValueError("evaluated records do not match the approved manifest population")


def _assert_record_population(
    records: Sequence[FidelityRecord], contract: PopulationContract
) -> None:
    categories: Counter[str] = Counter()
    for record in records:
        categories.update(record.categories)
    if (
        len(records) != contract.records
        or _sequence_digest([record.record_id for record in records]) != contract.record_ids_sha256
        or len({record.pair_id for record in records}) != contract.pairs
        or dict(sorted(categories.items())) != dict(contract.category_record_counts)
    ):
        raise ValueError("record file does not match the approved manifest population")


def evaluate_endpoint_records(
    records: Sequence[FidelityRecord],
    *,
    predict: PredictionFunction,
    population: PopulationContract,
) -> FidelitySummary:
    """Evaluate in memory so passages and raw generations never enter evidence files."""

    vocabulary = concept_vocabulary(records)
    evaluations: list[SurfaceEvaluation] = []
    for record in records:
        raw = predict(record.passage)
        evaluations.append(
            evaluate_surface(record, raw, surface="raw", concept_vocabulary=vocabulary)
        )
    summary = summarize_evaluations(evaluations)
    _assert_population(summary, population)
    return summary


def _endpoint_predictor(base_url: str, model: str, timeout_seconds: float) -> PredictionFunction:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("private fidelity evaluation requires a loopback HTTP endpoint")
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"

    def predict(passage: str) -> str:
        payload = {
            "model": model,
            "messages": _slot_messages(passage),
            "temperature": 0,
            "max_tokens": 64,
            "stream": False,
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with _NO_PROXY_OPENER.open(request, timeout=timeout_seconds) as response:
            document = json.loads(response.read())
        try:
            content = document["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("endpoint returned no assistant content") from error
        if not isinstance(content, str):
            raise RuntimeError("endpoint assistant content was not text")
        return content

    return predict


def _approval_token(identity: CandidateIdentity, population: PopulationContract) -> str:
    return (
        "EVALUATE_PRIVATE_HIDDEN_ONCE:"
        f"{identity.candidate_manifest_sha256}:{identity.engine_sha256}:"
        f"{population.record_ids_sha256}"
    )


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_custody_receipt(
    path: Path,
    *,
    manifest: FidelityDatasetManifest,
    manifest_sha256: str,
    hidden_sha256: str,
) -> str:
    _secure_regular(path, "custody receipt", exact_mode=0o600)
    receipt = _load_json_object(path, "custody receipt")
    expected = {
        "schema_version": "story-fidelity-custody-v1",
        "dataset_id": manifest.dataset_id,
        "hidden_sha256": hidden_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "generator_source_sha256": manifest.generator_source_sha256,
        "generator_config_sha256": manifest.generator_config_sha256,
        "generator_runtime": manifest.generator_runtime.model_dump(mode="json"),
        "hidden_derivation": manifest.hidden_derivation,
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise ValueError(f"custody receipt {name} is not bound to the dataset manifest")
    fingerprint = receipt.get("key_fingerprint_sha256")
    if not isinstance(fingerprint, str) or _SHA256.fullmatch(fingerprint) is None:
        raise ValueError("custody receipt has no valid key fingerprint")
    return _sha256(path)


def _sequence_digest(values: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(values)).encode()).hexdigest()


def _validate_private_hidden_inputs(
    *,
    records_path: Path,
    manifest_path: Path,
    manifest_sha256: str,
    custody_receipt_path: Path,
) -> tuple[list[FidelityRecord], PopulationContract, str]:
    _secure_regular(records_path, "private hidden split", exact_mode=0o600)
    manifest = validate_manifest(manifest_path, private_hidden_path=records_path)
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError("dataset manifest SHA-256 differs from the approved value")
    population = population_contract_from_manifest(
        manifest_path,
        expected_manifest_sha256=manifest_sha256,
        split=DatasetSplit.HIDDEN,
    )
    if _sha256(records_path) != population.content_sha256:
        raise ValueError("private hidden split checksum differs from the manifest")
    receipt_sha256 = _validate_custody_receipt(
        custody_receipt_path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        hidden_sha256=population.content_sha256,
    )
    records = _load_records(records_path)
    _assert_record_population(records, population)
    if _sequence_digest([record.passage_sha256 for record in records]) != (
        population.passage_hashes_sha256
    ):
        raise ValueError("private hidden passage population differs from the manifest")
    if _sequence_digest([record.template_family for record in records]) != (
        population.template_families_sha256
    ):
        raise ValueError("private hidden template population differs from the manifest")
    return records, population, receipt_sha256


def _claim_hidden_evaluation(
    *,
    identity: CandidateIdentity,
    population: PopulationContract,
    plan: Mapping[str, object],
    state_root: Path = HIDDEN_EVALUATION_STATE_ROOT,
) -> Path:
    if state_root.is_symlink():
        raise ValueError("hidden evaluation state root may not be a symbolic link")
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not state_root.is_dir() or stat.S_IMODE(state_root.stat().st_mode) != 0o700:
        raise ValueError("hidden evaluation state root must be a mode-0700 directory")
    identity_digest = hashlib.sha256(
        (f"{identity.candidate_manifest_sha256}\n{identity.engine_sha256}\n").encode()
    ).hexdigest()
    state = state_root / identity_digest
    try:
        state.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeError(
            "this candidate and serving engine already consumed hidden evaluation"
        ) from error
    _write_once(state / "intent.json", {**plan, "status": "started"})
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--custody-receipt", type=Path)
    parser.add_argument("--split", choices=("development", "hidden"), required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--serving-engine-sha256", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11436")
    parser.add_argument("--model", default="llm")
    parser.add_argument("--timeout-seconds", type=float, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    split = DatasetSplit(args.split)
    identity, candidate_manifest = _load_candidate_identity(
        args.candidate_manifest,
        expected_sha256=args.candidate_manifest_sha256,
        serving_engine_sha256=args.serving_engine_sha256,
    )
    population = population_contract_from_manifest(
        args.manifest,
        expected_manifest_sha256=args.manifest_sha256,
        split=split,
    )
    source_dataset_sha = candidate_manifest.get("source_dataset_manifest_sha256")
    if source_dataset_sha != args.manifest_sha256:
        raise ValueError("candidate manifest was built from another dataset manifest")
    token = _approval_token(identity, population)
    plan: dict[str, object] = {
        "schema_version": "story-fidelity-evaluation-v1",
        "split": split.value,
        "candidate_identity": asdict(identity),
        "dataset_manifest_sha256": args.manifest_sha256,
        "population": asdict(population),
        "retains_passages": False,
        "retains_model_outputs": False,
        "approval_token": token if split is DatasetSplit.HIDDEN else None,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation evidence: {args.output}")

    state: Path | None = None
    receipt_sha256: str | None = None
    if split is DatasetSplit.HIDDEN:
        if args.custody_receipt is None:
            raise RuntimeError("hidden evaluation requires the custody receipt")
        if os.environ.get("BOOKFORGE_HIDDEN_EVAL_APPROVAL") != token:
            raise RuntimeError("hidden evaluation requires the exact one-shot approval token")
        records, population, receipt_sha256 = _validate_private_hidden_inputs(
            records_path=args.records,
            manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256,
            custody_receipt_path=args.custody_receipt,
        )
        plan["population"] = asdict(population)
        plan["custody_receipt_sha256"] = receipt_sha256
        state = _claim_hidden_evaluation(identity=identity, population=population, plan=plan)
    else:
        validate_manifest(args.manifest)
        if _sha256(args.records) != population.content_sha256:
            raise ValueError("development split checksum differs from the manifest")
        records = _load_records(args.records)
        _assert_record_population(records, population)

    summary = evaluate_endpoint_records(
        records,
        predict=_endpoint_predictor(args.base_url, args.model, args.timeout_seconds),
        population=population,
    )
    report: dict[str, object] = {
        "schema_version": "story-fidelity-evaluation-v1",
        "split": split.value,
        "candidate_identity": asdict(identity),
        "dataset_manifest_sha256": args.manifest_sha256,
        "custody_receipt_sha256": receipt_sha256,
        "privacy": {"passages_recorded": False, "outputs_recorded": False},
        "summary": asdict(summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_once(args.output, report)
    if state is not None:
        _write_once(
            state / "completion.json",
            {**plan, "status": "succeeded", "report_sha256": _sha256(args.output)},
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
