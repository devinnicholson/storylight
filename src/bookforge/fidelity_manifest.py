"""Content-addressed manifests for Story Fidelity Lab datasets."""

from __future__ import annotations

import hashlib
import json
import platform
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import pydantic
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from bookforge.fidelity_dataset import (
    CATEGORIES,
    DATASET_ID,
    HIDDEN_DERIVATION,
    SPLIT_COUNTS,
    generator_config_sha256,
    generator_source_sha256,
    load_jsonl,
)
from bookforge.fidelity_schema import (
    GENERATOR_ID,
    GENERATOR_REVISION,
    SCHEMA_VERSION,
    DatasetSplit,
    Digest,
    FidelityRecord,
)

MANIFEST_VERSION = "2.0"
VersionText = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}$"),
]


class FidelityRuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    python_implementation: Literal["CPython"] = "CPython"
    python_version: VersionText
    pydantic_version: VersionText


def current_runtime_identity() -> FidelityRuntimeIdentity:
    return FidelityRuntimeIdentity(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        pydantic_version=pydantic.__version__,
    )


class FidelitySplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None
    sha256: Digest
    records: Annotated[int, Field(gt=0)]
    families: Annotated[int, Field(gt=0)]
    pairs: Annotated[int, Field(gt=0)]
    public: bool
    categories: dict[str, Annotated[int, Field(gt=0)]]
    record_ids_sha256: Digest
    passage_hashes_sha256: Digest
    template_families_sha256: Digest

    @model_validator(mode="after")
    def validate_visibility(self) -> FidelitySplitManifest:
        if self.public != (self.path is not None):
            raise ValueError("public split visibility and path disagree")
        if self.path is not None:
            path = PurePosixPath(self.path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("split path must be a safe relative path")
        return self


class FidelityDatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: Literal[MANIFEST_VERSION] = MANIFEST_VERSION
    dataset_id: Literal[DATASET_ID] = DATASET_ID
    record_schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    record_schema_sha256: Digest
    generator: Literal[GENERATOR_ID] = GENERATOR_ID
    generator_revision: Literal[GENERATOR_REVISION] = GENERATOR_REVISION
    generator_source_sha256: Digest
    generator_config_sha256: Digest
    generator_runtime: FidelityRuntimeIdentity
    hidden_derivation: Literal[HIDDEN_DERIVATION] = HIDDEN_DERIVATION
    total_records: Literal[5120] = 5120
    source_policy: Literal["original-templates-only"] = "original-templates-only"
    hidden_policy: Literal["private-external-content-addressed"] = (
        "private-external-content-addressed"
    )
    splits: dict[DatasetSplit, FidelitySplitManifest]

    @model_validator(mode="after")
    def validate_splits(self) -> FidelityDatasetManifest:
        if set(self.splits) != set(DatasetSplit):
            raise ValueError("manifest requires train, development, and hidden splits")
        if sum(item.records for item in self.splits.values()) != self.total_records:
            raise ValueError("split counts do not equal total_records")
        for split, expected_count in SPLIT_COUNTS.items():
            item = self.splits[split]
            if item.records != expected_count:
                raise ValueError(f"incorrect {split.value} count")
            if set(item.categories) != set(CATEGORIES):
                raise ValueError(f"{split.value} does not cover every fidelity category")
        if self.splits[DatasetSplit.HIDDEN].public:
            raise ValueError("hidden records may not be published")
        if not self.splits[DatasetSplit.TRAIN].public:
            raise ValueError("train records must be published")
        if not self.splits[DatasetSplit.DEVELOPMENT].public:
            raise ValueError("development records must be published")
        return self

    def canonical_json(self) -> str:
        return (
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_digest(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("utf-8")
    return sha256_bytes(payload)


def record_schema_sha256() -> str:
    schema = json.dumps(
        FidelityRecord.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(schema)


def build_split_manifest(
    records: Sequence[FidelityRecord],
    *,
    content_sha256: str,
    path: str | None,
) -> FidelitySplitManifest:
    if not records:
        raise ValueError("cannot manifest an empty split")
    split = records[0].split
    if any(record.split != split for record in records):
        raise ValueError("split manifest received mixed splits")
    categories: Counter[str] = Counter()
    for record in records:
        categories.update(record.categories)
    return FidelitySplitManifest(
        path=path,
        sha256=content_sha256,
        records=len(records),
        families=len({record.family_id for record in records}),
        pairs=len({record.pair_id for record in records}),
        public=path is not None,
        categories=dict(sorted(categories.items())),
        record_ids_sha256=_sequence_digest([record.record_id for record in records]),
        passage_hashes_sha256=_sequence_digest([record.passage_sha256 for record in records]),
        template_families_sha256=_sequence_digest([record.template_family for record in records]),
    )


def build_dataset_manifest(
    split_records: Mapping[DatasetSplit, Sequence[FidelityRecord]],
    split_hashes: Mapping[DatasetSplit, str],
) -> FidelityDatasetManifest:
    if set(split_records) != set(DatasetSplit) or set(split_hashes) != set(DatasetSplit):
        raise ValueError("all three splits are required")
    public_paths = {
        DatasetSplit.TRAIN: "train.jsonl",
        DatasetSplit.DEVELOPMENT: "development.jsonl",
        DatasetSplit.HIDDEN: None,
    }
    manifests = {
        split: build_split_manifest(
            split_records[split],
            content_sha256=split_hashes[split],
            path=public_paths[split],
        )
        for split in DatasetSplit
    }
    return FidelityDatasetManifest(
        record_schema_sha256=record_schema_sha256(),
        generator_source_sha256=generator_source_sha256(),
        generator_config_sha256=generator_config_sha256(),
        generator_runtime=current_runtime_identity(),
        splits=manifests,
    )


def validate_manifest(
    manifest_path: Path,
    *,
    private_hidden_path: Path | None = None,
) -> FidelityDatasetManifest:
    manifest = FidelityDatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    root = manifest_path.parent
    loaded: list[FidelityRecord] = []
    for split in (DatasetSplit.TRAIN, DatasetSplit.DEVELOPMENT):
        split_manifest = manifest.splits[split]
        if split_manifest.path is None:
            raise ValueError(f"{split.value} path is missing")
        path = root / split_manifest.path
        if sha256_path(path) != split_manifest.sha256:
            raise ValueError(f"{split.value} checksum mismatch")
        records = load_jsonl(path)
        if len(records) != split_manifest.records:
            raise ValueError(f"{split.value} record count mismatch")
        if any(record.split != split for record in records):
            raise ValueError(f"{split.value} file contains another split")
        recomputed = build_split_manifest(
            records,
            content_sha256=split_manifest.sha256,
            path=split_manifest.path,
        )
        if recomputed != split_manifest:
            raise ValueError(f"{split.value} manifest metadata mismatch")
        loaded.extend(records)

    if private_hidden_path is not None:
        if private_hidden_path.is_symlink():
            raise ValueError("private hidden file may not be a symbolic link")
        try:
            hidden_stat = private_hidden_path.stat()
        except FileNotFoundError as error:
            raise ValueError("private hidden file is missing") from error
        if not stat.S_ISREG(hidden_stat.st_mode):
            raise ValueError("private hidden path must be a regular file")
        if hidden_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ValueError("private hidden file must not be accessible by group or others")
        hidden_manifest = manifest.splits[DatasetSplit.HIDDEN]
        if sha256_path(private_hidden_path) != hidden_manifest.sha256:
            raise ValueError("hidden checksum mismatch")
        hidden_records = load_jsonl(private_hidden_path)
        if len(hidden_records) != hidden_manifest.records:
            raise ValueError("hidden record count mismatch")
        if any(record.split != DatasetSplit.HIDDEN for record in hidden_records):
            raise ValueError("private hidden file contains another split")
        recomputed_hidden = build_split_manifest(
            hidden_records,
            content_sha256=hidden_manifest.sha256,
            path=None,
        )
        if recomputed_hidden != hidden_manifest:
            raise ValueError("hidden manifest metadata mismatch")
        loaded.extend(hidden_records)

    family_splits: dict[str, DatasetSplit] = {}
    pair_splits: dict[str, DatasetSplit] = {}
    template_splits: dict[str, DatasetSplit] = {}
    for record in loaded:
        for value, registry, label in (
            (record.family_id, family_splits, "family"),
            (record.pair_id, pair_splits, "pair"),
            (record.template_family, template_splits, "template family"),
        ):
            previous = registry.setdefault(value, record.split)
            if previous != record.split:
                raise ValueError(f"{label} leakage across splits: {value}")
    if manifest.record_schema_sha256 != record_schema_sha256():
        raise ValueError("record schema checksum mismatch")
    if manifest.generator_source_sha256 != generator_source_sha256():
        raise ValueError("generator source checksum mismatch")
    if manifest.generator_config_sha256 != generator_config_sha256():
        raise ValueError("generator configuration checksum mismatch")
    return manifest
