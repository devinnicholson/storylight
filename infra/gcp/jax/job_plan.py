#!/usr/bin/env python3
"""Build, but never submit, one bounded Vertex AI JAX CustomJob."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ID = "your-gcp-project"
REGION = "us-east1"
MACHINE_TYPE = "ct6e-standard-1t"
TPU_CHIPS = 1
TIMEOUT_SECONDS = 2_700
GROSS_CEILING_USD = 3.50
REPLICAS = 1
_DIGEST_IMAGE = re.compile(r"^[a-z0-9][a-z0-9./_:-]*@sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[a-z][a-z0-9-]{7,62}$")


@dataclass(frozen=True, slots=True)
class JobInputs:
    run_id: str
    image_uri: str
    config_sha256: str
    dataset_manifest_sha256: str
    prepared_train_sha256: str
    input_manifest_sha256: str
    base_checkpoint_manifest_sha256: str
    base_checkpoint_receipt_sha256: str
    tokenizer_manifest_sha256: str
    service_account: str
    scratch_uri: str
    release_uri: str
    smoke: bool

    def validate(self) -> None:
        if not _RUN_ID.fullmatch(self.run_id):
            raise ValueError("run_id must be an immutable lowercase slug")
        if not _DIGEST_IMAGE.fullmatch(self.image_uri):
            raise ValueError("image_uri must be pinned by sha256 digest")
        if not _SHA256.fullmatch(self.config_sha256):
            raise ValueError("config_sha256 must be a lowercase SHA-256")
        if not _SHA256.fullmatch(self.dataset_manifest_sha256):
            raise ValueError("dataset_manifest_sha256 must be a lowercase SHA-256")
        if not _SHA256.fullmatch(self.prepared_train_sha256):
            raise ValueError("prepared_train_sha256 must be a lowercase SHA-256")
        if not _SHA256.fullmatch(self.input_manifest_sha256):
            raise ValueError("input_manifest_sha256 must be a lowercase SHA-256")
        if not _SHA256.fullmatch(self.base_checkpoint_manifest_sha256):
            raise ValueError("base checkpoint manifest must be a lowercase SHA-256")
        if not _SHA256.fullmatch(self.base_checkpoint_receipt_sha256):
            raise ValueError("base checkpoint receipt must be a lowercase SHA-256")
        if not _SHA256.fullmatch(self.tokenizer_manifest_sha256):
            raise ValueError("tokenizer manifest must be a lowercase SHA-256")
        expected_account_suffix = f"@{PROJECT_ID}.iam.gserviceaccount.com"
        if not self.service_account.endswith(expected_account_suffix):
            raise ValueError("service_account must belong to the Bookforge project")
        if not self.scratch_uri.startswith("gs://") or not self.release_uri.startswith("gs://"):
            raise ValueError("scratch and release locations must be gs:// URIs")
        scratch_bucket = self.scratch_uri.removeprefix("gs://").split("/", 1)[0]
        release_bucket = self.release_uri.removeprefix("gs://").split("/", 1)[0]
        if scratch_bucket == release_bucket:
            raise ValueError("scratch and release buckets must be separate")


def build_custom_job(inputs: JobInputs) -> dict[str, object]:
    """Return the exact one-worker job body accepted by the Vertex REST API."""

    inputs.validate()
    input_prefix = f"{inputs.scratch_uri.rstrip('/')}/inputs/{inputs.run_id}"
    release_prefix = f"{inputs.release_uri.rstrip('/')}/releases/{inputs.run_id}"
    arguments = [
        "--run-id",
        inputs.run_id,
        "--input-prefix",
        input_prefix,
        "--release-prefix",
        release_prefix,
        "--config-sha256",
        inputs.config_sha256,
        "--dataset-manifest-sha256",
        inputs.dataset_manifest_sha256,
        "--prepared-train-sha256",
        inputs.prepared_train_sha256,
        "--input-manifest-sha256",
        inputs.input_manifest_sha256,
        "--base-checkpoint-manifest-sha256",
        inputs.base_checkpoint_manifest_sha256,
        "--base-checkpoint-receipt-sha256",
        inputs.base_checkpoint_receipt_sha256,
        "--tokenizer-manifest-sha256",
        inputs.tokenizer_manifest_sha256,
    ]
    if inputs.smoke:
        arguments.append("--smoke")
    return {
        "displayName": inputs.run_id,
        "jobSpec": {
            "workerPoolSpecs": [
                {
                    "replicaCount": str(REPLICAS),
                    "machineSpec": {"machineType": MACHINE_TYPE},
                    "containerSpec": {
                        "imageUri": inputs.image_uri,
                        "command": [
                            "python3",
                            "/opt/bookforge/infra/gcp/jax/vertex_entrypoint.py",
                        ],
                        "args": arguments,
                        "env": [
                            {"name": "JAX_PLATFORMS", "value": "tpu"},
                            {"name": "JAX_COMPILATION_CACHE_DIR", "value": "/tmp/jax-cache"},
                            {"name": "HF_HUB_OFFLINE", "value": "1"},
                            {"name": "HF_DATASETS_OFFLINE", "value": "1"},
                            {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                            {"name": "BOOKFORGE_PAID_ATTEMPT", "value": "1"},
                        ],
                    },
                }
            ],
            "serviceAccount": inputs.service_account,
            "baseOutputDirectory": {"outputUriPrefix": release_prefix},
            "scheduling": {
                "timeout": f"{TIMEOUT_SECONDS}s",
                "restartJobOnWorkerRestart": False,
                "disableRetries": True,
            },
        },
        "labels": {
            "bookforge-run": inputs.run_id[:63],
            "config-sha": inputs.config_sha256[:16],
            "dataset-sha": inputs.dataset_manifest_sha256[:16],
            "prepared-sha": inputs.prepared_train_sha256[:16],
            "inputs-sha": inputs.input_manifest_sha256[:16],
            "attempt": "one",
        },
    }


def approval_token(run_id: str, spec_sha256: str) -> str:
    return f"APPROVE_GCP_JAX_RUN:{run_id}:{spec_sha256}"


def build_plan(inputs: JobInputs) -> dict[str, object]:
    body = build_custom_job(inputs)
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    spec_sha256 = hashlib.sha256(canonical).hexdigest()
    input_bindings = {
        "config_sha256": inputs.config_sha256,
        "dataset_manifest_sha256": inputs.dataset_manifest_sha256,
        "prepared_train_sha256": inputs.prepared_train_sha256,
        "input_manifest_sha256": inputs.input_manifest_sha256,
        "base_checkpoint_manifest_sha256": inputs.base_checkpoint_manifest_sha256,
        "base_checkpoint_receipt_sha256": inputs.base_checkpoint_receipt_sha256,
        "tokenizer_manifest_sha256": inputs.tokenizer_manifest_sha256,
    }
    input_bindings_sha256 = hashlib.sha256(
        json.dumps(input_bindings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "1.0",
        "mode": "plan-only",
        "backend": "vertex-custom-job",
        "project": PROJECT_ID,
        "region": REGION,
        "run_id": inputs.run_id,
        "resource": {
            "machine_type": MACHINE_TYPE,
            "tpu_chips": TPU_CHIPS,
            "replicas": REPLICAS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "automatic_retries": 0,
            "endpoint_created": False,
        },
        "gross_ceiling_usd": GROSS_CEILING_USD,
        "gross_ceiling_policy": "declared-estimate-not-provider-enforced",
        "required_preflight": {
            "active_project": PROJECT_ID,
            "billing_enabled": True,
            "promotional_credits_verified": True,
            "job_id_absent": True,
        },
        "spec_sha256": spec_sha256,
        "input_bindings": input_bindings,
        "input_bindings_sha256": input_bindings_sha256,
        "approval_token": approval_token(inputs.run_id, spec_sha256),
        "create_url": (
            f"https://{REGION}-aiplatform.googleapis.com/v1/projects/"
            f"{PROJECT_ID}/locations/{REGION}/customJobs"
        ),
        "custom_job": body,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--prepared-train-sha256", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--base-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--base-checkpoint-receipt-sha256", required=True)
    parser.add_argument("--tokenizer-manifest-sha256", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--scratch-uri", required=True)
    parser.add_argument("--release-uri", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = build_plan(
        JobInputs(
            run_id=args.run_id,
            image_uri=args.image_uri,
            config_sha256=args.config_sha256,
            dataset_manifest_sha256=args.dataset_manifest_sha256,
            prepared_train_sha256=args.prepared_train_sha256,
            input_manifest_sha256=args.input_manifest_sha256,
            base_checkpoint_manifest_sha256=args.base_checkpoint_manifest_sha256,
            base_checkpoint_receipt_sha256=args.base_checkpoint_receipt_sha256,
            tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
            service_account=args.service_account,
            scratch_uri=args.scratch_uri,
            release_uri=args.release_uri,
            smoke=args.smoke,
        )
    )
    rendered = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
