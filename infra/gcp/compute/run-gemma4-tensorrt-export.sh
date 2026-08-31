#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_ID="${BOOKFORGE_GCP_PROJECT:-your-gcp-project}"
readonly ZONE="${BOOKFORGE_GCP_EXPORT_ZONE:-us-central1-b}"
readonly INSTANCE_NAME="${BOOKFORGE_GCP_EXPORT_INSTANCE:-bookforge-gemma4-trt-export}"
readonly MACHINE_TYPE="${BOOKFORGE_GCP_EXPORT_MACHINE_TYPE:-g4-standard-48}"
readonly SERVICE_ACCOUNT="bookforge-tensorrt-export@$PROJECT_ID.iam.gserviceaccount.com"
readonly EXPORT_BUCKET="${BOOKFORGE_GCP_EXPORT_BUCKET:-$PROJECT_ID-tensorrt-edge-llm}"
readonly EXPORT_IMAGE="${BOOKFORGE_GCP_EXPORT_IMAGE:-us-central1-docker.pkg.dev/$PROJECT_ID/bookforge/gemma4-tensorrt-export@sha256:23ffd14e65e7c6ecdb4cdff4034f6e822328ecb59c7edc9a1f45708b4d617f4b}"
readonly RUN_ID="${BOOKFORGE_GCP_EXPORT_RUN_ID:-compute-g4-$(date -u +%Y%m%d-%H%M%S)}"
readonly LOG_OBJECT="diagnostics/compute/$RUN_ID/startup.log"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly STARTUP_SCRIPT="$SCRIPT_DIR/gemma4-tensorrt-export-startup.sh"
readonly APPLY_TOKEN="${BOOKFORGE_GCP_EXPORT_APPLY:-}"
readonly REQUIRED_APPLY_TOKEN="I_UNDERSTAND_THIS_CREATES_A_FINITE_BILLABLE_G4_VM"

if [[ "$PROJECT_ID" != "your-gcp-project" ]]; then
  echo "Refusing unexpected GCP project: $PROJECT_ID" >&2
  exit 2
fi
if [[ ! "$RUN_ID" =~ ^[a-z0-9][a-z0-9-]{0,95}$ ]]; then
  echo "BOOKFORGE_GCP_EXPORT_RUN_ID must be a lowercase bounded slug" >&2
  exit 2
fi
if [[ ! "$EXPORT_IMAGE" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "BOOKFORGE_GCP_EXPORT_IMAGE must be pinned to an immutable sha256 digest" >&2
  exit 2
fi
if [[ ! -f "$STARTUP_SCRIPT" ]]; then
  echo "Startup script is missing: $STARTUP_SCRIPT" >&2
  exit 2
fi
if gcloud compute instances describe "$INSTANCE_NAME" --zone "$ZONE" --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "Refusing to replace existing instance: $INSTANCE_NAME" >&2
  exit 2
fi

if [[ "$APPLY_TOKEN" != "$REQUIRED_APPLY_TOKEN" ]]; then
  cat <<EOF
Dry run only; no resources were created.
Project: $PROJECT_ID
Instance: $INSTANCE_NAME
Zone: $ZONE
Machine: $MACHINE_TYPE (one NVIDIA RTX PRO 6000, 96 GB VRAM)
Image: $EXPORT_IMAGE
Run ID: $RUN_ID
Hard lifetime: 45 minutes with automatic deletion.
Published list-price VM ceiling for 45 minutes: approximately \$3.38, plus bounded disk/network/storage.

To launch exactly one attempt, set:
BOOKFORGE_GCP_EXPORT_APPLY=$REQUIRED_APPLY_TOKEN
EOF
  exit 0
fi

gcloud compute instances create "$INSTANCE_NAME" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --machine-type "$MACHINE_TYPE" \
  --image-family common-cu129-ubuntu-2204-nvidia-580 \
  --image-project deeplearning-platform-release \
  --boot-disk-size 200GB \
  --boot-disk-type pd-balanced \
  --maintenance-policy TERMINATE \
  --no-restart-on-failure \
  --max-run-duration 45m \
  --instance-termination-action DELETE \
  --service-account "$SERVICE_ACCOUNT" \
  --scopes cloud-platform \
  --labels bookforge-purpose=tensorrt-export,bookforge-finite=true \
  --metadata-from-file "startup-script=$STARTUP_SCRIPT" \
  --metadata \
    "bookforge-export-image=$EXPORT_IMAGE,bookforge-export-bucket=$EXPORT_BUCKET,bookforge-export-run-id=$RUN_ID,bookforge-export-log-object=$LOG_OBJECT"

cat <<EOF
Started finite Gemma 4 TensorRT export.
Instance: $INSTANCE_NAME
Zone: $ZONE
Machine: $MACHINE_TYPE
Run ID: $RUN_ID
Hard lifetime: 45 minutes, then Compute Engine deletes the VM.
Delete the VM sooner after observing either the completion manifest or a terminal startup log.
Log: gs://$EXPORT_BUCKET/$LOG_OBJECT
Manifest: gs://$EXPORT_BUCKET/tensorrt-edge-llm/gemma4-e2b-it-int4-awq-v010/$RUN_ID/export.manifest.json
EOF
