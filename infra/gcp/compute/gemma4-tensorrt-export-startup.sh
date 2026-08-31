#!/usr/bin/env bash
set -Eeuo pipefail

readonly METADATA_ROOT="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
readonly METADATA_HEADER="Metadata-Flavor: Google"
readonly LOCAL_LOG="/var/log/bookforge-gemma4-tensorrt-export.log"

metadata() {
  curl --fail --silent --show-error \
    --header "$METADATA_HEADER" \
    "$METADATA_ROOT/$1"
}

readonly EXPORT_IMAGE="$(metadata bookforge-export-image)"
readonly EXPORT_BUCKET="$(metadata bookforge-export-bucket)"
readonly EXPORT_RUN_ID="$(metadata bookforge-export-run-id)"
readonly LOG_OBJECT="$(metadata bookforge-export-log-object)"

exec > >(tee -a "$LOCAL_LOG") 2>&1

upload_log() {
  gcloud storage cp "$LOCAL_LOG" "gs://$EXPORT_BUCKET/$LOG_OBJECT" || true
}
trap upload_log EXIT

echo '{"event":"bookforge_compute_export_boot","stage":"startup_script"}'
nvidia-smi

if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install --yes --no-install-recommends docker.io
fi
systemctl enable --now docker

if command -v nvidia-ctk >/dev/null 2>&1; then
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker pull "$EXPORT_IMAGE"
docker image inspect "$EXPORT_IMAGE" --format '{{json .RepoDigests}}'

docker run \
  --rm \
  --gpus all \
  --shm-size 16g \
  --name bookforge-gemma4-tensorrt-export \
  --env "BOOKFORGE_EXPORT_BUCKET=$EXPORT_BUCKET" \
  --env "BOOKFORGE_EXPORT_RUN_ID=$EXPORT_RUN_ID" \
  "$EXPORT_IMAGE"

echo '{"event":"bookforge_compute_export_boot","stage":"container_complete"}'
