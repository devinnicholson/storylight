#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-gcp-project}"
REGION="${BOOKFORGE_GKE_REGION:-us-central1}"
CLUSTER="${BOOKFORGE_GKE_CLUSTER:-bookforge-anticipatory}"
RENDERER_REGION="${BOOKFORGE_RENDERER_REGION:-us-central1}"
RENDERER_SERVICE="${BOOKFORGE_RENDERER_SERVICE:-bookforge-scene-rtx}"

for command in gcloud jq; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command is missing: ${command}" >&2
    exit 1
  fi
done

REGION_JSON="$(
  gcloud compute regions describe "${REGION}" \
    --project "${PROJECT_ID}" \
    --format json
)"
L4_LIMIT="$(jq -r '[.quotas[] | select(.metric == "NVIDIA_L4_GPUS") | .limit] | first // 0' <<<"${REGION_JSON}")"
L4_USAGE="$(jq -r '[.quotas[] | select(.metric == "NVIDIA_L4_GPUS") | .usage] | first // 0' <<<"${REGION_JSON}")"
L4_AVAILABLE="$(jq -nr --arg limit "${L4_LIMIT}" --arg usage "${L4_USAGE}" '$limit | tonumber - ($usage | tonumber)')"

RENDERER_URL="$(
  gcloud run services describe "${RENDERER_SERVICE}" \
    --project "${PROJECT_ID}" \
    --region "${RENDERER_REGION}" \
    --format 'value(status.url)' 2>/dev/null || true
)"

CONTAINER_ENABLED="$(
  gcloud services list \
    --project "${PROJECT_ID}" \
    --enabled \
    --filter 'config.name=container.googleapis.com' \
    --format 'value(config.name)'
)"

CLUSTER_STATE="absent"
if [[ -n "${CONTAINER_ENABLED}" ]]; then
  CLUSTER_STATE="$(
    gcloud container clusters describe "${CLUSTER}" \
      --project "${PROJECT_ID}" \
      --region "${REGION}" \
      --format 'value(status)' 2>/dev/null || echo absent
  )"
fi

printf '%s\n' \
  "Bookforge anticipatory GKE preflight" \
  "  project:             ${PROJECT_ID}" \
  "  region:              ${REGION}" \
  "  L4 quota:            ${L4_LIMIT}" \
  "  L4 in use:           ${L4_USAGE}" \
  "  L4 available:        ${L4_AVAILABLE}" \
  "  cluster:             ${CLUSTER} (${CLUSTER_STATE})" \
  "  renderer region:     ${RENDERER_REGION}" \
  "  renderer URL:        ${RENDERER_URL:-missing}" \
  "  container API:       ${CONTAINER_ENABLED:-disabled}"

if ! jq -en --arg available "${L4_AVAILABLE}" '$available | tonumber >= 1' >/dev/null; then
  echo "Preflight failed: one free NVIDIA L4 is required in ${REGION}." >&2
  exit 2
fi
if [[ -z "${RENDERER_URL}" ]]; then
  echo "Preflight failed: private Cloud Run renderer was not found." >&2
  exit 3
fi
