#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-gcp-project}"
REGION="${BOOKFORGE_GKE_REGION:-us-central1}"
CLUSTER="${BOOKFORGE_GKE_CLUSTER:-bookforge-anticipatory}"

if [[ "${BOOKFORGE_GKE_DELETE:-}" != "I_UNDERSTAND_THIS_DELETES_THE_ANTICIPATORY_CLUSTER" ]]; then
  printf '%s\n' \
    "Dry guard active. This would permanently delete only:" \
    "  project: ${PROJECT_ID}" \
    "  cluster: ${CLUSTER} (${REGION})" \
    "" \
    "The Cloud Run renderer, Artifact Registry images, and local evidence remain." \
    "Set BOOKFORGE_GKE_DELETE=I_UNDERSTAND_THIS_DELETES_THE_ANTICIPATORY_CLUSTER to apply."
  exit 2
fi

gcloud container clusters describe "${CLUSTER}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" >/dev/null

gcloud container clusters delete "${CLUSTER}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --quiet

echo "Deleted GKE cluster ${CLUSTER} in ${REGION}; persistent cluster charges are stopped."
