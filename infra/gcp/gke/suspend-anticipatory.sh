#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-gcp-project}"
REGION="${BOOKFORGE_GKE_REGION:-us-central1}"
CLUSTER="${BOOKFORGE_GKE_CLUSTER:-bookforge-anticipatory}"

if [[ "${BOOKFORGE_GKE_SUSPEND:-}" != "I_UNDERSTAND_THIS_STOPS_THE_GKE_GPU" ]]; then
  printf '%s\n' \
    "Dry guard active. This would scale bookforge-anticipatory to zero in:" \
    "  project: ${PROJECT_ID}" \
    "  cluster: ${CLUSTER} (${REGION})" \
    "" \
    "Set BOOKFORGE_GKE_SUSPEND=I_UNDERSTAND_THIS_STOPS_THE_GKE_GPU to apply."
  exit 2
fi

gcloud container clusters get-credentials "${CLUSTER}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}"
kubectl -n bookforge scale deployment/bookforge-anticipatory --replicas=0
kubectl -n bookforge rollout status deployment/bookforge-anticipatory --timeout=10m
echo "Bookforge anticipatory GPU workload is scaled to zero."
