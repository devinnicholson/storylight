#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-gcp-project}"
REGION="${STORYLIGHT_GKE_REGION:-us-central1}"
CLUSTER="${STORYLIGHT_GKE_CLUSTER:-storylight-anticipatory}"

if [[ "${STORYLIGHT_GKE_SUSPEND:-}" != "I_UNDERSTAND_THIS_STOPS_THE_GKE_GPU" ]]; then
  printf '%s\n' \
    "Dry guard active. This would scale storylight-nemotron to zero in:" \
    "  project: ${PROJECT_ID}" \
    "  cluster: ${CLUSTER} (${REGION})" \
    "" \
    "Set STORYLIGHT_GKE_SUSPEND=I_UNDERSTAND_THIS_STOPS_THE_GKE_GPU to apply."
  exit 2
fi

gcloud container clusters get-credentials "${CLUSTER}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}"
kubectl -n storylight scale deployment/storylight-nemotron --replicas=0
kubectl -n storylight rollout status deployment/storylight-nemotron --timeout=10m
echo "Storylight anticipatory GPU workload is scaled to zero."
