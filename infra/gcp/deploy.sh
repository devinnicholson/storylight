#!/usr/bin/env bash
set -euo pipefail

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
: "${HF_TOKEN:?Set HF_TOKEN after accepting the Gemma license on Hugging Face}"

STORYLIGHT_GCP_REGION="${STORYLIGHT_GCP_REGION:-us-central1}"
STORYLIGHT_GCP_REPOSITORY="${STORYLIGHT_GCP_REPOSITORY:-storylight}"
STORYLIGHT_IMAGE="${STORYLIGHT_GCP_REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${STORYLIGHT_GCP_REPOSITORY}/api:latest"

gcloud builds submit --tag "${STORYLIGHT_IMAGE}" .

kubectl apply -f infra/gcp/k8s/namespace.yaml
kubectl -n storylight create secret generic hf-token \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  --dry-run=client \
  -o yaml | kubectl apply -f -
kubectl apply -f infra/gcp/k8s/gemma-e2b.yaml
kubectl apply -f infra/gcp/k8s/api.yaml
kubectl -n storylight set image deployment/storylight-api api="${STORYLIGHT_IMAGE}"
