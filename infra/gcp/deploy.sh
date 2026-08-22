#!/usr/bin/env bash
set -euo pipefail

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
: "${HF_TOKEN:?Set HF_TOKEN after accepting the Gemma license on Hugging Face}"

BOOKFORGE_GCP_REGION="${BOOKFORGE_GCP_REGION:-us-central1}"
BOOKFORGE_GCP_REPOSITORY="${BOOKFORGE_GCP_REPOSITORY:-bookforge}"
BOOKFORGE_IMAGE="${BOOKFORGE_GCP_REGION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT}/${BOOKFORGE_GCP_REPOSITORY}/api:latest"

gcloud builds submit --tag "${BOOKFORGE_IMAGE}" .

kubectl apply -f infra/gcp/k8s/namespace.yaml
kubectl -n bookforge create secret generic hf-token \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  --dry-run=client \
  -o yaml | kubectl apply -f -
kubectl apply -f infra/gcp/k8s/gemma-e2b.yaml
kubectl apply -f infra/gcp/k8s/api.yaml
kubectl -n bookforge set image deployment/bookforge-api api="${BOOKFORGE_IMAGE}"

