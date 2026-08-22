#!/usr/bin/env bash
set -euo pipefail

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to the billing-enabled project ID}"

BOOKFORGE_GCP_REGION="${BOOKFORGE_GCP_REGION:-us-central1}"
BOOKFORGE_GCP_CLUSTER="${BOOKFORGE_GCP_CLUSTER:-bookforge-dev}"
BOOKFORGE_GCP_REPOSITORY="${BOOKFORGE_GCP_REPOSITORY:-bookforge}"
BOOKFORGE_GCP_BUCKET="${BOOKFORGE_GCP_BUCKET:-${GOOGLE_CLOUD_PROJECT}-bookforge-story-packs}"

if [[ "${BOOKFORGE_GCP_APPLY:-}" != "I_UNDERSTAND_THIS_CREATES_BILLABLE_RESOURCES" ]]; then
  echo "Dry guard active. This script would configure project ${GOOGLE_CLOUD_PROJECT}:"
  echo "  region:     ${BOOKFORGE_GCP_REGION}"
  echo "  cluster:    ${BOOKFORGE_GCP_CLUSTER} (GKE Autopilot)"
  echo "  repository: ${BOOKFORGE_GCP_REPOSITORY}"
  echo "  bucket:     gs://${BOOKFORGE_GCP_BUCKET}"
  echo
  echo "Set BOOKFORGE_GCP_APPLY=I_UNDERSTAND_THIS_CREATES_BILLABLE_RESOURCES to apply."
  exit 2
fi

gcloud config set project "${GOOGLE_CLOUD_PROJECT}"

gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  container.googleapis.com \
  storage.googleapis.com

gcloud artifacts repositories describe "${BOOKFORGE_GCP_REPOSITORY}" \
  --location "${BOOKFORGE_GCP_REGION}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${BOOKFORGE_GCP_REPOSITORY}" \
  --repository-format docker \
  --location "${BOOKFORGE_GCP_REGION}" \
  --description "Bookforge service images"

gcloud storage buckets describe "gs://${BOOKFORGE_GCP_BUCKET}" >/dev/null 2>&1 || \
gcloud storage buckets create "gs://${BOOKFORGE_GCP_BUCKET}" \
  --location "${BOOKFORGE_GCP_REGION}" \
  --uniform-bucket-level-access

gcloud container clusters describe "${BOOKFORGE_GCP_CLUSTER}" \
  --region "${BOOKFORGE_GCP_REGION}" >/dev/null 2>&1 || \
gcloud container clusters create-auto "${BOOKFORGE_GCP_CLUSTER}" \
  --region "${BOOKFORGE_GCP_REGION}" \
  --release-channel regular

gcloud container clusters get-credentials "${BOOKFORGE_GCP_CLUSTER}" \
  --region "${BOOKFORGE_GCP_REGION}"

