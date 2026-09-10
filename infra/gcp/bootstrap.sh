#!/usr/bin/env bash
set -euo pipefail

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT to the billing-enabled project ID}"

STORYLIGHT_GCP_REGION="${STORYLIGHT_GCP_REGION:-us-central1}"
STORYLIGHT_GCP_CLUSTER="${STORYLIGHT_GCP_CLUSTER:-storylight-dev}"
STORYLIGHT_GCP_REPOSITORY="${STORYLIGHT_GCP_REPOSITORY:-storylight}"
STORYLIGHT_GCP_BUCKET="${STORYLIGHT_GCP_BUCKET:-${GOOGLE_CLOUD_PROJECT}-storylight-story-packs}"

if [[ "${STORYLIGHT_GCP_APPLY:-}" != "I_UNDERSTAND_THIS_CREATES_BILLABLE_RESOURCES" ]]; then
  echo "Dry guard active. This script would configure project ${GOOGLE_CLOUD_PROJECT}:"
  echo "  region:     ${STORYLIGHT_GCP_REGION}"
  echo "  cluster:    ${STORYLIGHT_GCP_CLUSTER} (GKE Autopilot)"
  echo "  repository: ${STORYLIGHT_GCP_REPOSITORY}"
  echo "  bucket:     gs://${STORYLIGHT_GCP_BUCKET}"
  echo
  echo "Set STORYLIGHT_GCP_APPLY=I_UNDERSTAND_THIS_CREATES_BILLABLE_RESOURCES to apply."
  exit 2
fi

gcloud config set project "${GOOGLE_CLOUD_PROJECT}"

gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  container.googleapis.com \
  storage.googleapis.com

gcloud artifacts repositories describe "${STORYLIGHT_GCP_REPOSITORY}" \
  --location "${STORYLIGHT_GCP_REGION}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${STORYLIGHT_GCP_REPOSITORY}" \
  --repository-format docker \
  --location "${STORYLIGHT_GCP_REGION}" \
  --description "Storylight service images"

gcloud storage buckets describe "gs://${STORYLIGHT_GCP_BUCKET}" >/dev/null 2>&1 || \
gcloud storage buckets create "gs://${STORYLIGHT_GCP_BUCKET}" \
  --location "${STORYLIGHT_GCP_REGION}" \
  --uniform-bucket-level-access

gcloud container clusters describe "${STORYLIGHT_GCP_CLUSTER}" \
  --region "${STORYLIGHT_GCP_REGION}" >/dev/null 2>&1 || \
gcloud container clusters create-auto "${STORYLIGHT_GCP_CLUSTER}" \
  --region "${STORYLIGHT_GCP_REGION}" \
  --release-channel regular

gcloud container clusters get-credentials "${STORYLIGHT_GCP_CLUSTER}" \
  --region "${STORYLIGHT_GCP_REGION}"
