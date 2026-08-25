#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-gcp-project}"
REGION="${BOOKFORGE_GCP_REGION:-us-central1}"
REPOSITORY="${BOOKFORGE_GCP_REPOSITORY:-bookforge}"
SERVICE="${BOOKFORGE_GCP_SCENE_SERVICE:-bookforge-live-scene}"
SERVICE_ACCOUNT="bookforge-renderer@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/live-scene:${BOOKFORGE_IMAGE_TAG:-latest}"
QUOTA_ID="NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion"

if [[ "${BOOKFORGE_GCP_APPLY:-}" != "I_UNDERSTAND_THIS_CREATES_BILLABLE_RESOURCES" ]]; then
  echo "Dry guard active. This script would create/update:"
  echo "  project:       ${PROJECT_ID}"
  echo "  region:        ${REGION}"
  echo "  image:         ${IMAGE}"
  echo "  service:       ${SERVICE}"
  echo "  GPU:           one NVIDIA L4, no zonal redundancy"
  echo "  autoscaling:   zero to one instance, concurrency one"
  echo "  access:        IAM authenticated only"
  echo
  echo "Set BOOKFORGE_GCP_APPLY=I_UNDERSTAND_THIS_CREATES_BILLABLE_RESOURCES to apply."
  exit 2
fi

for command in gcloud jq; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command is missing: ${command}" >&2
    exit 1
  fi
done

# Fail before another image build when Cloud Run cannot schedule the one bounded
# instance. A zero-valued regional quota is represented by an empty details
# object, so the absent value must be interpreted as zero.
QUOTA_JSON="$(
  gcloud beta quotas info describe "${QUOTA_ID}" \
    --service run.googleapis.com \
    --project "${PROJECT_ID}" \
    --format json
)"
QUOTA_VALUE="$(
  jq -r --arg region "${REGION}" \
    '[.dimensionsInfos[] | select(.dimensions.region == $region) | ((.details.value // "0") | tonumber)] | max // 0' \
    <<<"${QUOTA_JSON}"
)"
if (( QUOTA_VALUE < 1 )); then
  echo "Cloud Run non-zonal L4 quota is ${QUOTA_VALUE} in ${REGION}; one is required." >&2
  echo "No build or deployment was started." >&2
  exit 3
fi

gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  iam.googleapis.com \
  run.googleapis.com \
  --project "${PROJECT_ID}"

gcloud artifacts repositories describe "${REPOSITORY}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPOSITORY}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --repository-format docker \
  --description "Bookforge private GPU services"

gcloud artifacts repositories set-cleanup-policies "${REPOSITORY}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --policy infra/gcp/cloud-run/live-scene-cleanup-policy.json

gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud iam service-accounts create bookforge-renderer \
  --project "${PROJECT_ID}" \
  --display-name "Bookforge private live-scene renderer"

gcloud builds submit deploy/gcp_live_scene_worker \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --tag "${IMAGE}" \
  --timeout 3600s \
  --machine-type e2-highcpu-8 \
  --disk-size 100

IMAGE_DIGEST="$(
  gcloud artifacts docker images describe "${IMAGE}" \
    --project "${PROJECT_ID}" \
    --format 'value(image_summary.digest)'
)"
if [[ ! "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Could not resolve an immutable digest for ${IMAGE}" >&2
  exit 1
fi
IMMUTABLE_IMAGE="${IMAGE%:*}@${IMAGE_DIGEST}"

gcloud run deploy "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMMUTABLE_IMAGE}" \
  --service-account "${SERVICE_ACCOUNT}" \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu 8 \
  --memory 32Gi \
  --concurrency 1 \
  --min 0 \
  --max 1 \
  --timeout 300s \
  --cpu-boost \
  --no-cpu-throttling \
  --no-allow-unauthenticated \
  --labels app=bookforge,component=live-scene,model=sana-sprint \
  --quiet

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -n 1)"
gcloud run services add-iam-policy-binding "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --member "user:${ACTIVE_ACCOUNT}" \
  --role roles/run.invoker \
  --quiet

gcloud run services describe "${SERVICE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format 'yaml(metadata.name,status.url,status.latestReadyRevisionName,spec.template.metadata.annotations,spec.template.spec.containerConcurrency,spec.template.spec.containers[0].resources)'
