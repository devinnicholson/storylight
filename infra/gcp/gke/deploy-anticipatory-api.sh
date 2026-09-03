#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-gcp-project}"
REGION="${BOOKFORGE_GKE_REGION:-us-central1}"
CLUSTER="${BOOKFORGE_GKE_CLUSTER:-bookforge-anticipatory}"
REPOSITORY="${BOOKFORGE_GCP_REPOSITORY:-bookforge}"
IMAGE_TAG="${BOOKFORGE_ANTICIPATORY_IMAGE_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/anticipatory:${IMAGE_TAG}"
NAMESPACE="bookforge"
API_DEPLOYMENT="bookforge-anticipatory"
GPU_DEPLOYMENT="bookforge-nemotron"
RENDERER_REGION="${BOOKFORGE_RENDERER_REGION:-us-central1}"
RENDERER_SERVICE="${BOOKFORGE_RENDERER_SERVICE:-bookforge-scene-rtx}"

if [[ "${BOOKFORGE_GKE_API_APPLY:-}" != "I_UNDERSTAND_THIS_RUNS_A_BILLABLE_CLOUD_BUILD" ]]; then
  printf '%s\n' \
    "Dry guard active. This would build and release only the CPU API:" \
    "  project:        ${PROJECT_ID}" \
    "  cluster:        ${CLUSTER} (${REGION})" \
    "  API deployment: ${API_DEPLOYMENT}" \
    "  GPU invariant:  ${GPU_DEPLOYMENT} replica count is never changed" \
    "" \
    "Set BOOKFORGE_GKE_API_APPLY=I_UNDERSTAND_THIS_RUNS_A_BILLABLE_CLOUD_BUILD to apply."
  exit 2
fi

for command in gcloud kubectl; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Required command is missing: ${command}" >&2
    exit 1
  fi
done

if ! command -v gke-gcloud-auth-plugin >/dev/null 2>&1; then
  GCLOUD_SDK_ROOT="$(gcloud info --format='value(installation.sdk_root)')"
  if [[ -x "${GCLOUD_SDK_ROOT}/bin/gke-gcloud-auth-plugin" ]]; then
    export PATH="${GCLOUD_SDK_ROOT}/bin:${PATH}"
  else
    echo "Required command is missing: gke-gcloud-auth-plugin" >&2
    exit 1
  fi
fi

gcloud container clusters get-credentials "${CLUSTER}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}"

for deployment in "${API_DEPLOYMENT}" "${GPU_DEPLOYMENT}"; do
  if ! kubectl -n "${NAMESPACE}" get "deployment/${deployment}" >/dev/null 2>&1; then
    echo "Missing deployment/${deployment}; run deploy-anticipatory.sh once first." >&2
    exit 1
  fi
done

GPU_REPLICAS_BEFORE="$(
  kubectl -n "${NAMESPACE}" get "deployment/${GPU_DEPLOYMENT}" \
    -o jsonpath='{.spec.replicas}'
)"

gcloud builds submit . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --ignore-file deploy/gke_anticipatory/.gcloudignore \
  --config infra/gcp/gke/cloudbuild-anticipatory.yaml \
  --substitutions "_IMAGE=${IMAGE}"

IMAGE_DIGEST="$(
  gcloud artifacts docker images describe "${IMAGE}" \
    --project "${PROJECT_ID}" \
    --format 'value(image_summary.digest)'
)"
if [[ ! "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Could not resolve an immutable anticipatory image digest." >&2
  exit 1
fi
IMMUTABLE_IMAGE="${IMAGE%:*}@${IMAGE_DIGEST}"
RENDERER_URL="$(
  gcloud run services describe "${RENDERER_SERVICE}" \
    --project "${PROJECT_ID}" \
    --region "${RENDERER_REGION}" \
    --format 'value(status.url)'
)"

kubectl -n "${NAMESPACE}" set image "deployment/${API_DEPLOYMENT}" \
  "anticipatory-api=${IMMUTABLE_IMAGE}"
kubectl -n "${NAMESPACE}" set env "deployment/${API_DEPLOYMENT}" \
  "BOOKFORGE_RENDERER_URL=${RENDERER_URL}" \
  "BOOKFORGE_RENDERER_AUDIENCE=${RENDERER_URL}"
kubectl -n "${NAMESPACE}" rollout status "deployment/${API_DEPLOYMENT}" --timeout=5m

GPU_REPLICAS_AFTER="$(
  kubectl -n "${NAMESPACE}" get "deployment/${GPU_DEPLOYMENT}" \
    -o jsonpath='{.spec.replicas}'
)"
if [[ "${GPU_REPLICAS_AFTER}" != "${GPU_REPLICAS_BEFORE}" ]]; then
  echo "GPU replica invariant changed unexpectedly; inspect the cluster immediately." >&2
  exit 1
fi

printf '%s\n' \
  "Bookforge CPU API release is ready." \
  "Image: ${IMMUTABLE_IMAGE}" \
  "Nemotron replicas remained ${GPU_REPLICAS_AFTER}."
