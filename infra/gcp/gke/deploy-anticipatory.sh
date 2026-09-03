#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-gcp-project}"
REGION="${BOOKFORGE_GKE_REGION:-us-central1}"
CLUSTER="${BOOKFORGE_GKE_CLUSTER:-bookforge-anticipatory}"
REPOSITORY="${BOOKFORGE_GCP_REPOSITORY:-bookforge}"
IMAGE_TAG="${BOOKFORGE_ANTICIPATORY_IMAGE_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/anticipatory:${IMAGE_TAG}"
GSA="bookforge-anticipator@${PROJECT_ID}.iam.gserviceaccount.com"
KSA="bookforge-anticipator"
NAMESPACE="bookforge"
GPU_DEPLOYMENT="bookforge-nemotron"
API_DEPLOYMENT="bookforge-anticipatory"
RENDERER_REGION="${BOOKFORGE_RENDERER_REGION:-us-central1}"
RENDERER_SERVICE="${BOOKFORGE_RENDERER_SERVICE:-bookforge-scene-rtx}"
MANIFEST="infra/gcp/k8s/anticipatory.yaml"

if [[ "${BOOKFORGE_GKE_APPLY:-}" != "I_UNDERSTAND_THIS_CREATES_BILLABLE_GKE_GPU_RESOURCES" ]]; then
  printf '%s\n' \
    "Dry guard active. This would create or update:" \
    "  project:       ${PROJECT_ID}" \
    "  cluster:       ${CLUSTER} (GKE Autopilot, ${REGION})" \
    "  workload:      one NVIDIA L4 with Nemotron NIM 1.3.1" \
    "  companion:     Bookforge anticipatory API" \
    "  renderer:      ${RENDERER_SERVICE} (${RENDERER_REGION})" \
    "  exposure:      ClusterIP only; no public load balancer" \
    "  failure path:  scale the GPU deployment back to zero" \
    "" \
    "Set BOOKFORGE_GKE_APPLY=I_UNDERSTAND_THIS_CREATES_BILLABLE_GKE_GPU_RESOURCES to apply."
  exit 2
fi

: "${NGC_API_KEY:?Set NGC_API_KEY without writing it into the repository}"

for command in gcloud jq kubectl; do
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
    echo "Install it with: gcloud components install gke-gcloud-auth-plugin" >&2
    exit 1
  fi
fi

./infra/gcp/gke/preflight-anticipatory.sh

GPU_SCALED=0
CLUSTER_CREATED=0
SECRET_DIR=""
cleanup() {
  exit_code=$?
  if (( exit_code != 0 && GPU_SCALED == 1 )); then
    echo "Deployment failed; scaling the GKE GPU workload to zero." >&2
    kubectl -n "${NAMESPACE}" scale "deployment/${GPU_DEPLOYMENT}" --replicas=0 >/dev/null 2>&1 || true
  fi
  if (( exit_code != 0 && CLUSTER_CREATED == 1 )); then
    echo "Deployment failed; deleting the cluster created by this run." >&2
    gcloud container clusters delete "${CLUSTER}" \
      --project "${PROJECT_ID}" \
      --region "${REGION}" \
      --quiet >/dev/null 2>&1 || true
  fi
  if [[ -n "${SECRET_DIR}" && "${SECRET_DIR}" == /tmp/bookforge-ngc.* ]]; then
    rm -f "${SECRET_DIR}/ngc-key" "${SECRET_DIR}/dockerconfigjson"
    rmdir "${SECRET_DIR}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  container.googleapis.com \
  iamcredentials.googleapis.com \
  --project "${PROJECT_ID}"

gcloud artifacts repositories describe "${REPOSITORY}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPOSITORY}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --repository-format docker \
  --description "Bookforge private GPU services"

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

gcloud iam service-accounts describe "${GSA}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud iam service-accounts create bookforge-anticipator \
  --project "${PROJECT_ID}" \
  --display-name "Bookforge GKE anticipatory story engine"

grant_renderer_invoker() {
  local attempt
  for attempt in {1..12}; do
    if gcloud run services add-iam-policy-binding "${RENDERER_SERVICE}" \
      --project "${PROJECT_ID}" \
      --region "${RENDERER_REGION}" \
      --member "serviceAccount:${GSA}" \
      --role roles/run.invoker \
      --quiet; then
      return 0
    fi
    if (( attempt == 12 )); then
      echo "Service-account propagation did not finish after 12 attempts." >&2
      return 1
    fi
    echo "Waiting for the new service account to propagate (${attempt}/12)." >&2
    sleep 5
  done
}

grant_renderer_invoker

if ! gcloud container clusters describe "${CLUSTER}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" >/dev/null 2>&1; then
  gcloud container clusters create-auto "${CLUSTER}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --release-channel regular
  CLUSTER_CREATED=1
fi

gcloud container clusters get-credentials "${CLUSTER}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}"

kubectl apply -f infra/gcp/k8s/namespace.yaml
kubectl apply -f "${MANIFEST}"

gcloud iam service-accounts add-iam-policy-binding "${GSA}" \
  --project "${PROJECT_ID}" \
  --member "serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/${KSA}]" \
  --role roles/iam.workloadIdentityUser \
  --quiet
kubectl -n "${NAMESPACE}" annotate serviceaccount "${KSA}" \
  "iam.gke.io/gcp-service-account=${GSA}" \
  --overwrite

SECRET_DIR="$(mktemp -d /tmp/bookforge-ngc.XXXXXX)"
umask 077
printf '%s' "${NGC_API_KEY}" >"${SECRET_DIR}/ngc-key"
NGC_AUTH="$(printf '%s' "\$oauthtoken:${NGC_API_KEY}" | base64 | tr -d '\n')"
printf '{"auths":{"nvcr.io":{"username":"$oauthtoken","auth":"%s"}}}' \
  "${NGC_AUTH}" >"${SECRET_DIR}/dockerconfigjson"
kubectl -n "${NAMESPACE}" create secret generic ngc-api \
  --from-file="NGC_API_KEY=${SECRET_DIR}/ngc-key" \
  --dry-run=client \
  -o yaml | kubectl apply -f -
kubectl -n "${NAMESPACE}" create secret generic ngc-secret \
  --type=kubernetes.io/dockerconfigjson \
  --from-file=".dockerconfigjson=${SECRET_DIR}/dockerconfigjson" \
  --dry-run=client \
  -o yaml | kubectl apply -f -

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

GPU_SCALED=1
kubectl -n "${NAMESPACE}" scale "deployment/${GPU_DEPLOYMENT}" --replicas=1
kubectl -n "${NAMESPACE}" rollout status "deployment/${GPU_DEPLOYMENT}" --timeout=30m
kubectl -n "${NAMESPACE}" rollout status "deployment/${API_DEPLOYMENT}" --timeout=5m
GPU_SCALED=0

printf '%s\n' \
  "Bookforge anticipatory GKE workload is ready." \
  "Image: ${IMMUTABLE_IMAGE}" \
  "For later CPU-only releases, use infra/gcp/gke/deploy-anticipatory-api.sh." \
  "Open a private local tunnel with:" \
  "  kubectl -n ${NAMESPACE} port-forward service/bookforge-anticipatory 18082:8080" \
  "Suspend the billable GPU immediately after the experiment with:" \
  "  BOOKFORGE_GKE_SUSPEND=I_UNDERSTAND_THIS_STOPS_THE_GKE_GPU ./infra/gcp/gke/suspend-anticipatory.sh"
