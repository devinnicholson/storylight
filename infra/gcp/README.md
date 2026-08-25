# Google Cloud deployment

## Current live-scene path: private Cloud Run GPU

The production migration keeps the privacy and latency split explicit:

- Jetson Gemma converts the passage into bounded visual direction locally. Raw reading text,
  microphone audio, and camera frames do not enter the renderer request.
- A private Cloud Run service uses one NVIDIA L4 for pinned SANA-Sprint master generation and
  Depth Anything V2. It scales from zero to at most one instance and accepts only IAM-authenticated
  requests.
- Nemotron is a separate asynchronous visual critic. It is not placed on the first-image critical
  path and must not be presented as pixel-aware until its multimodal deployment is enabled.
- The Jetson validates checksum-addressed output and performs depth-aware motion locally.

The guarded deployment is in `cloud-run/deploy-live-scene.sh`. It fails before building if the
regional non-zonal L4 quota is below one, pins the deployed revision to an immutable image digest,
and retains only two recent images while deleting images older than 30 days.

```bash
export GOOGLE_CLOUD_PROJECT=your-gcp-project
export BOOKFORGE_GCP_REGION=us-central1
export BOOKFORGE_IMAGE_TAG=20260824-2

# Inspect the exact scope; creates nothing.
./infra/gcp/cloud-run/deploy-live-scene.sh

# Explicitly authorize the bounded billable deployment.
export BOOKFORGE_GCP_APPLY=I_UNDERSTAND_THIS_CREATES_BILLABLE_RESOURCES
./infra/gcp/cloud-run/deploy-live-scene.sh
```

The default service configuration is one RTX PRO 6000, 20 vCPU, 80 GiB, concurrency one, minimum
zero, maximum one, and no unauthenticated access. RTX quota is measured in milliGPUs, so exactly one
GPU requires a quota value of 1,000. Set `BOOKFORGE_GCP_GPU_TYPE=nvidia-l4` to use the lower-cost
8-vCPU/32-GiB L4 profile instead. The renderer service account receives no project-wide role. The
operator account receives only `roles/run.invoker` on this service. For local keyless invocation,
grant the operator the narrower `roles/iam.serviceAccountOpenIdTokenCreator` role on this dedicated
account and grant the account `roles/run.invoker` on the service; no service-account key is created.

Run Bookforge against the deployed URL with Application Default Credentials:

```bash
export BOOKFORGE_LIVE_SCENE_BACKEND=gcp_cloud_run
export BOOKFORGE_LIVE_SCENE_GCP_URL='https://SERVICE_HASH.us-central1.run.app'
export BOOKFORGE_LIVE_SCENE_GCP_AUDIENCE="${BOOKFORGE_LIVE_SCENE_GCP_URL}"
export BOOKFORGE_LIVE_SCENE_GCP_IMPERSONATE_SERVICE_ACCOUNT='bookforge-renderer@your-gcp-project.iam.gserviceaccount.com'
export BOOKFORGE_LIVE_SCENE_GCP_GPU=RTX_PRO_6000
export BOOKFORGE_LIVE_SCENE_ENABLE_MOTION=false
export BOOKFORGE_LIVE_SCENE_ENABLE_PREVIEW=false
```

Cloud Run verifies the Google-signed identity token; the browser and Jetson projector never receive
GCP credentials or the private service URL. Keep minimum instances at zero outside a supervised
demo. The project-scoped billing guard below remains the last-resort containment layer.

The 2026-08-25 RTX deployment is control-plane healthy, but its default HTTPS endpoint is affected
by an upstream Google Frontend route-provisioning defect: both URL forms return a 1,568-byte Google
404, and the revision has zero request logs. Do not prewarm or generate until `/healthz` reaches the
worker. Evidence and the exact immutable revision are recorded in
[`benchmarks/gcp-rtx-cloud-run-deployment-2026-08-25.json`](../../benchmarks/gcp-rtx-cloud-run-deployment-2026-08-25.json).

### Nemotron visual critic

Bookforge's optional critic client targets NVIDIA's OpenAI-compatible
[`llama-3.1-nemotron-nano-vl-8b-v1`](https://build.nvidia.com/nvidia/llama-3.1-nemotron-nano-vl-8b-v1)
NIM. It runs only after `master_ready`, so a critic cold start cannot delay the first projected image.
For a private GCP-hosted NIM, configure its IAM-authenticated URL and audience:

```bash
export BOOKFORGE_LIVE_SCENE_CRITIC_BACKEND=nemotron
export BOOKFORGE_LIVE_SCENE_CRITIC_URL='https://NEMOTRON_HASH.us-central1.run.app'
export BOOKFORGE_LIVE_SCENE_CRITIC_AUDIENCE="${BOOKFORGE_LIVE_SCENE_CRITIC_URL}"

curl -sS -X POST \
  "http://127.0.0.1:8080/v1/live-scenes/SCENE_JOB_ID:critique"
```

The endpoint retrieves the already-generated master from the local checksum cache. Its request
schema contains only the synthetic image, the privacy-gated visual brief, expected subjects, and
forbidden visual content. A deterministic/fallback scene is rejected rather than sent because such
legacy packs may still contain source text. NIM deployment is a separate resource and requires an
NGC API key plus appropriate GPU quota; do not put either credential in browser configuration or
commit it to the repository.

## Legacy compiler experiment: GKE Gemma

This creates the first cloud Story Compiler path:

- a GKE Autopilot cluster;
- one NVIDIA L4 workload serving `google/gemma-4-e2b-it` with vLLM;
- the Bookforge API pointing at vLLM's OpenAI-compatible API;
- Artifact Registry for the API image;
- a Cloud Storage bucket reserved for versioned Story Packs.

Nothing in this directory has been applied automatically. GKE and L4 workloads are billable.

## Prerequisites

1. Select or create a billing-enabled Google Cloud project.
2. Run `gcloud auth login` and `gcloud auth application-default login`.
3. Accept the Gemma license on Hugging Face and create a read-only token.
4. Check L4 quota and availability in the selected region.

## Bootstrap with a cost guard

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export BOOKFORGE_GCP_REGION="us-central1"

# Prints the resources without creating them.
./infra/gcp/bootstrap.sh

# Explicitly unlocks billable creation.
export BOOKFORGE_GCP_APPLY="I_UNDERSTAND_THIS_CREATES_BILLABLE_RESOURCES"
./infra/gcp/bootstrap.sh
```

## Build and deploy

```bash
export HF_TOKEN="hf_read_only_token"
./infra/gcp/deploy.sh

kubectl -n bookforge rollout status deployment/gemma-vllm --timeout=20m
kubectl -n bookforge port-forward service/bookforge-api 8080:8080
```

Then call `http://127.0.0.1:8080/v1/story-packs:compile` using the example request in
`examples/moon-gate.request.json`.

## Cost containment

The contest project also has enforced service caps plus a project-scoped $10 gross-cost emergency
billing disconnect. Its reviewed source and threat boundary are documented in
`infra/gcp/billing-kill-switch/README.md`. Budget notifications are asynchronous, so this guard is
not a promise of a mathematically exact $10 ceiling.

Delete the GPU deployment whenever it is not being tested:

```bash
kubectl -n bookforge delete deployment gemma-vllm
```

Deleting the full cluster stops cluster and workload charges:

```bash
gcloud container clusters delete bookforge-dev --region us-central1
```

The GKE E2B/L4 deployment is retained only as a plumbing and evaluation baseline. The accepted
architecture now keeps Gemma on the Jetson for privacy and uses Cloud Run only for visual rendering.
Once the complete Story Pack schema
is stable, benchmark a larger Gemma model for cloud compilation instead of increasing model size
before the output can be measured.
