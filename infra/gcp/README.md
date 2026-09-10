# Google Cloud deployment

Storylight uses Google Cloud for authenticated image generation, optional visual criticism, and finite model-export work. Speech, reader identity, raw transcripts, and projector control stay on local hardware in the documented deployment.

## Live image generation

The private Cloud Run worker lives in [`deploy/gcp_live_scene_worker`](../../deploy/gcp_live_scene_worker/). The guarded deployment script is [`cloud-run/deploy-live-scene.sh`](cloud-run/deploy-live-scene.sh). It performs a read-only preflight unless the operator supplies the explicit billable-resource authorization variable.

```bash
export GOOGLE_CLOUD_PROJECT=your-gcp-project
export STORYLIGHT_GCP_REGION=us-central1
export STORYLIGHT_IMAGE_TAG=YYYYMMDD-N

# Read-only preflight.
./infra/gcp/cloud-run/deploy-live-scene.sh

# Explicit billable deployment.
export STORYLIGHT_GCP_APPLY=I_UNDERSTAND_THIS_CREATES_BILLABLE_RESOURCES
./infra/gcp/cloud-run/deploy-live-scene.sh
```

The service is IAM-authenticated, scales from zero, permits one GPU worker, and does not grant unauthenticated access. Runtime credentials stay on the Jetson or server process:

```bash
export STORYLIGHT_LIVE_SCENE_BACKEND=gcp_cloud_run
export STORYLIGHT_LIVE_SCENE_GCP_URL='https://SERVICE_HASH.REGION.run.app'
export STORYLIGHT_LIVE_SCENE_GCP_AUDIENCE="$STORYLIGHT_LIVE_SCENE_GCP_URL"
export STORYLIGHT_LIVE_SCENE_GCP_IMPERSONATE_SERVICE_ACCOUNT='storylight-renderer@your-gcp-project.iam.gserviceaccount.com'
export STORYLIGHT_LIVE_SCENE_ENABLE_MOTION=false
```

The browser never receives the service URL, identity token, or service-account credentials. Returned assets are checksum-verified before entering the local cache.

## Managed Vertex route and safe fallback

The resilient provider can use a private Cloud Run renderer, managed Vertex image generation, and a budget-checked Modal fallback.

```bash
export STORYLIGHT_LIVE_SCENE_BACKEND=gcp_resilient
export STORYLIGHT_LIVE_SCENE_VERTEX_PROJECT_ID=your-gcp-project
export STORYLIGHT_LIVE_SCENE_VERTEX_LOCATION=global
export STORYLIGHT_LIVE_SCENE_VERTEX_MODEL=gemini-3.1-flash-lite-image
export STORYLIGHT_LIVE_SCENE_VERTEX_SESSION_COST_CAP_USD=0.50
export STORYLIGHT_LIVE_SCENE_ROUTING_FAILURE_COOLDOWN_SECONDS=300
```

Fallback is allowed only after readiness fails or a provider explicitly reports that generation did not begin. A timeout or disconnect after generation starts is billably ambiguous and terminal; Storylight does not automatically create a second image through another provider.

## Nemotron visual critic

The optional critic uses NVIDIA Llama 3.1 Nemotron Nano VL through NIM. The GKE service is defined in [`deploy/gke_anticipatory`](../../deploy/gke_anticipatory/) and [`k8s/anticipatory.yaml`](k8s/anticipatory.yaml).

Run the read-only preflight first:

```bash
./infra/gcp/gke/preflight-anticipatory.sh
./infra/gcp/gke/deploy-anticipatory.sh
```

The deployment script requires a separate explicit authorization before creating billable resources. The service receives a bounded visual brief and generated artwork, excluding microphone audio and the original transcript. It is optional and does not sit on the default first-image path.

Configure the client with server-side values:

```bash
export STORYLIGHT_LIVE_SCENE_CRITIC_BACKEND=nemotron
export STORYLIGHT_LIVE_SCENE_CRITIC_URL='https://NEMOTRON_SERVICE_URL'
export STORYLIGHT_LIVE_SCENE_CRITIC_AUDIENCE="$STORYLIGHT_LIVE_SCENE_CRITIC_URL"
```

## Finite Gemma 4 TensorRT export

The Compute Engine launcher in [`compute/run-gemma4-tensorrt-export.sh`](compute/run-gemma4-tensorrt-export.sh) creates one deadline-bounded export VM. It uses an immutable exporter image, uploads its completion manifest last, and does not change production routing.

```bash
export GOOGLE_CLOUD_PROJECT=your-gcp-project
export STORYLIGHT_GCP_REGION=us-central1
export STORYLIGHT_GCP_EXPORT_APPLY=I_UNDERSTAND_THIS_CREATES_A_FINITE_BILLABLE_G4_VM
./infra/gcp/compute/run-gemma4-tensorrt-export.sh
```

Use a dedicated service account with access only to the required Artifact Registry repository and private export bucket. Delete the VM as soon as the completion manifest or terminal diagnostic appears. The automatic deadline is a final containment mechanism, not a substitute for supervised cleanup.

The exported checkpoint remains an experimental planner candidate. It must pass target-device memory, grammar, grounding, privacy, refusal, and latency gates before it can replace the deterministic compiler.

## Monitoring and cost containment

[`monitoring/live-scene-dashboard.json`](monitoring/live-scene-dashboard.json) contains a Cloud Monitoring dashboard based on built-in Cloud Run metrics. It does not use prompts, transcripts, session IDs, audio, or camera data as dimensions.

```bash
gcloud monitoring dashboards create \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --config-from-file infra/gcp/monitoring/live-scene-dashboard.json
```

The billing kill switch is documented in [`billing-kill-switch/README.md`](billing-kill-switch/README.md). Budget notifications are asynchronous and cannot guarantee an exact ceiling. Keep minimum instances at zero, use explicit sample and time limits, and tear down GPU resources immediately after a supervised run.

## Trust boundary

- Credentials belong in local environment files or a secret manager.
- Browser code must never receive cloud credentials or private provider URLs.
- Cloud requests contain reviewed scene contracts, never microphone recordings.
- Generated assets must pass local checksum and schema validation.
- Paid prewarm, deployment, export, and benchmark actions require explicit operator authorization.

See [Architecture](../../docs/architecture.md), [Privacy and data flow](../../docs/privacy.md), and [Research results](../../docs/research-results.md) for the runtime boundary and measured limitations.
