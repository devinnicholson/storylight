# Google Cloud development environment

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

Delete the GPU deployment whenever it is not being tested:

```bash
kubectl -n bookforge delete deployment gemma-vllm
```

Deleting the full cluster stops cluster and workload charges:

```bash
gcloud container clusters delete bookforge-dev --region us-central1
```

The E2B/L4 deployment is a plumbing and evaluation baseline. Once the complete Story Pack schema
is stable, benchmark a larger Gemma model for cloud compilation instead of increasing model size
before the output can be measured.

