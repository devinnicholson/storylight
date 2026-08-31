# Bookforge full-stack optimization program

Status: active, 2026-08-26

Implementation checkpoint: the privacy-safe Cloud Monitoring dashboard is deployed as
`projects/your-gcp-project/dashboards/YOUR_DASHBOARD_ID`; the strict renderer digest
is staged at zero traffic; and `bookforge.gcp_scene_benchmark` now provides bounded probe/prepared
experiments with no automatic retry and prompt hashes instead of prompt text.

Current external blocker: two authenticated, zero-traffic health samples were rejected by the Cloud
Run front end as billing-disabled before the container responded, while the Cloud Billing API
reported account `000000-000000-000000` open and the project linked/enabled. No emergency-guard
disable event exists. Google documents that services may take up to 24 hours to resume after billing
is re-enabled and that some services require a manual restart. Do not spend additional canary calls
inside that propagation window; after it, take one health-only sample and escalate to Cloud Billing
Support if the API/front-end contradiction remains.

Bookforge is no longer bottlenecked by one slow model. The accepted standalone path already has a
faithful Jetson planner, a subsecond prepared renderer, stable 30 Hz depth motion, checksum-bound
assets, and restart-safe caches. The remaining work is to make that performance predictable while
improving semantic fidelity without moving private reading data off the device.

The rule for adding a Google Cloud or NVIDIA product is simple: it must improve a measured latency,
quality, reliability, privacy, cost, or presentation gate. Product count is not an acceptance gate.

## Outcome targets

| Boundary | Current evidence | Promotion target |
| --- | ---: | ---: |
| Immediate semantic draft | under 100 ms | p95 under 100 ms; 30 fps; zero blank frames |
| Uncached Jetson scene plan | 2.70 s mean | p95 under 3.0 s with 5/5 semantic passes |
| Prepared cloud preview | 0.685-0.974 s | p95 under 0.8 s |
| Prepared master plus depth | 0.50-1.99 s | p95 under 1.0 s |
| Uncached text to master | 4.20-4.86 s | p95 under 4.0 s |
| Exact replay | 4-5 ms | p95 under 10 ms; zero provider work |
| Cold renderer readiness | 19-52 s accepted history | p95 under 60 s; no failed starts |
| Physical moving projection | 30 Hz accepted | p95 frame interval under 40 ms; zero software WebGL |

Quality gates are not negotiable for speed:

- one primary subject when the passage requests one;
- required setting, actor, object, action, and transformation present;
- no unintended readable text, logos, watermarks, or duplicate subjects;
- projection-bright luminance and a useful depth separation;
- raw passage, audio, camera frames, names, and learner telemetry remain local;
- every promoted artifact passes media, dimension, checksum, provenance, and cost validation.

All latency claims report cold and warm paths separately and include p50, p95, maximum, failure
count, GPU, model revision, image digest, and the scope of any cost estimate.

## Selected architecture

```text
book / typed text
       |
       v
Jetson Orin Nano
  local privacy gate -> local Gemma semantic plan -> procedural moving draft
       |                         |
       |                         +-> TensorRT Edge-LLM shadow candidate
       v
private IAM-authenticated Cloud Run GPU
  NVIDIA RTX PRO 6000 -> SANA-Sprint -> Depth Anything -> checksums
       |
       +-> optional asynchronous Nemotron visual critic
       v
Jetson cache -> NVIDIA WebGL depth renderer -> projector
```

The Jetson remains the privacy, planning, cache, reader-alignment, and projection boundary. Cloud
Run receives only the locally validated visual brief, style, seed, and fixed renderer parameters.

## Product decisions

### Use now

| Product or feature | Useful role | Reason it stays |
| --- | --- | --- |
| NVIDIA Jetson Orin Nano | private edge planner and projector | Keeps reading data local and removes a network hop from interaction and playback. |
| NVIDIA CUDA and TensorRT Edge-LLM | Gemma 4 E2B shadow planner | The only remaining local inference candidate with a credible chance of improving both reasoning and latency. Promotion remains benchmark-gated. |
| Cloud Run GPU with NVIDIA RTX PRO 6000 | SANA/depth renderer | Existing warm inference is already far below one second and the service can scale to zero. |
| Cloud Run service-level minimum instance | supervised judged window only | Removes scale-to-zero uncertainty. It is enabled only for a timed rehearsal or presentation and reset to zero afterward. |
| Cloud Run Job with RTX PRO 6000 | finite TensorRT checkpoint export | Keeps the 30.8 GB source checkpoint away from the 8 GB Jetson and writes only a target-ready bundle to GCS. |
| Artifact Registry | immutable renderer and exporter images | Digest pinning, layer reuse, retention policy, and rollback. |
| Cloud Build | reproducible GPU images | Existing layer cache is retained; high-CPU builders are considered only when build time is again on the critical path. |
| Cloud Storage | private TensorRT bundle and evidence archive | Uniform access, public-access prevention, checksum manifests, and automatic expiry. |
| IAM service identities and impersonated ID tokens | keyless private renderer access | No browser credentials and no committed service-account key. |
| Cloud Logging, Monitoring, and Trace | latency and availability evidence | Cloud Run request traces correlate with prompt-free structured stage logs; built-in metrics separate startup, request latency, GPU load, instances, and billable time. No story content is logged. |
| NVIDIA Grounding DINO | in-container subject/object gate | A failed first candidate gets at most one bounded retry; fidelity failures never silently become the master. |

### Experiment behind a gate

| Candidate | Gate | Stop condition |
| --- | --- | --- |
| TensorRT Edge-LLM Gemma 4 E2B INT4-AWQ | 5/5 normal plus adversarial schema, semantics, privacy, memory, and latency | Any OOM, privacy regression, schema miss, or no material end-to-end win. |
| NVIDIA nvImageCodec/nvJPEG | byte-identical dimensions, master SSIM >= 0.995, depth SSIM >= 0.994, lower packaging p95 | Less than 20 ms p95 end-to-end gain or added cold-start cost larger than the gain. |
| PyTorch 2.8 compile or CUDA Graph capture on Blackwell | same pixels or accepted visual A/B; lower repeated inference p95 | Compilation/capture increases cold readiness, memory, or failure rate more than warm savings. |
| Vertex AI-hosted NVIDIA NIM for Nemotron VL | asynchronous critic under 5 s warm with better retry selection than deterministic gates | Never block first projection; reject if source-image identity cannot be retained or NGC/GPU cost is disproportionate. |
| Cloud Run Rapid Cache or concurrent GCS model loading | five cold starts beat the immutable image by at least 20% | Do not add Direct VPC, cache, or storage complexity for a marginal win. |

### Do not add to the critical path

- Cosmos/world-model video: local depth motion is immediate and stable; video generation is a later,
  optional showcase upgrade.
- GKE for the renderer: Cloud Run already supplies the needed single-GPU private endpoint with much
  less operational and idle-cost overhead.
- Triton solely for branding: the current single-request diffusion pipeline would gain another
  server boundary without a demonstrated batching or throughput benefit.
- Cloud CDN, Pub/Sub, Cloud Tasks, BigQuery, Firestore, or Vertex Pipelines until a concrete measured
  need appears. They remain valid for distribution, asynchronous criticism, or large evaluation
  corpora, but none belongs in the first-image path today.

## Experiment sequence

### 1. Make the strict Cloud Run image boring and reliable

1. Stage immutable digest `sha256:d9bda0e00acd3889eb214b10841dac00105a18676953708be9621359736481ad`
   as a tagged, zero-traffic canary. Do not rebuild it.
2. Use a one-second TCP startup probe with a 900-attempt ceiling. The container must bind quickly;
   model readiness is verified separately by authenticated `/v1/prewarm`.
3. Run exactly one authenticated health request, one prewarm, and one synthetic generation.
4. Require the pinned RTX, CUDA, PyTorch, SANA, and depth identities plus artifact checksums.
5. Shift traffic only after the canary passes. On failure, keep the accepted revision at 100% and
   preserve logs; do not retry automatically.

### 2. Measure availability instead of guessing

Run three bounded modes against the same digest and prompt set:

1. `min=0`, cold start: five independent samples after confirmed scale-to-zero.
2. `min=0`, explicit prewarm: ten prepared requests inside one 90-second window.
3. `min=1`, supervised window: ten requests during a timed 20-minute judged-session simulation.

The service remains concurrency one and maximum one. The selected presentation mode must achieve
the latency target with zero failures. A minimum instance is reset to zero in a `finally`-equivalent
operator step even when the benchmark fails.

At the currently recorded full RTX instance rate ceiling of $0.00088522 per second, a 20-minute
minimum-instance experiment has an approximate $1.07 compute ceiling. This is an experiment limit,
not a price guarantee. The project-level gross-cost disconnect remains authoritative.

### 3. Finish the edge TensorRT decision

1. Use the completed, checksum-bound G4 Flex-start export
   `compute-g4-20260831-053351`; do not repeat the paid export unless its immutable manifest fails.
2. Keep the GCP VM and boot disk deleted. The accepted checkpoint has nine files totaling
   7,318,589,073 bytes and was verified both before and after its atomic Jetson install.
3. Retry the device-specific engine build with the fixed temporary 8 GiB NVMe swap. The first
   no-swap attempt completed TensorRT engine generation in 269.546 seconds, then was OOM-killed
   during in-memory serialization. Remove swap after the shadow benchmark; steady-state inference
   must pass with swap absent.
4. Run the same five semantic cases plus adversarial privacy/schema cases against Gemma 3/Ollama and
   Gemma 4/TensorRT Edge-LLM in counterbalanced order.
5. Record output tokens, p50/p95, peak RAM, GPU utilization, thermals, schema validity, semantic
   review, and projector contention.
6. Promote only if end-to-end latency materially improves and every quality/privacy gate passes.

### 4. Profile and optimize the warm renderer

Use CUDA events for GPU stages and monotonic clocks for the rest:

- text encoding;
- SANA denoising;
- depth inference;
- master/depth conversion and encoding;
- base64 serialization and response transfer;
- Jetson checksum/cache write;
- browser decode, first WebGL draw, and scene commit.

Test nvImageCodec first because encoding is isolated and reversible. Test compile/graph capture only
in a finite canary job. A candidate must improve p95, not merely a single best sample.

### 5. Add reasoning where it improves quality without delaying delight

Deploy Nemotron VL through NVIDIA NIM on Vertex AI only after the renderer path is reliable. It
receives the synthetic master and privacy-safe visual contract after `master_ready`, returns a
structured score, and may authorize one background refinement. The first image remains visible and
moving throughout. Use NVIDIA GenAI-Perf for the NIM latency report if this path is deployed.

### 6. Close with physical evidence

The final acceptance is not a laptop screenshot. It includes:

- 20 mixed-difficulty passages, at least five uncached;
- matched SSE, Cloud Trace, provider, cache, and projector timestamps;
- physical 1080p projector video with no blank frames;
- p50/p95/max latency and 20/20 semantic decisions;
- cold, prepared, replay, and offline-fallback demonstrations;
- exact Google/NVIDIA model, image, driver, runtime, and hardware provenance;
- authoritative gross project cost, credit scope, and teardown evidence.

## Guardrails

- One cloud GPU maximum; no automatic retry for paid generation or export.
- The $10 gross-cost emergency disconnect is a last-resort containment layer, not a target.
- Every paid experiment declares a worst-case ceiling before launch.
- No raw passage, learner identity, audio, camera frame, or credential in Cloud Logging, Trace,
  benchmark prompts, or committed artifacts.
- Canaries receive zero traffic until they pass. The previous ready revision remains the rollback.
- Minimum instances, NIM endpoints, GKE nodes, and export jobs are explicitly stopped or deleted
  after their bounded window.
- A faster candidate that misses semantics is a failed experiment.

## Primary references

- [Cloud Run GPU configuration](https://docs.cloud.google.com/run/docs/configuring/services/gpu)
- [Cloud Run GPU inference best practices](https://docs.cloud.google.com/run/docs/configuring/services/gpu-best-practices)
- [Cloud Run minimum instances](https://docs.cloud.google.com/run/docs/configuring/min-instances)
- [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)
- [NVIDIA NIM on Google Cloud](https://docs.nvidia.com/nim/large-language-models/latest/deployment/csp-deployment/google-cloud.html)
- [NVIDIA nvImageCodec](https://developer.nvidia.com/nvimagecodec)
