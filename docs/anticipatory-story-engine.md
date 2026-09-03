# Bookforge anticipatory story engine

Status: bounded private GKE v5 acceptance and a real Jetson offline-activation test passed on
2026-09-03; the NVIDIA L4 workload was then scaled to zero. The combined test exposed an action
fidelity error despite Nemotron acceptance, so visual fidelity is not a passed gate.

## The selling story

Bookforge is not a cloud image generator attached to a microphone. It is a privacy-split reading
instrument that begins preparing the next visual beat while the child is still reading the current
one:

1. The Jetson hears or receives the passage and uses local Gemma/TensorRT to understand it.
2. The Jetson removes the passage, audio, camera data, learner identity, and stable session identity.
3. A small CPU-only GKE API coordinates one known next scene, or at most two bounded predictions
   for an improvised story.
4. The existing private Cloud Run RTX service renders the synthetic master and depth map.
5. A separately scaled NVIDIA L4 Deployment runs Nemotron Nano VL NIM and inspects a temporary
   512-pixel review copy against the sanitized visual contract. It accepts, rejects, or authorizes
   exactly one repair.
6. The Jetson commits only the branch that actually occurs, verifies both asset hashes, and turns
   the still master plus depth map into a continuously moving NVIDIA-accelerated projection.

The visible result is a moving world that appears to understand the story without uploading the
child's voice, face, or book passage. Anticipation changes generation latency from a pause after the
sentence into background work hidden inside reading time.

## Why each platform is present

| System | Necessary job | Measured gate |
| --- | --- | --- |
| Jetson Orin Nano | Private Gemma scene planning, alignment, checksum cache, WebGL motion, projector output | No private source data in any cloud payload; no blank projection frame |
| TensorRT | Low-latency, memory-bounded local Gemma execution | Must beat the accepted local planner without losing schema or semantics |
| GKE Autopilot CPU API plus one separately scaled NVIDIA L4 | Bounded branch coordination without coupling API releases to NIM restarts | All six live requests accepted; NIM starts at zero replicas and is scaled only for a supervised run |
| NVIDIA NIM | Reproducible, structured multimodal Nemotron serving | Schema-valid verdicts; generated image is supplied before the contract text |
| Cloud Run RTX PRO 6000 | Fast SANA-Sprint master and Depth Anything output | Correct pinned identity, dimensions, hashes, and per-scene cost |
| Workload Identity | Keyless GKE-to-Cloud-Run authentication | No service-account key in the image, Pod, repository, browser, or Jetson |
| Artifact Registry and Cloud Build | Reproducible API image | Immutable image digest before the GPU Deployment is scaled above zero |

GKE is not replacing the renderer. Cloud Run is already the right scale-to-zero single-request
renderer. GKE adds value where it is materially different: a warm multimodal quality gate, branch
coordination, cancellation, cache reuse, and measured promotion before an asset reaches the child.
The coordinator and NIM are separate Deployments and Services. Routine CPU releases use
`deploy-anticipatory-api.sh`, which verifies that the NIM replica count did not change, so an API
update does not restart the expensive model container or repeat its TensorRT startup work.

GKE Inference Gateway is intentionally not enabled for the first experiment. With one NIM replica
and one L4 quota, it has no useful routing choice. It becomes a justified follow-up only if two or
more replicas are approved and prefix/cache-aware routing measurably improves p95 or reliability.

## Privacy boundary

`AnticipatorySceneSpec` is the only accepted cloud input. Pydantic rejects extra fields. Its required
attestation records that the following were removed on the Jetson:

- original passage or transcript;
- microphone audio;
- webcam or page-camera data;
- actual reader identity and stable learner/session identifiers.

The sanitized visual direction may retain fictional character or place names when they are needed
to render the story. It is not a general PII anonymizer, and the documentation does not claim that
every name-like story token remains local.

The cloud receives a visual brief, style, expected visible subjects, forbidden content, a random
seed, a one-use unlinkable session token, expiry, checksums, and hard cost reservations. The
contract can describe story meaning; it is not claimed to conceal the meaning of the requested
illustration. It does prevent the cloud API from accepting the source media or learner record.

The GKE API has no public Service. The first experiment uses `kubectl port-forward`. Production
internet ingress is a separate gate and must use HTTPS plus Identity-Aware Proxy or an equivalent
verified identity layer; a public unauthenticated LoadBalancer is not an acceptable shortcut.

Workload Identity is the only GKE-to-Cloud-Run credential path. The API NetworkPolicy permits DNS,
HTTPS, the NIM Service, and the GKE metadata endpoints at `169.254.169.254:80` and
`169.254.169.252:988`. The NIM policy accepts port 8000 only from the coordinator Pod and permits
only DNS and HTTPS egress. No service-account key is copied into an image, Pod, benchmark, or edge
device.

## Verified deployment shape

- `bookforge-anticipatory`: one CPU-only coordinator replica with a private ClusterIP Service.
- `bookforge-nemotron`: one-L4 NIM Deployment declared at zero replicas, with a private ClusterIP
  Service used by the coordinator at cluster-local DNS.
- `bookforge-nim-cache`: an 80 GiB `standard-rwo` volume that preserves downloaded model artifacts
  across NIM Pods. The NIM 1.3.1 low-memory TensorRT engine is still built in temporary storage, so
  the volume reduces model-download work but does not eliminate every engine rebuild.
- NIM memory is intentionally bounded at a 32 GiB request and 41 GiB limit on a `g2-standard-12`;
  smaller 16, 24, and 26 GiB attempts were rejected by measured OOM kills rather than hidden.
- Three clean-node pulls of the exact NIM image took 233.111-272.844 seconds, and two successful
  TensorRT engine builds took 243.53-252.68 seconds. These are startup measurements, not per-scene
  latency.

## Runtime limits

- One exact known-next scene, or two speculative branches; never an unbounded tree.
- Two render attempts per candidate maximum: initial image plus one Nemotron-directed repair.
- One concurrent render call, matching the Cloud Run service's concurrency-one, maximum-one shape.
- Candidate expiry between 30 and 900 seconds.
- Per-candidate render reservation at or below $0.25 and per-session reservation at or below $1;
  the benchmark uses $0.012 per attempt and $0.024 for the initial-plus-repair pair. Across six
  requests its declared GPU reservation is $0.144.
- Synthetic coordinator asset store bounded to 64 MiB by default and wiped with the API Pod.
- Persistent NIM model cache bounded by its dedicated 80 GiB volume.
- Cloud Run is never probed by Kubernetes readiness because a probe could wake a billable GPU.
- Renderer and critic prewarm require the same explicit billable authorization as the live harness.
- Failed rollout cleanup and the suspend script scale only the NIM GPU Deployment to zero; the CPU
  coordinator can stay available for inspection without holding an L4.
- A least-privilege in-cluster watchdog has only `get` and `patch` access to the NIM scale
  subresource and scales that Deployment to zero after one hour. Its projected token is mounted
  only into the watchdog sidecar; the NIM container cannot read it. The actual projected credential
  completed an authenticated zero-to-zero scale patch in the live cluster.

## Initial verified preflight and resulting state

On 2026-09-03, the read-only preflight for `your-gcp-project` reported:

- region `us-central1`;
- NVIDIA L4 regional quota 1, usage 0, available 1;
- private renderer `bookforge-scene-rtx` present in `us-central1`;
- GKE cluster `bookforge-anticipatory` absent;
- GKE API disabled;
- project billing linked to account `000000-000000-000000`;
- gross-spend alerts at a $150 monthly budget and emergency disconnect notification at $175,
  both excluding credits from the spend calculation.

Those are observations, not permanent guarantees. Run preflight again immediately before an
experiment. The control-plane observations and repaired disconnect revision are recorded in
`benchmarks/gcp-guardrail-audit-2026-09-03.json`. Preflight creates nothing:

The guarded experiment subsequently created the dedicated cluster. After the passing v5 run, the
`bookforge-nemotron` Deployment was scaled back to zero. The CPU API and 80 GiB cache may remain
until the explicit cluster-deletion step; neither is evidence that an L4 is still allocated. Check
the live quota and Deployment state before making any cost claim.

```bash
./infra/gcp/gke/preflight-anticipatory.sh
```

## Non-billable verification

```bash
.venv/bin/ruff check \
  src/bookforge/anticipatory.py \
  src/bookforge/anticipatory_app.py \
  src/bookforge/anticipatory_edge.py \
  src/bookforge/anticipatory_gcp.py \
  src/bookforge/anticipatory_service.py \
  src/bookforge/anticipatory_benchmark.py \
  src/bookforge/anticipatory_simulator.py \
  deploy/gke_anticipatory/app.py

.venv/bin/pytest -q \
  tests/test_anticipatory.py \
  tests/test_anticipatory_edge.py \
  tests/test_anticipatory_gcp.py \
  tests/test_anticipatory_service.py \
  tests/test_anticipatory_simulator.py \
  tests/test_anticipatory_benchmark.py \
  tests/test_gke_anticipatory_app.py \
  tests/test_nemotron_critic.py

.venv/bin/python -m bookforge.anticipatory_simulator \
  --output benchmarks/anticipatory-simulation-YYYY-MM-DD.json
```

Simulation output is labeled `deterministic_simulation_not_hardware_measurement`. It is a design
hypothesis, never contest performance evidence.

## Bounded live deployment

The deployment requires an NGC API key in the current shell. It writes the key only into temporary
mode-0700 files and Kubernetes Secrets, then removes the temporary files. Do not paste the key into
the repository, a command argument, a screenshot, or benchmark evidence.

First run the guarded script without authorization and inspect its exact scope:

```bash
./infra/gcp/gke/deploy-anticipatory.sh
```

Only when a supervised billable test is intended:

```bash
export NGC_API_KEY='REDACTED_VALUE_FROM_NGC'
export BOOKFORGE_GKE_APPLY=I_UNDERSTAND_THIS_CREATES_BILLABLE_GKE_GPU_RESOURCES
./infra/gcp/gke/deploy-anticipatory.sh
```

The script enables required APIs, builds from locked dependencies, resolves the API image to a
digest, creates a dedicated Google service account, grants only Cloud Run invocation, configures
Workload Identity, installs the CPU API at one replica and the NIM Deployment at zero, and scales
only NIM to one after every prerequisite exists. The NIM, Python, uv, and Cloud Build builder images
are digest-pinned. If this run created the cluster and any later deployment step fails, the cleanup
trap scales NIM to zero and deletes that newly created cluster. It never deletes a cluster that
existed before the run.

That full script is for initial creation or an intentional infrastructure reconcile. Routine API
code releases do not reapply the zero-replica NIM manifest or require a free L4:

```bash
export BOOKFORGE_GKE_API_APPLY=I_UNDERSTAND_THIS_RUNS_A_BILLABLE_CLOUD_BUILD
./infra/gcp/gke/deploy-anticipatory-api.sh
```

The API-only script records NIM replicas before the build, changes only the API image and renderer
environment, waits for the CPU rollout, and then fails if the NIM count differs. It needs no NGC
credential.

## Measured acceptance

For a workstation-only acceptance run, open the private tunnel in one terminal:

```bash
kubectl -n bookforge port-forward service/bookforge-anticipatory 18082:8080
```

The readiness route checks the separate, already-running Nemotron Service; it does not prewarm
either paid GPU dependency:

```bash
curl -fsS http://127.0.0.1:18082/ready
```

For the Jetson integration run, use the bounded two-hop bridge instead. It combines the GKE
port-forward with an SSH reverse tunnel over the Jetson's existing Tailscale connection. Nothing is
published to the internet, the SSH key stays on the workstation, and both tunnels are removed when
the command exits:

```bash
export BOOKFORGE_JETSON_HOST=192.0.2.10
export BOOKFORGE_JETSON_USER=operator
export BOOKFORGE_JETSON_KEY=/Users/operator/.ssh/bookforge_jetson
export BOOKFORGE_JETSON_HOST_KEY_ALIAS=jetson.local
./infra/gcp/gke/bridge-jetson-anticipatory.sh
```

The Jetson then reaches the private service at `http://127.0.0.1:18082`; that is why its explicit
loopback-only HTTP setting below remains safe for this experiment. This bridge is for a supervised
test, not unattended production operation.

Then run the live harness once. Its explicitly authorized prewarm starts the private Cloud Run
renderer and Nemotron critic concurrently. The critic prewarm compiles its first structured request,
while renderer prewarm loads the diffusion and depth pipelines. The harness then sends three
synthetic sanitized contracts, repeats them to prove cache reuse, waits for Nemotron decisions,
commits each accepted result, downloads master and depth, and verifies both hashes. It refuses to
overwrite evidence and has no automatic paid retry:

```bash
.venv/bin/python -m bookforge.anticipatory_benchmark \
  --base-url http://127.0.0.1:18082 \
  --ready-timeout-seconds 120 \
  --authorization I_UNDERSTAND_THIS_WAKES_BILLABLE_GPUS \
  --output benchmarks/anticipatory-gke-YYYY-MM-DD.json
```

Promotion requires every scene accepted, every asset verified, all three replays served from cache,
replay p95 below 250 ms, commit-plus-fetch p95 below 750 ms, and estimated render cost below $0.15.
Startup time, GPU memory, and authoritative Cloud Billing cost must be recorded separately before a
contest claim is written.

### 2026-09-03 measured results

The immutable evidence is `benchmarks/anticipatory-gke-2026-09-03-v3.json`:

- concurrent dependency prewarm completed in 42.484 seconds wall time: Cloud Run reported 23.05
  seconds and Nemotron reported 42.16 seconds;
- all six requests were accepted and all six master/depth pairs passed checksum verification;
- first-pass ready time was 9.022-11.672 seconds, with 0.400-0.461-second rendering and
  8.385-11.079-second Nemotron review;
- all three replay requests were cache hits, ready in 72.545-74.607 ms, with zero render attempts;
- commit-plus-fetch was 152.306-158.458 ms on replay; the six-case p95, which includes first-pass
  downloads, was 510.970 ms;
- the three new renders' estimated GPU cost totaled $0.0003239403;
- no repair attempt was needed, and every declared promotion gate passed.

Two preserved failed runs explain why the passing numbers are credible rather than cherry-picked.
The first run could not obtain a Cloud Run identity because NetworkPolicy blocked the GKE metadata
server; the policy now admits only the two metadata-server paths above. The second run exposed cold
renderer and critic timeouts. The fix was an explicit concurrent prewarm, a maximum-512-pixel JPEG
review copy, and a compact primitive-only structured-output schema that stays on NIM's fast
xgrammar path instead of falling back to the slower Outlines grammar. Pydantic still performs full
semantic validation after decoding. The 512-pixel copy exists only for review; the full-resolution
projection master and depth assets are unchanged.

The compact-wire v4 experiment is also preserved rather than hidden. It reduced one successful
critic call to 6.422 seconds, but failed four requests because NIM represented an optional accept
correction as `""` or `"None"`. The accept-only sentinel normalization was corrected, the output
ceiling was raised from 96 to 128 tokens, and the fix shipped through the CPU-only release path
without restarting NIM.

The resulting `benchmarks/anticipatory-gke-2026-09-03-v5.json` is the current passing target:

- all six requests were accepted and all six master/depth pairs passed checksum verification;
- uncached ready time was 6.820-7.865 seconds, a 32.6% p95 reduction from v3;
- first-pass rendering was 0.396-0.477 seconds and Nemotron review was 6.241-7.295 seconds;
- each critic request used 948-958 input tokens and 91-108 output tokens, leaving measured room
  beneath the 128-token ceiling;
- all replays were cache hits ready in 66.984-69.524 ms with zero renderer or critic work;
- replay commit-plus-fetch took 141.178-146.930 ms; all-case p95 was 457.714 ms;
- the concurrent warm-runtime check took 6.236 seconds because both model processes were already
  resident after the API-only rollout; and
- estimated incremental renderer request cost was $0.0003141211. This excludes Cloud Run prewarm,
  GKE/NIM runtime, storage, cluster management, and authoritative Cloud Billing totals.

The deployment identities and startup observations are bound separately in
`benchmarks/anticipatory-gke-infrastructure-2026-09-03.json`; request evidence never pretends that
its renderer-only estimate is the complete cloud cost.

## Jetson integration

The main Bookforge API now exposes a local-only control surface:

- `POST /v1/anticipations:prepare` accepts private local text, invokes the configured Gemma planner,
  and sends only the sanitized contract;
- `GET /v1/anticipations/{token}/{sequence}` polls bounded progress;
- `POST /v1/anticipations:commit` selects a ready branch and cancels its sibling;
- `DELETE /v1/anticipations/{token}/{sequence}` cancels unused work;
- `GET /v1/anticipations/{token}/{sequence}/{branch}/{master|depth}` retrieves a hash-verified
  synthetic asset for the projector.

The feature is off by default. During the private port-forward experiment, configure:

```bash
export BOOKFORGE_LIVE_SCENE_PLANNER=model
export BOOKFORGE_ANTICIPATORY_BACKEND=gke
export BOOKFORGE_ANTICIPATORY_URL=http://127.0.0.1:18082
export BOOKFORGE_ANTICIPATORY_ALLOW_LOOPBACK_HTTP=true
```

Do not enable loopback HTTP for a non-loopback hostname or IP. The client enforces that rule.

For the standalone system service, shell `export` commands do not change its root-owned
`/etc/bookforge/bookforge.env`. After the bridge is reachable from the Jetson, copy the repository's
`deploy/jetson/configure-anticipatory.py` to that device and run it there with system Python:

```bash
sudo /usr/bin/python3 -I ./configure-anticipatory.py --user operator
# Optional: add --dry-run to validate without changing files or restarting.
# Roll back this feature later with --disable; no live bridge is required to disable it.
```

This helper is a one-time privileged configuration change, not a new password-free administration
capability. It preserves credentials and the accepted local planner, changes only the four bridge
settings, creates a private rollback copy, and restarts only the API. Failed readiness restores the
previous configuration. It never scales a GPU, adds a public listener, installs a credential, or
changes sudoers. The bridge's `/health` identity must match before enabling; a lost bridge fails
before configuration is changed. The private tunnel is pinned to the Bookforge GKE context rather
than whichever cluster happens to be selected in the workstation shell.

## Workbench next-page rehearsal

Implemented on 2026-09-03: an exact known-next-page path that keeps cloud work off the page-turn
critical path. The workbench has a separate next-passage field; it does not overwrite the current
story input until the user chooses **Show prepared scene**.

1. **Prepare next scene** runs the existing local planner once. The coordinator validates the
   compact visual contract against Nemotron's 1,400-byte input budget before a cloud submission.
   One candidate reserves $0.012 per render attempt, $0.024 including the optional repair. These
   are render reservations, not an all-in cap on GKE/NIM infrastructure charges.
   Before planning or submitting, a five-second read-only `/ready` check confirms that Nemotron is
   already available. If the bridge or NIM is off, preparation fails without a render request.
   The explicit warmup button applies the same guard before it can wake the separate renderer;
   it does not scale the GKE Deployment itself. This readiness check cannot guarantee the GPU will
   remain available afterward, so normal failures and cost reservations still apply.
2. Read-only progress polling waits for an accepting critic verdict. The device downloads master
   and depth concurrently, checks both SHA-256 hashes, decodes both images, validates dimensions,
   and writes them to the existing local asset cache. It commits the single known branch remotely
   at this stage, but leaves the current projector session untouched.
3. **Show prepared scene** validates the cached pack and atomically publishes a completed
   `master_ready` job into the existing session event stream. No cloud request, local model call,
   intermediate draft, or page reload is required. The projector retains its existing depth-motion
   renderer and visual blend. Client image decoding/upload and the blend still take time; this is
   not a claim of zero visual-onset latency.

The device retains up to eight prepared-page handles for 15 minutes from the initial preparation,
or until the API restarts. Staged status, activation, and discard work without cloud connectivity.
Metadata for a page that has not been shown is process-local; this is not yet a durable offline
library. Activation checks the server epoch and current session revision, and cannot replace an
in-progress generation. Duplicate activation of the current page is idempotent. Concurrent planning
of a different page does not hold the activation lock.

Local-only endpoints, also allowlisted through the paired controller:

- `GET /v1/prepared-projections/runtime`: configuration only; no cloud request.
- `POST /v1/prepared-projections`: prepare `{text, visual_style, session_id, seed?}`.
- `GET /v1/prepared-projections/{prepared_id}`: inspect progress.
- `POST /v1/prepared-projections/{prepared_id}/stage`: verify and cache approved assets.
- `POST /v1/prepared-projections:activate`: select a staged page with expected server epoch/revision.
- `DELETE /v1/prepared-projections/{prepared_id}`: discard the handle; no projector change.
- `POST /v1/prepared-projections:prewarm`: explicit billable warmup; never invoked on panel load.

Verification: fixture-backed HTTP integration covers source privacy, asset checksums and decoding,
rejection, expiry/capacity, stale activation, offline replay, and the paired route allowlist. A real
browser followed prepare → stage → show → projector asset loads with no console warnings/errors;
the artwork and timing inputs in that browser check were synthetic test fixtures, not new GPU
measurements. At that checkpoint, the live GKE v5 benchmark was the cloud evidence and the combined
Jetson/GKE/projector acceptance run remained a separate gate. Those fixture tests do not establish
physical frame rate or scene-generation quality; the real delivery result is recorded below.

The new runtime wheel was installed on the Jetson on 2026-09-03. The running API and paired
controller both passed readiness checks; the served next-page JavaScript and installed playback
module matched the local SHA-256 hashes. The GKE preparation flag remained disabled at that initial
release, which did not create cloud work. The wheel digest is
`40e2249dec2334df26329606c49ebd55055a52f5d52303deefc84d0f10ae61a0`; the previous application files
are retained on the device at
`/home/operator/bookforge-next-page-40e2249dec23/runtime-before.tgz`.

Follow-up bridge release: wheel
`4ae7c837a93580e008edc2592a6dc717f31791fa315c4e0dfbf46d9b49d93bfe` adds the pre-render NIM readiness
guard. The actual GPU-off service denied the check before any render/warmup request. The private
bridge health contract was also verified from the Jetson. The user subsequently ran the one-time
configuration helper at `/home/operator/bookforge-connect-gke-301d14ec6270.py` (SHA-256
`301d14ec627010c510ed13be557716d9a2aeeac40b29142fecaa3e671f6dd595`), and the running API confirmed
the backend enabled. This does not add a new password-free sudo capability. The bridge
requires the Mac to remain awake for the supervised test; it is not permanent standalone ingress.

### Real Jetson delivery acceptance

`benchmarks/anticipatory-jetson-playback-2026-09-03.json` records one new scene entered through the
real workbench, planned by local Gemma/TensorRT, rendered on Cloud Run, reviewed by GKE/Nemotron,
and staged on the Jetson. It is separate from the earlier six-case cloud benchmark and browser
fixtures. The full suite passed 1,053 tests before the run.

- Local planning took 1,379 ms in a preceding uncached check; preparation reused that plan in
  0.82 ms. Rendering took 426 ms and Nemotron review took 5,676 ms. These components are not a
  measured end-to-end click-to-ready time.
- The scene was downloaded and verified before publication. NIM was scaled to zero and the
  private cloud bridge disconnected before **Show prepared scene** was clicked. Staged status
  remained available with the cloud endpoint refusing connections.
- Local publication took 3.41 ms. Client activation took 126.8 ms in the Mac preview and 233 ms
  on the Jetson, with a separate configured 320 ms blend. Neither figure measures click-to-photon
  latency or establishes a distribution from one sample.
- After closing the Mac preview to isolate telemetry, the Jetson reported approximately 30
  depth-rendered FPS across three samples. Presentation mode disables display-frame sampling, so
  this is not evidence of physical display FPS or zero dropped display frames. Observer
  confirmation of the physical projection was not supplied during this run.
- Both cached assets retained their expected hashes after disconnect. Repeating activation
  returned the same job and revision without new generation.
- Cold startup from Pod creation to readiness took 1,001 seconds; its language-engine build
  alone took 273.65 seconds. Warmup then reported 23.97 seconds for the renderer and 37.67 seconds
  for the critic. Startup must be completed before a rehearsal; it is not hidden in warm latency.

Visual review found a genuine failure case: the image showed a silver fox beside an oversized
lantern, not clearly carrying it. The local contract did preserve “carrying,” but Nemotron accepted
the image anyway. The next critic evaluation must include this actor-object relationship failure;
successful delivery and attractive artwork are not proof of exact story fidelity. No additional
generation was used to replace this result in the record.

The next-page panel itself never automatically warms the cloud. The pre-existing regular-generation
panel still warms its renderer when the workbench opens; do not describe the entire workbench as
having no billable startup behavior. The private GKE bridge was closed after this acceptance run.

Reproduce without cloud credentials or GPU work:

```bash
.venv/bin/pytest -q tests/test_anticipatory_playback.py tests/test_anticipatory_workbench.py
PYTHONPATH=src:tests .venv/bin/python tests/serve_prepared_projection_fixture.py
# Open http://127.0.0.1:18085/workbench?session=fixture
# Fixture passage: Mira whispered that the fox had finally found the hidden garden.
```

For a supervised live rehearsal, use the private bridge and configuration above. The bridge depends
on the workstation remaining connected; laptop-independent authenticated ingress and automatic
reading alignment are not enabled by this milestone. Do not expose the local API on a public port
to bypass that remaining connection work.

## Immediate shutdown and deletion

Scaling the NIM Deployment to zero stops the declared GKE GPU workload without taking down the CPU
coordinator. It does not promise an immediate end to every cluster-level charge:

```bash
export BOOKFORGE_GKE_SUSPEND=I_UNDERSTAND_THIS_STOPS_THE_GKE_GPU
./infra/gcp/gke/suspend-anticipatory.sh
```

After evidence is copied, delete the bounded experiment cluster to stop persistent cluster charges:

```bash
export BOOKFORGE_GKE_DELETE=I_UNDERSTAND_THIS_DELETES_THE_ANTICIPATORY_CLUSTER
./infra/gcp/gke/delete-anticipatory-cluster.sh
```

The existing project budget notifications and billing disconnect are last-resort containment. They
are asynchronous and are not a mathematically exact hard cap. A supervised start, measured run,
scale-to-zero, and cluster deletion remain the primary controls.

## Claims allowed after the live gate

Allowed only with recorded evidence:

- “Gemma understands the story locally on Jetson.”
- “GKE runs a pinned NVIDIA Nemotron NIM that judges only synthetic imagery and sanitized intent.”
- “The bounded GKE path accepted and checksum-verified all six v5 requests, including three cached
  replays ready in under 70 ms.”
- “All three uncached scenes were ready in under 7.9 seconds in the measured v5 run, 32.6% faster
  at p95 than v3, so work can begin before the reader reaches the next story beat.”
- “The projector commits checksum-verified master and depth assets without a blank frame.”

Not allowed without additional evidence:

- that GKE is faster than Cloud Run at rendering;
- that simulation values are hardware measurements;
- that budget alerts guarantee zero overage;
- that Nemotron verified a scene when the request fell back or timed out;
- that the cloud learns nothing about the requested illustration's semantic content.
