# Storylight

Storylight turns spoken descriptions into illustrations on a nearby monitor or projector. Describe a scene from a book, then change a detail and watch the picture respond.

The best learning happens when you try something because you're curious. Learning a language should leave room to play with a sentence just to see what happens. Storylight brings that idea to reading, with pictures that follow your words.

The picture needs to be accurate and arrive before the conversation moves on. The inference experiments test how fast open models can meet that requirement.

![Moon Gate: an example story scene](src/storylight/static/assets/moon-gate-hero-v1.png)

*Artwork included with Storylight. Depth and parallax add movement to still images.*

## How it works

A green cat chasing a mouse needs to stay green, and it needs to be the one doing the chasing. If the reader changes the description, the system needs to follow the correction.

In the current demo, the browser captures speech and a local Whisper service on the workbench transcribes it. A gateway sends the description to the Jetson, which checks the proposed scene against the transcript and removes recognized private information before sending a visual prompt to the cloud renderer.

Generation can start from a usable partial transcript while the reader is still speaking. Later speech can supersede that request; the previous image stays visible until completed artwork matches the latest accepted description. An exact cached scene avoids another generation request. Superseded cloud work may still incur a charge if it has already started.

[Architecture](docs/architecture.md) · [Voice and projector setup](docs/demo.md)

## Google models, NVIDIA hardware

The NVIDIA Jetson Orin Nano is the local appliance. Its 8 GB of unified memory sets the budget for on-device model work, while it runs the scene service and serves the projector display. In the demonstrated voice setup, the workbench handles transcription and the Jetson checks the description before requesting cloud artwork. Completed scenes return to its local cache for display on the connected monitor or projector.

Google's Gemma has been part of the on-device planning work, including a quantized Gemma 3 1B configuration and a TensorRT planner integration. The current voice compiler uses bounded parsing and local language checks; the Gemma 4 QLoRA experiments explore a learned replacement. Those experiments run on NVIDIA L4 GPUs in Google Cloud, where there is room to train adapters and compare decoding implementations before attempting deployment on the Jetson.

Image-generation work includes open-weight FLUX.2 Klein 4B and SANA-Sprint backends. Google Cloud integrations cover GPU rendering through Cloud Run and managed image generation through Vertex AI. A separate GKE workflow uses NVIDIA's Nemotron to review generated artwork before a prepared scene is staged on the Jetson. Each model has a defined job, and the local service controls which scene reaches the reader.

## JAX and MaxText: training the scene planner

We built an offline Gemma 4 E2B training pipeline with JAX and Google's MaxText, using LoRA to teach the model the structured scene descriptions consumed by the application. The work covered Hugging Face checkpoint conversion into MaxText and merged export back, with checks on model outputs across conversion. This used ordinary LoRA, separate from the later quantized training experiment.

On two NVIDIA L4 GPUs, we verified that the native training state contained 205 paired LoRA modules and that training produced nonzero gradients and changed adapter weights. Persistent JAX compilation caching reduced the measured cold/warm job duration from 738 to 331 seconds, with provider cost falling from about $0.64 to $0.32. These were Modal runs using Google's training software on NVIDIA hardware, not Google Cloud measurements.

That shorter turnaround makes repeated specialization experiments cheaper to run. The training harness is archived, and its adapters have not replaced the live planner; the result demonstrates training and compilation reuse rather than faster scene generation.

## Privacy and open models

A child's voice should stay on local hardware, and their name should never become part of a cloud image prompt. In the documented demo, recordings are transcribed locally and descriptions are checked locally. The renderer receives a visual prompt with recognized private names and contact details excluded. It receives neither the recording nor the original transcript.

Open weights let us run speech and language processing on hardware we control. Access to the implementation lets us inspect what leaves the device and change the inference code that slows it down. Publishing Storylight's source gives other people that same freedom to adapt it for their readers.

The application code is licensed under Apache 2.0. Model weights have separate licenses; see the [model notes](docs/research-results.md#models-and-licensing). Managed Vertex image generation still requires cloud access.

The privacy checks cannot recognize every sensitive detail, and a visual prompt can still disclose private information. Local text and assets can persist. Other provider configurations can send text to remote endpoints; review the [privacy documentation](docs/privacy.md) before using private material.

## Inference experiments

The QLoRA and compiled-decoding experiments used Gemma 4 E2B on an NVIDIA L4 in Google Cloud. A separate JAX training campaign used two NVIDIA L4s on Modal. These candidates are separate from the demo's current scene analysis; their timings do not measure image generation or the full time from speech to display.

### QLoRA: adapting a model to a reading task

QLoRA trains small adapters attached to a quantized model. This experiment used 4,800 synthetic training examples and 256 independently authored development examples to select the recipe and checkpoint, without collecting readers' voices or reading histories.

The next direction is adapters for specific reading tasks. A reader practicing “under” and “behind” needs the picture to preserve that distinction, and task-specific training lets us target it directly. Sharing adapters trained on curated or synthetic examples would let others extend that work to the vocabulary their readers need.

Multilingual adapters remain future work. The current candidate needs further quality and device validation before deployment; the [research notes](docs/research-results.md) contain the full evaluation, including refusal behavior.

### Compiled decoding

The runtime processes prompts dynamically, then transfers the attention-cache state into static buffers for compiled decoding. Across 128 previously exposed descriptions, every measured output-token pair matched while median extraction time fell from 2.85 to 1.09 seconds. The median reduction within pairs was 59.49%.

This was one measurement per path per description in fixed path order, excluding tokenization and output decoding. Testing on fresh inputs and deployment hardware will establish whether the gain carries through to the application.

### Reusing compiler preparation

Reusing compiler work across processes can reduce the wait after a restart. Keeping the Python hash seed fixed reduced the first compiled preparation phase from 109.25 to 22.49 seconds, a 79.41% reduction in the measured cycle. Changing the seed brought the time back to 102.61 seconds; inspection linked the change to attention-input ordering and graph-cache misses.

These measurements cover one cycle with eight training probes, after earlier inference modes had warmed the model and GPU. They measure compiler preparation, not whole-service startup. Repeating the experiment across environments will test whether the improvement holds.

### Nemotron: reviewing artwork and reusing TensorRT engines

We ran NVIDIA Llama 3.1 Nemotron Nano VL 8B through NIM 1.3.1 on a Google Kubernetes Engine L4. It reviews a generated image against a bounded visual brief and returns a structured verdict, with a correction when needed. Its input excludes the reader's recording and original transcript. In the prepared-page workflow, the Jetson verifies downloaded asset hashes and dimensions before staging the approved scene. Nemotron review is optional and stays off the live demo's first-image path.

We also persisted NIM's TensorRT vision and language engines so they could be reused after a restart. Container-start-to-ready time fell from 631 to 229 seconds, a 63.7% reduction, with all 12 baseline review verdicts unchanged. This was a same-node restart comparison, not a clean-node cold start. Review prompts still need broader fidelity testing; the next-page integration and the engine-reuse result are implemented work, not evidence that visual review catches every mistake.

The [research notes](docs/research-results.md) and [numeric summary](research/results.json) document the Gemma results and their limits. The [Nemotron client](src/storylight/nemotron_critic.py) and [GKE deployment](infra/gcp/k8s/anticipatory.yaml) contain the review integration. Original operational receipts and large artifacts remain private, so the published material is not yet a complete reproduction package.

## Running the prototype

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv sync --dev
STORYLIGHT_MODEL_BACKEND=fake \
STORYLIGHT_ASSET_BACKEND=fake \
STORYLIGHT_LIVE_SCENE_BACKEND=fake \
STORYLIGHT_ASR_BACKEND=disabled \
STORYLIGHT_ANTICIPATORY_BACKEND=disabled \
STORYLIGHT_LIVE_SCENE_CRITIC_BACKEND=disabled \
uv run uvicorn storylight.api:app --host 127.0.0.1 --port 8080
```

Open [the workbench](http://127.0.0.1:8080/workbench). This simulated configuration exercises the interface without model downloads or inference requests. It does not transcribe speech or generate AI artwork. The [demo guide](docs/demo.md) covers provider configuration for voice input and generated scenes.

## Further work

The next inference work is to measure speech-to-display time on deployment hardware, including corrections that arrive during generation. That will show how much of the measured speed gain reaches the reader.

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) to work on the code or report results. Use synthetic stories and recordings in public reports, and follow [SECURITY.md](.github/SECURITY.md) for sensitive issues.

[Apache 2.0](LICENSE) · [Third-party notices](NOTICE)
