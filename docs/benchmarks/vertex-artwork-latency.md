# Vertex artwork latency: 4.32 seconds from checked scene to image

Storylight produced a finished 1376 x 768 scene in 4,319.633 ms after accepting a synthetic visual description. The managed Vertex request accounted for 4,127.179 ms, scene planning took 104.218 ms, and local packaging took 7.582 ms. The request reached `master_ready` on the first attempt with no scene-cache hit.

| Measurement | Result |
| --- | ---: |
| Checked description to `master_ready` | 4,319.633 ms |
| Vertex provider request | 4,127.179 ms |
| Scene planning | 104.218 ms |
| Local packaging | 7.582 ms |
| Output | 1376 x 768 JPEG |
| Managed model | `gemini-3.1-flash-lite-image` |

Once the local scene contract had been accepted, 95.5 percent of the measured interval sat inside the managed image request. The surrounding Storylight work was small enough that further local optimization would barely move this sample. For this route, perceived speed depends on beginning the checked request early and keeping the preceding language stage bounded, while the previous scene remains on the projector until the new revision is complete.

## The request path

The Vertex provider receives a `FastSceneRequest` after Storylight's grounding and privacy checks. It sends the checked visual prompt, a negative prompt, one integer seed, and image-generation controls through Vertex AI's `generateContent` endpoint. The request asks for one 16:9 JPEG at quality 95. Explicit counts found in the checked prompt are repeated as count locks, while the general contract asks the model to preserve subjects, colors, actions, and spatial relations.

Authentication stays in the service. Google Application Default Credentials supply a Cloud Platform access token, cached for five minutes behind an async lock so a readiness probe and a generation request cannot trigger competing refreshes. A nonbillable `HEAD` request prepares DNS, TLS, and the HTTP connection before generation. The client keeps one HTTP/1.1 connection alive because requests are serialized and an earlier HTTP/2 path exposed a closed-connection PING failure.

The response is accepted only when it contains a supported inline JPEG or PNG, valid base64, bounded byte length, and parseable dimensions. Storylight hashes the image, writes it into a new scene directory, and creates a deterministic grayscale depth sidecar locally. That sidecar makes basic parallax available immediately and avoids a second paid media request; a Jetson depth model can replace it later.

Failure handling matters here because a transport exception after submission may still represent billable work. Storylight labels that state ambiguous and reserves the estimated request cost instead of sending the same scene to another provider. Explicit non-2xx responses can enter the fallback path, `429` honors `Retry-After`, and a session-local cost reservation blocks further calls before the configured cap is crossed.

## Measurement boundary

This record covers one synthetic request. It begins with an accepted scene description and ends when the master image and depth sidecar are packaged, so microphone capture and local Whisper time are outside the interval. The provider's GPU warm state was not independently observed. One sample cannot describe tail latency or regional variance.

The visual inspection found one cat chasing one mouse through a dark alleyway, which matched the requested subject count and action. That inspection supports this request's acceptance; it is not an image-quality study.

## What we learned and what is next

The experiment established a working managed route with bounded input, explicit cost accounting, checksum-bound output, and a complete projector artifact in 4.32 seconds. Storylight now treats provider time as the dominant term for this route and overlaps earlier work where correctness allows it, while revision IDs prevent a late result from replacing a newer scene.

The next measurement should use repeated, randomized requests across more than one Vertex location and report median, p95, transport reuse, model warm state, and failure rate separately. Microphone-to-projector latency belongs in another record because combining it with provider time would hide which system changed.

Raw values: [`voice-smoke-2026-09-07.json`](voice-smoke-2026-09-07.json). Implementation: [`vertex_scene_provider.py`](../../src/storylight/vertex_scene_provider.py).
