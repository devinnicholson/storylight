# Contributing to Storylight

Use Python 3.11 or later, `uv`, and Node.js 20 or later. Node is used only
for the dependency-free browser-side tests. Install the development
dependencies and start the local API with:

```sh
make install
make dev
```

The development server listens on `http://127.0.0.1:8080`. Copy `.env.example` to `.env` when local configuration is needed. Keep credentials and generated runtime state out of Git.

Run the checks relevant to your change, then the project checks before submitting:

```sh
make test
make lint
```

Keep changes focused. Add tests for behavior that can fail, especially cancellation, stale results, parsing, and privacy boundaries. Avoid duplicate tests or assertions that merely repeat the implementation.

For performance changes, record the model and runtime versions, hardware, input scope, and timing boundaries. Report preparation costs separately from warm inference. Preserve failed runs and output differences; a faster result is not an improvement if it loses required facts. Synthetic or previously used examples do not establish performance on new user input.

Use synthetic inputs in examples and reports. Do not commit private transcripts, recordings, model weights, compiler caches, cloud credentials, or account-specific deployment receipts. Put large experimental artifacts outside the source tree and link to a sanitized summary when useful.

Cloud experiments need explicit spending authorization, a time limit, and cleanup of resources after evidence is collected. A code change does not authorize deployment, publication, or a paid benchmark.
