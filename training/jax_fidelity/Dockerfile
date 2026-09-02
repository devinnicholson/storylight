# syntax=docker/dockerfile:1.7
FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none --no-checkout https://github.com/AI-Hypercomputer/maxtext.git /opt/MaxText \
    && git -C /opt/MaxText checkout --detach 538fe7a3f3376d94cf3f04e77741aa6d7e8efa45 \
    && test "$(git -C /opt/MaxText rev-parse HEAD)" = "538fe7a3f3376d94cf3f04e77741aa6d7e8efa45" \
    && test -z "$(git -C /opt/MaxText status --porcelain)"

WORKDIR /opt/bookforge
COPY training/jax_fidelity/requirements.lock /tmp/bookforge-jax-requirements.lock
RUN python -m pip install --requirement /tmp/bookforge-jax-requirements.lock

COPY . /opt/bookforge
RUN python -m training.jax_fidelity --config experiments/jax-fidelity-lab/config.json \
    && python -m training.jax_fidelity.verify_runtime --write-lock /opt/bookforge/runtime.lock.json \
    && python -m training.jax_fidelity.verify_runtime --lock /opt/bookforge/runtime.lock.json

ENTRYPOINT ["python", "-m", "training.jax_fidelity"]
CMD ["--config", "experiments/jax-fidelity-lab/config.json"]
