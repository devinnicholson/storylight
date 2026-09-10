FROM nvcr.io/nvidia/pytorch:25.12-py3

ARG EDGELLM_REVISION=71dd1bae032e70771265917ec74d3ff4cad07a10

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/storylight-export/huggingface \
    HF_HUB_CACHE=/tmp/storylight-export/huggingface/hub \
    HF_XET_HIGH_PERFORMANCE=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git /opt/tensorrt-edge-llm \
    && git -C /opt/tensorrt-edge-llm checkout "${EDGELLM_REVISION}" \
    && test "$(git -C /opt/tensorrt-edge-llm rev-parse HEAD)" = "${EDGELLM_REVISION}" \
    && python -m pip install --no-cache-dir -e '/opt/tensorrt-edge-llm[tools]' \
       google-cloud-storage==3.4.1 \
    && tensorrt-edgellm-quantize --help >/dev/null \
    && tensorrt-edgellm-export --help >/dev/null

COPY export.py /opt/storylight/export.py

ENTRYPOINT ["python", "/opt/storylight/export.py"]
