# Official CUDA runtime; no engine source patches or model weights.
FROM docker.io/nvidia/cuda@sha256:62f16f5a4d21773127cb088160969852d1bd01c39fbfe537a7e92fe7ee23f522

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Valen vLLM serving" \
      org.opencontainers.image.description="Pinned official vLLM with the Valen decision plugin" \
      org.opencontainers.image.source="https://github.com/Valen-Team/Valen" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VLLM_PLUGINS=valen \
    VLLM_USE_V2_MODEL_RUNNER=0
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv ca-certificates libnuma1 libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

COPY serving/vllm/docker/runtime-wheels.txt /opt/valen/runtime-wheels.txt
# Direct URLs and hashes select exactly the already tested wheel set.
RUN python3 -m pip install --no-cache-dir --no-deps --require-hashes \
      -r /opt/valen/runtime-wheels.txt \
    && python3 -m pip install --no-cache-dir --no-deps \
      'https://wheels.vllm.ai/30aded62d46dfcf52c815925a8acec5a8bebddc6/vllm-0.32.1rc1.dev253%2Bg30aded62d-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=8c424b1a95e4f893f1c31f2073001e07c2e645d17e7f0745e1abdddaa864fb94'

COPY serving/vllm /opt/valen/plugin
RUN python3 -m pip wheel --no-deps --no-build-isolation \
      --wheel-dir /opt/valen/wheels /opt/valen/plugin \
    && python3 -m pip install --no-deps /opt/valen/wheels/valen_vllm-*.whl \
    && python3 -m valen_vllm.compatibility --source-tree > /opt/valen/engine-fingerprint.json \
    && python3 -m pip check \
    && python3 -m pip freeze > /opt/valen/runtime-packages.txt \
    && sha256sum /opt/valen/wheels/*.whl > /opt/valen/plugin-wheel.sha256 \
    && printf '%s\n' "${VCS_REF}" > /opt/valen/source-revision.txt

WORKDIR /opt/valen
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
ENTRYPOINT ["python3", "-m", "valen_vllm.launch_server", "--host", "0.0.0.0"]
CMD []
