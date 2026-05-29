# syntax=docker/dockerfile:1
# Usage:
#   Self-contained build (default: builds from main):
#     docker buildx build -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
#
#   Self-contained build (specific git ref):
#     docker buildx build -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.6.0 --tag <registry>/nemo-rl:r0.6.0 --push .
#
#   Self-contained build (remote NeMo RL source; no need for a local clone of NeMo RL):
#     docker buildx build -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.6.0 --tag <registry>/nemo-rl:r0.6.0 --push https://github.com/NVIDIA-NeMo/RL.git
#
#   Local NeMo RL source override:
#     docker buildx build --build-context nemo-rl=. -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
#
# Optional build args to skip vLLM, SGLang, or TensorRT-LLM dependencies:
#   --build-arg SKIP_VLLM_BUILD=1    # Skip vLLM dependencies
#   --build-arg SKIP_SGLANG_BUILD=1  # Skip SGLang dependencies
#   --build-arg SKIP_TRTLLM_BUILD=1  # Skip TensorRT-LLM (defaults to skip in CI)

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04
FROM scratch AS nemo-rl
ARG NRL_GIT_REF=main
ADD --keep-git-dir=true https://github.com/NVIDIA-NeMo/RL.git#${NRL_GIT_REF} /

FROM ${BASE_IMAGE} AS base
# An environment variable to indicate that we are in a container.
ENV NRL_CONTAINER=1

# It is more convenient for users to run as root
USER root

RUN <<"EOF" bash -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles

apt-get update
apt-get install -y --no-install-recommends \
    jq \
    curl \
    git \
    rsync \
    wget \
    less \
    vim \
    python3-dev \
    libopenmpi-dev \
    libzmq3-dev \
    pkg-config \
    ninja-build \

# Nsight
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(uname -m | sed 's/aarch64/sbsa/')/7fa2af80.pub
apt update
apt install -y nsight-systems-cli

# To fix CVE-2025-68973
apt install -y --only-upgrade gnupg

apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# CMake (for sglang build)
RUN GITHUB_ARTIFACTORY=github.com \
    && CMAKE_VERSION=3.31.1 \
    && ARCH=$(uname -m) \
    && CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}" \
    && curl --retry 3 --retry-delay 2 -fsSL -o "${CMAKE_INSTALLER}.tar.gz" \
        "https://${GITHUB_ARTIFACTORY}/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" \
    && tar -xzf "${CMAKE_INSTALLER}.tar.gz" \
    && cp -r "${CMAKE_INSTALLER}/bin/"* /usr/local/bin/ \
    && cp -r "${CMAKE_INSTALLER}/share/"* /usr/local/share/ \
    && rm -rf "${CMAKE_INSTALLER}" "${CMAKE_INSTALLER}.tar.gz"

# Install uv and python
ARG UV_VERSION=0.11.6
ARG PYTHON_VERSION=3.13
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

# Disable usage stats by default for users who are sensitive to sharing usage.
# Users are encouraged to enable if the wish.
ENV RAY_USAGE_STATS_ENABLED=0
# After ray>=2.47, this feature is enabled by default which creates uv venvs for any py_executable starting with `uv run`.
# There is severe contention and performance issues with this enabled considering our dependencies are so large and occasionally
# need to be compiled, so NeMo RL has an implementation in nemo_rl/utils/venv.py that does it once per node as opposed to once per task.
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs


FROM base AS hermetic

WORKDIR /opt/nemo-rl

# Variables to control the build of TE. If there are issues with parallelization, consider
# setting these to 1.
ARG MAX_JOBS
ARG NVTE_BUILD_THREADS_PER_JOB
# Only use for custom vllm installs. Learn more at https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/use-custom-vllm.md
ARG BUILD_CUSTOM_VLLM
ARG BUILD_CUSTOM_VLLM_URL
ARG BUILD_CUSTOM_VLLM_REF
ARG BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION
# Only use for custom flashinfer installs.
ARG BUILD_CUSTOM_FLASHINFER
ARG BUILD_CUSTOM_FLASHINFER_URL
ARG BUILD_CUSTOM_FLASHINFER_REF
# Skip building vLLM, SGLang, or TensorRT-LLM dependencies (set to any non-empty value to skip)
ARG SKIP_VLLM_BUILD
ARG SKIP_SGLANG_BUILD
# Skip building TensorRT-LLM (defaults to skip in CI). Set to empty to enable.
ARG SKIP_TRTLLM_BUILD
# TRT-LLM source: must be built from source for cp313/cu13/aarch64 (no PyPI wheel).
ARG TRTLLM_GIT_URL=https://github.com/NVIDIA/TensorRT-LLM.git
ARG TRTLLM_GIT_REF=v1.3.0rc15
# modelopt source: must be built from source for cp313 (PyPI only has 0.43+ which conflicts with TRT-LLM's setuptools pin).
ARG MODELOPT_GIT_URL=https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
ARG MODELOPT_GIT_REF=0.37.0

ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy

# Ensure DeepEP is built for H100 and B200 (also mcore inference unified memory API now invokes a torch API that requires these to be set)
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0"

## CUDA 13 moved standard library headers under include/cccl; expose that path so
## builds that include <cuda/std/...> (e.g., deep_gemm) can find them.
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl

# First copy only the dependency files
COPY --from=nemo-rl pyproject.toml uv.lock ./
# Copy in the top level __init__.py/package_info.py since build-custom-vllm.sh needs the nemo_rl package to exist.
COPY --from=nemo-rl nemo_rl/__init__.py nemo_rl/package_info.py ./nemo_rl/
COPY --from=nemo-rl tools/build-custom-vllm.sh ./tools/build-custom-vllm.sh
COPY --from=nemo-rl tools/build-custom-flashinfer.sh ./tools/build-custom-flashinfer.sh
COPY --from=nemo-rl --link research/ ./research/
COPY --from=nemo-rl --link 3rdparty/ ./3rdparty/

RUN --mount=type=ssh <<"EOF" bash -exu
uv venv --seed
if [[ -n "${BUILD_CUSTOM_VLLM:-}" ]]; then
    bash tools/build-custom-vllm.sh ${BUILD_CUSTOM_VLLM_URL} ${BUILD_CUSTOM_VLLM_REF} ${BUILD_CUSTOM_VLLM_PRECOMPILED_WHEEL_LOCATION}
    source 3rdparty/vllm/nemo-rl.env
fi
if [[ -n "${BUILD_CUSTOM_FLASHINFER:-}" ]]; then
    bash tools/build-custom-flashinfer.sh ${BUILD_CUSTOM_FLASHINFER_URL} ${BUILD_CUSTOM_FLASHINFER_REF}
fi
# uv sync has a more reliable resolver than simple uv pip install which can fail

# Sync each training + inference backend one at a time (since they may conflict)
# to warm the uv cache, then at the end just sync the default dependencies.
# Do everything in one layer to prevent large layers.

# The venv is symlinked to avoid bloating the layer size
uv sync --link-mode symlink --locked --no-install-project
if [[ -z "${SKIP_VLLM_BUILD:-}" ]]; then
    uv sync --link-mode symlink --locked --extra vllm --no-install-project
fi
if [[ -z "${SKIP_SGLANG_BUILD:-}" ]]; then
    uv sync --link-mode symlink --locked --extra sglang --no-install-project
fi
uv sync --link-mode symlink --locked --extra mcore --no-install-project
uv sync --link-mode symlink --locked --extra automodel --no-install-project
uv sync --link-mode symlink --locked --extra modelopt --no-install-project
uv sync --link-mode symlink --locked --all-groups --no-install-project

# TensorRT-LLM: no cp313/cu13/aarch64 wheel exists on PyPI, must build from source.
# Skipped by default in CI (SKIP_TRTLLM_BUILD=1). Unset SKIP_TRTLLM_BUILD to enable.
if [[ -z "${SKIP_TRTLLM_BUILD:-}" ]]; then
    # 1) Sync trtllm runtime deps (mpi4py, cutlass-dsl) into the venv first.
    uv sync --link-mode symlink --locked --extra trtllm --no-install-project

    # 2) Build modelopt 0.37 from source. PyPI ships modelopt 0.43+ for cp313, but
    #    0.43+ requires setuptools>=80 which conflicts with TRT-LLM's `setuptools<80`
    #    pin. v0.37 doesn't have that conflict — build a cp313 wheel against our venv.
    git clone --depth=1 --branch "${MODELOPT_GIT_REF}" "${MODELOPT_GIT_URL}" /tmp/modelopt
    /opt/nemo_rl_venv/bin/python3 -m pip wheel --no-deps -w /tmp/modelopt-wheel /tmp/modelopt
    /opt/nemo_rl_venv/bin/python3 -m pip install --no-deps --force-reinstall /tmp/modelopt-wheel/*.whl
    rm -rf /tmp/modelopt /tmp/modelopt-wheel

    # 3) Clone TRT-LLM and apply cp313-required patches before building:
    #    (a) `nvidia-modelopt[torch]~=0.37.0` → `>=0.37.0` — our 0.37 install above satisfies it
    #    (b) `setuptools<80` → unpin — modelopt 0.37 doesn't need <80, but the pin breaks resolver
    #    (c) CUTLASS CMakeLists `setup_library.py develop --user` → bash wrapper that
    #        first `cp setup_library.py setup.py` (modern setuptools' develop shells out
    #        to `pip install -e .` which needs setup.py to exist) and drops `--user`
    #        (invalid inside a venv).
    git clone --depth=1 --branch "${TRTLLM_GIT_REF}" "${TRTLLM_GIT_URL}" /tmp/TensorRT-LLM
    pushd /tmp/TensorRT-LLM
    git submodule update --init --recursive --depth=1
    sed -i 's|nvidia-modelopt\[torch\]~=0\.37\.0|nvidia-modelopt[torch]>=0.37.0|' requirements.txt
    sed -i 's|^setuptools<80$|setuptools|' requirements.txt
    sed -i 's|COMMAND \${Python3_EXECUTABLE} setup_library.py develop --user|COMMAND bash -c "cp -f setup_library.py setup.py \&\& \${Python3_EXECUTABLE} setup_library.py develop"|' \
        cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt

    # 4) Build the wheel WITHOUT --install (which would do editable install), then
    #    pip install the wheel non-editable. This way the venv has frozen source — no
    #    drift if /tmp/TensorRT-LLM is removed or its source changes.
    . /opt/nemo_rl_venv/bin/activate
    python3 scripts/build_wheel.py \
        -a "100-real" \
        -G Ninja \
        --clean \
        --use_ccache \
        --nvrtc_dynamic_linking \
        -D "ENABLE_UCX=OFF"
    pip install --no-deps --force-reinstall build/tensorrt_llm-*.whl
    deactivate
    popd
    rm -rf /tmp/TensorRT-LLM   # safe — wheel is now copied into venv site-packages

    # 5) CUDA lib symlinks. PyPI nvidia-cu13 wheels ship libnvJitLink/libcublas/
    #    libcublasLt that lack symbols required by TE/trtllm at runtime. System
    #    /usr/local/cuda-13.2/ has the complete versions.
    for spec in \
        "libnvJitLink.so.13.2.51:libnvJitLink.so.13" \
        "libcublas.so.13.3.0.5:libcublas.so.13" \
        "libcublasLt.so.13.3.0.5:libcublasLt.so.13"; do
        SYS_NAME="${spec%%:*}"
        SHORT_NAME="${spec##*:}"
        SYS=/usr/local/cuda-13.2/targets/sbsa-linux/lib/${SYS_NAME}
        VENV_F=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cu13/lib/${SHORT_NAME}
        if [ -f "$SYS" ] && [ -e "$VENV_F" ]; then
            ln -sf "$SYS" "$VENV_F"
        fi
    done

    # 6) trtllm_py_wrapper.sh: Ray spawns trtllm actors as plain python subprocesses,
    #    but v13's OpenMPI 4.1.9a1 can't bootstrap singleton MPI (opal_init fails).
    #    Wrap each actor with `mpirun -n 1` so MPI initializes as a 1-rank world.
    #    Must clear outer PMIX env (from srun --mpi=pmix) so the inner mpirun isn't
    #    refused as "recursive". orchestrator_type="ray" handles cross-rank comms.
    cat > /opt/nemo-rl/trtllm_py_wrapper.sh <<'WRAPPER_EOF'
#!/bin/bash
for v in $(env | grep -oE '^(OMPI|PMI|PMIX|MV2)_[A-Za-z0-9_]+'); do
    case "$v" in
        OMPI_ALLOW_RUN_AS_ROOT|OMPI_ALLOW_RUN_AS_ROOT_CONFIRM) continue ;;
        *) unset "$v" ;;
    esac
done
exec /usr/local/mpi/bin/mpirun -n 1 --allow-run-as-root /opt/nemo_rl_venv/bin/python3 "$@"
WRAPPER_EOF
    chmod +x /opt/nemo-rl/trtllm_py_wrapper.sh

    # 7) Per-actor venv symlinks. nemo-rl creates per-actor venvs at runtime by running
    #    `uv run --locked --extra mcore` — which would re-trigger deep_ep compilation
    #    and hit the CCCL include issue. Pre-create symlinks so the `_env_builder`'s
    #    `python_path.exists()` check short-circuits.
    mkdir -p /opt/ray_venvs
    for actor in \
        nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker \
        nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker \
        nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2 \
    ; do
        ln -sfn /opt/nemo_rl_venv /opt/ray_venvs/${actor}
    done
fi

# Remove the aiohttp in this uv cache dir to fully address CVE GHSA-mqqc-3gqh-h2x8
# The ray install will include the older aiohttp version in its cache
find /root/.cache/uv -type d -path "*ray/_private/runtime_env/agent/thirdparty_files/aiohttp*" -exec rm -rf {} +
EOF

# Point Ray's TRT-LLM actor launcher at the mpirun wrapper above. Only takes effect
# at runtime; if SKIP_TRTLLM_BUILD was set the file won't exist and Ray will fail
# fast if it tries to use the wrapper (intended — fail-loud is better than silently
# spawning actors that crash on `import tensorrt_llm`).
ENV NEMO_RL_PY_EXECUTABLES_TRTLLM=/opt/nemo-rl/trtllm_py_wrapper.sh
# OpenMPI flags for running as root inside enroot/docker containers.
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

ENV PATH="/opt/nemo_rl_venv/bin:$PATH"
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs
# Point TE at the pip-installed cuDNN (nvidia-cudnn-cu12) instead of the system one.
# The container's system cuDNN (/lib/x86_64-linux-gnu/libcudnn*.so.9) may be a different
# version than what pip installed. TE prioritizes system libraries by default, causing a
# version mismatch crash (e.g. system 9.10.1 vs pip 9.19.0). CUDNN_HOME makes TE's Python
# code find the pip version first, and LD_LIBRARY_PATH makes the dynamic linker resolve
# cuDNN sub-libraries from pip when loading libtransformer_engine.so.
ENV CUDNN_HOME=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn
ENV LD_LIBRARY_PATH="/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
# Verify with: python -c "import transformer_engine.pytorch as te; print(te.get_cudnn_version())"

WORKDIR /opt/nemo-rl

FROM hermetic AS release

# Re-declare build args for this stage
ARG SKIP_VLLM_BUILD
ARG SKIP_SGLANG_BUILD
ARG NEMO_RL_COMMIT
ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ARG RC_DATE=00.00
ARG TARGETARCH
ENV NEMO_RL_COMMIT=${NEMO_RL_COMMIT:-<unknown>}
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
ENV NVIDIA_BUILD_REF=${NVIDIA_BUILD_REF:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

ENV NEMO_RL_VENV_DIR=/opt/ray_venvs

# Copy in source from build context (defaults to cloned repo, can be overridden)
# Exclude pyproject.toml and uv.lock since those may be altered by build-custom-vllm.sh
COPY --from=nemo-rl --exclude=pyproject.toml --exclude=uv.lock . /opt/nemo-rl
# Unshallow the repo to get the full history (in the case it was from the scratch layer).
# Potentially not necessary if the repo is passed in as a complete repository (w/ full git history),
# so do a quick check before trying to unshallow.
RUN git rev-parse --is-shallow-repository | grep -q true && git fetch --unshallow || true
RUN <<"EOF" bash -exu
NEGATIVE_FILTERS=""
if [[ -n "${SKIP_VLLM_BUILD:-}" ]]; then
    NEGATIVE_FILTERS="$NEGATIVE_FILTERS vllm"
fi
if [[ -n "${SKIP_SGLANG_BUILD:-}" ]]; then
    NEGATIVE_FILTERS="$NEGATIVE_FILTERS sglang"
fi
if [[ -n "$NEGATIVE_FILTERS" ]]; then
    UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py --negative-filters $NEGATIVE_FILTERS
else
    UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py
fi
EOF

# Generate container fingerprint for frozen environment support
# Store outside /opt/nemo-rl to avoid being overwritten by user mounts
RUN python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

# NOTICES.txt file points to where the OSS source code is archived
RUN echo "This distribution includes open source which is archived at the following URL: https://opensource.nvidia.com/oss/teams/nvidia/nemo-rl/${RC_DATE}:linux-${TARGETARCH}/index.html" > NOTICES.txt && \
    echo "For further inquiries or assistance, contact us at oss-requests@nvidia.com" >> NOTICES.txt
