#!/bin/bash
# =============================================================================
# Launch the NanoV3.5 SWE rollout benchmark (generation-only, no training)
# on Lyris (GB200). Adapted from run_grpo_nanov35_swe_trtllm.sh.
#
# Geometry: R generation nodes only (default 2), TP4 per replica, 4 GPUs/node.
# One "step" rolls out PPS=GBS/GPP prompts x GPP generations = GBS trajectories
# through eval (examples/nemo_gym/run_grpo_rollout_benchmark.py).
#
# Usage:
#   BACKEND=trtllm bash examples/nemo_gym/run_rollout_benchmark_lyris.sh
#   BACKEND=vllm  R=2 MAX_TURNS=60 RUN_IDX=2 bash examples/nemo_gym/run_rollout_benchmark_lyris.sh
#   DRY_RUN=1 BACKEND=trtllm bash examples/nemo_gym/run_rollout_benchmark_lyris.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Shared read-only assets (container/model/data) — reference in place.
SHARED=/lustre/fsw/coreai_comparch_trtllm/shikiw
# Your writable outputs (caches/secrets/gym venvs/wandb staging). Defaults to
# your own account dir; override with MY_DIR=... to relocate.
MY_DIR="${MY_DIR:-/lustre/fsw/coreai_comparch_trtllm/${USER}}"

# ----- backend (required: env or first positional arg) -----------------------
BACKEND="${BACKEND:-${1:-}}"
if [ "${BACKEND}" != "trtllm" ] && [ "${BACKEND}" != "vllm" ]; then
  echo "ERROR: set BACKEND=trtllm or BACKEND=vllm (env var or first arg)"; exit 1
fi

# ----- benchmark knobs (rollout-benchmark doc defaults, all overridable) ------
R="${R:-2}"                          # generation node count (= all nodes)
NUM_GPU=4
GBS="${GBS:-64}"                     # trajectories per benchmark step
GPP="${GPP:-16}"                     # generations per prompt
PPS=$((GBS / GPP))                   # prompts per step (default 4)
CONCURRENCY="${CONCURRENCY:-$((PPS * GPP))}"   # = GBS; NOT the training 2x
MAX_TURNS="${MAX_TURNS:-60}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-1200}"
VAL_PATH="${VAL_PATH:-${SHARED}/data/swe_val_20inst_rollout_bench.jsonl}"
# Defaults to a timestamp so every launch gets a fresh log dir / job name /
# W&B run — no singleton queue collisions, no re-locking or appending into a
# previous run's directory. Pass RUN_IDX=<n> explicitly for numbered series
# (e.g. 5-run mean±std sets).
RUN_IDX="${RUN_IDX:-$(date +%m%d-%H%M%S)}"
GEN_TP=4
WALLTIME="${WALLTIME:-04:00:00}"

# ----- rollout-only geometry --------------------------------------------------
NUM_GEN_NODES="${R}"
TOTAL_NODES="${NUM_GEN_NODES}"       # no train nodes
# Same two-layer topology alignment as the training launcher (sbatch --segment
# + cluster.segment_size); with gen-only nodes the segment is just R.
SBATCH_SEGMENT="${TOTAL_NODES}"

# ----- paths / artifacts ------------------------------------------------------
# Same image for both backends (control-arm parity, as in the training launcher).
CONTAINER="${CONTAINER:-${SHARED}/images/nemo-rl-genonly-v2-trtllm-rc24-vllm025-aarch64-20260810.sqsh}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/examples/nemo_gym/grpo_nanov35_swe_${BACKEND}.yaml}"
# Gym venvs: your own build (either naming from the handoff guide works).
if [ -z "${NEMO_GYM_VENV_DIR:-}" ]; then
  for _d in "${MY_DIR}/gym_venvs_rollout" "${MY_DIR}/gym_venvs_rlmain"; do
    [ -d "${_d}" ] && NEMO_GYM_VENV_DIR="${_d}" && break
  done
  NEMO_GYM_VENV_DIR="${NEMO_GYM_VENV_DIR:-${MY_DIR}/gym_venvs_rollout}"
fi
# Ray worker venvs: the image-baked ones embed shikiw's repo path and only
# work for shikiw — everyone else gets a per-user Lustre dir (built once on
# first launch, reused afterwards; see handoff Step 3 for the wheel-cache seed).
if [ "${USER}" != "shikiw" ]; then
  NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-${MY_DIR}/ray_venvs_rc24}"
  # Must be exported: the ray worker venv builder runs under the RAYLET's
  # environment (started by ray.sub inside the container), not the driver's.
  # sbatch exports the submission env; pyxis carries it into the container.
  export NEMO_RL_VENV_DIR
  # The OpenHands framework repo must be OWNED by the runner: the one-time
  # setup git-clones it, and git refuses cross-user sources ("detected
  # dubious ownership"). Make a personal copy once (242 MB) and point the
  # recipe at it (train + val agents).
  OPENHANDS_REPO="${OPENHANDS_REPO:-${MY_DIR}/nv-OpenHands}"
  if [ ! -d "${OPENHANDS_REPO}/.git" ]; then
    echo "One-time copy of the OpenHands framework repo -> ${OPENHANDS_REPO} ..."
    rsync -a "${SHARED}/nv-OpenHands/" "${OPENHANDS_REPO}/"
  fi
  EXTRA_ARGS="${EXTRA_ARGS:-} \
env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_framework_repo=${OPENHANDS_REPO} \
env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_framework_repo=${OPENHANDS_REPO}"
fi
RAY_SUB="${REPO_ROOT}/ray.sub"

# ----- naming / dirs ----------------------------------------------------------
EXP_NAME="${EXP_NAME:-rollout-bench-${BACKEND}-run${RUN_IDX}}"
WANDB_PROJ="${WANDB_PROJ:-nemorl-mlperf-${USER}}"
WANDB_NAME="${WANDB_NAME:-nanov35-rollout-bench-${BACKEND}-mt${MAX_TURNS}-run${RUN_IDX}}"
WANDB_GROUP="nanov35-rollout-bench"
BASE_LOG_DIR="${REPO_ROOT}/logs/${EXP_NAME}"
RUN_LOG_DIR="${BASE_LOG_DIR}"
NEMO_LOG_DIR="${BASE_LOG_DIR}"
mkdir -p "${RUN_LOG_DIR}"
chmod 700 "${BASE_LOG_DIR}" || true

# ----- caches (persistent, per-experiment) ------------------------------------
PERSISTENT_CACHE="${MY_DIR}/nemo_rl_cache/${EXP_NAME}"
mkdir -p "${PERSISTENT_CACHE}/uv" "${PERSISTENT_CACHE}/inductor" "${PERSISTENT_CACHE}/triton" "${PERSISTENT_CACHE}/gym_uv"
HF_HOME="${MY_DIR}/hf_home"
WANDB_STAGE="${MY_DIR}/wandb_stage/${EXP_NAME}"
mkdir -p "${HF_HOME}" "${WANDB_STAGE}"

# ----- secrets ------------------------------------------------------------------
if [ -f "${MY_DIR}/.secrets/nemo-rl.env" ]; then
  set -a; source "${MY_DIR}/.secrets/nemo-rl.env"; set +a
fi

# ----- TRT-LLM logging ------------------------------------------------------------
# print_iter_log lines emit at logger.info; TRT-LLM's default level is "error".
export TLLM_LOG_LEVEL="${TLLM_LOG_LEVEL:-INFO}"

# ----- nsys profiling -------------------------------------------------------------
# nsys-wraps the TRT-LLM GPU workers via NeMo-RL's Ray nsight injection; capture
# covers executor iterations TLLM_PROFILE_START_STOP (cudaProfilerApi). Reports
# land in ${BASE_LOG_DIR}/<jobid>-logs/ray/**/logs/nsight/. NSYS=0 for score runs.
NSYS="${NSYS:-1}"
if [ "${NSYS}" != 0 ] && [ "${NSYS}" != 1 ]; then
  echo "ERROR: NSYS must be 0 or 1"; exit 1
fi
if [ "${NSYS}" = 1 ]; then
  TLLM_PROFILE_START_STOP="${TLLM_PROFILE_START_STOP:-1000-1050}"
  if ! [[ "${TLLM_PROFILE_START_STOP}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    echo "ERROR: TLLM_PROFILE_START_STOP must be one start-stop executor range (e.g. 1000-1050)"; exit 1
  fi
  PROFILE_START_ITER="${BASH_REMATCH[1]}"
  PROFILE_STOP_ITER="${BASH_REMATCH[2]}"
  if (( 10#${PROFILE_START_ITER} >= 10#${PROFILE_STOP_ITER} )); then
    echo "ERROR: TLLM_PROFILE_START_STOP must be a non-empty half-open range"; exit 1
  fi
  export NRL_NSYS_WORKER_PATTERNS="trtllm_async_generation_worker"
  # Required alongside WORKER_PATTERNS; report label only — capture is gated
  # by TLLM_PROFILE_START_STOP.
  export NRL_NSYS_PROFILE_STEP_RANGE="${PROFILE_START_ITER}:${PROFILE_STOP_ITER}"
  if [ -z "${NRL_NSYS_EXTRA_OPTIONS:-}" ]; then
    # End the capture range without terminating the worker.
    NRL_NSYS_EXTRA_OPTIONS='{"capture-range-end":"repeat:1:async"}'
  fi
  export NRL_NSYS_EXTRA_OPTIONS
  export TLLM_PROFILE_START_STOP
  # Reports appear on node-local /tmp only at worker exit: the log-sync sidecar
  # (off unless RAY_LOG_SYNC_FREQUENCY is set) copies them out during the
  # teardown grace, before the job ends.
  export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"
  NSYS_TEARDOWN_GRACE="${NSYS_TEARDOWN_GRACE:-120}"
else
  NSYS_TEARDOWN_GRACE=0
  unset NRL_NSYS_WORKER_PATTERNS NRL_NSYS_PROFILE_STEP_RANGE NRL_NSYS_EXTRA_OPTIONS TLLM_PROFILE_START_STOP
fi

# ----- mounts -------------------------------------------------------------------
export MOUNTS="${MOUNTS:-/lustre:/lustre,/dev/fuse:/dev/fuse}"
export CONTAINER
export GPUS_PER_NODE="${NUM_GPU}"

export COMMAND="cd ${REPO_ROOT} && \
trap 'touch ${BASE_LOG_DIR}/\${SLURM_JOB_ID}-logs/ENDED 2>/dev/null || true' EXIT && \
date && \
OMP_NUM_THREADS=16 \
TRTLLM_USE_MAMBA_FI_SSD=${TRTLLM_USE_MAMBA_FI_SSD:-0} \
NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-false} \
${NEMO_RL_VENV_DIR:+NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR}} \
RAY_DEDUP_LOGS=1 \
TORCHINDUCTOR_CACHE_DIR=${PERSISTENT_CACHE}/inductor \
TRITON_CACHE_DIR=${PERSISTENT_CACHE}/triton \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
UV_LOCK_TIMEOUT=1200 \
NEMO_GYM_VENV_DIR=${NEMO_GYM_VENV_DIR} \
TRTLLM_WHEEL_CACHE_DIR=${MY_DIR}/trtllm_wheel_cache \
HF_HOME=${HF_HOME} \
HF_TOKEN=\${HF_TOKEN:-} \
WANDB_API_KEY=\${WANDB_API_KEY:-} \
WANDB_DIR=${WANDB_STAGE} \
WANDB_CACHE_DIR=${WANDB_STAGE}/cache \
WANDB_DATA_DIR=${WANDB_STAGE}/data \
uv run ./examples/nemo_gym/run_grpo_rollout_benchmark.py \
  --config ${CONFIG_PATH} \
  grpo.num_prompts_per_step=${PPS} \
  grpo.num_generations_per_prompt=${GPP} \
  data.train.data_path=${VAL_PATH} \
  data.validation.data_path=${VAL_PATH} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=${MAX_TURNS} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.concurrency=${CONCURRENCY} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=${MAX_TURNS} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=${AGENT_TIMEOUT} \
  env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.concurrency=${CONCURRENCY} \
  policy.generation.${BACKEND}_cfg.tensor_parallel_size=${GEN_TP} \
  policy.generation.colocated.enabled=False \
  policy.generation.colocated.resources.num_nodes=${NUM_GEN_NODES} \
  policy.generation.colocated.resources.gpus_per_node=${NUM_GPU} \
  cluster.num_nodes=${TOTAL_NODES} \
  cluster.gpus_per_node=${NUM_GPU} \
  cluster.segment_size=${SBATCH_SEGMENT} \
  checkpointing.enabled=False \
  logger.log_dir=${NEMO_LOG_DIR} \
  logger.wandb_enabled=True \
  logger.wandb.name=${WANDB_NAME} \
  logger.wandb.project=${WANDB_PROJ} \
  ++logger.wandb.group=${WANDB_GROUP} \
  ${EXTRA_ARGS:-}; NRL_BENCH_RC=\$?; sleep ${NSYS_TEARDOWN_GRACE}; exit \${NRL_BENCH_RC}"

SBATCH_ARGS=(
  --segment="${SBATCH_SEGMENT}"
  --nodes="${TOTAL_NODES}"
  --account="coreai_comparch_trtllm"
  --job-name="${WANDB_NAME}"
  --partition="gb200"
  --time="${WALLTIME}"
  --exclusive
  --mem=0
  --dependency=singleton
  --output="${RUN_LOG_DIR}/slurm-%j.out"
)

echo "Config:    ${CONFIG_PATH}"
echo "Container: ${CONTAINER}"
echo "Geometry:  ${TOTAL_NODES} gen-only nodes (${BACKEND}-TP${GEN_TP}); PPS=${PPS} GPP=${GPP} GBS=${GBS} concurrency=${CONCURRENCY}"
echo "Agents:    max_turns=${MAX_TURNS} timeout=${AGENT_TIMEOUT}s data=${VAL_PATH}"
echo "WandB:     ${WANDB_PROJ}/${WANDB_NAME}"
echo "Gym venvs: ${NEMO_GYM_VENV_DIR}"
echo "Ray venvs: ${NEMO_RL_VENV_DIR:-<image-baked (only valid for shikiw)>}"
if [ "${NSYS}" = 1 ]; then
  echo "Nsys:      ON — executor iters ${TLLM_PROFILE_START_STOP}, $((R * GEN_TP)) worker reports -> ${BASE_LOG_DIR}/<jobid>-logs/ray/**/logs/nsight/"
else
  echo "Nsys:      off"
fi
[ -f "${CONTAINER}" ] || { echo "ERROR: container missing: ${CONTAINER}"; exit 1; }
[ -f "${VAL_PATH}" ] || { echo "ERROR: benchmark data missing: ${VAL_PATH}"; exit 1; }
[ -d "${NEMO_GYM_VENV_DIR}" ] || { echo "ERROR: gym venvs missing: ${NEMO_GYM_VENV_DIR}"; exit 1; }

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] sbatch ${SBATCH_ARGS[*]} ${RAY_SUB}"
  exit 0
fi

SBATCH_OUTPUT="$(sbatch "${SBATCH_ARGS[@]}" "${RAY_SUB}")"
echo "${SBATCH_OUTPUT}"
JOB_ID="$(echo "${SBATCH_OUTPUT}" | grep -oE '[0-9]+' | tail -1)"
echo "Job ID: ${JOB_ID}"
chmod 700 "${RUN_LOG_DIR}" 2>/dev/null || true
