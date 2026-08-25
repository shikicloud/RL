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
GEN_TP="${GEN_TP:-4}"
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
# Exported so ray.sub places <jobid>-logs in the per-experiment dir.
export BASE_LOG_DIR
RUN_LOG_DIR="${BASE_LOG_DIR}"
NEMO_LOG_DIR="${BASE_LOG_DIR}"
mkdir -p "${RUN_LOG_DIR}"
chmod 750 "${BASE_LOG_DIR}" || true

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

# ----- post-run results archiving ---------------------------------------------
# Each rollout batch leaves ~9.7k small files (320 instance dirs x ~30 entries)
# under the Gym results dir, on a Lustre project where the INODE quota — not
# bytes — is the contended resource. After a SUCCESSFUL run, the COMMAND tail
# packs each batch this job created into a single .tar.gz (tar-verified before
# the source is deleted; a quiesce re-check skips batches a concurrent live job
# is still writing into). Failed runs are left unpacked for debugging.
# NRL_ARCHIVE_RESULTS=0 keeps raw directories.
#
# The helper script is generated below (after the DRY_RUN gate, same pattern as
# ray.sub's driver_command.sh) so this launcher stays one self-contained file.
# A copy lands in each run's log dir and works standalone for backlog cleanup:
#   bash <logdir>/archive_swe_results.sh <results-dir> 0
NRL_ARCHIVE_RESULTS="${NRL_ARCHIVE_RESULTS:-1}"
SWE_RESULTS_DIR="${REPO_ROOT}/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/results"
ARCHIVE_HELPER="${RUN_LOG_DIR}/archive_swe_results.sh"

export COMMAND="cd ${REPO_ROOT} && \
NRL_JOB_START=\$(date +%s) && \
# SLURM_JOB_ID is absent inside the head container; driver_command.sh lives
# in LOG_DIR, so derive the ENDED path from \$0.
trap 'touch \$(dirname \$0)/ENDED 2>/dev/null || true' EXIT && \
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
  ${EXTRA_ARGS:-}; NRL_BENCH_RC=\$?; \
if [ ${NRL_ARCHIVE_RESULTS} -eq 1 ] && [ \${NRL_BENCH_RC} -eq 0 ]; then \
  bash ${ARCHIVE_HELPER} ${SWE_RESULTS_DIR} \${NRL_JOB_START} || true; \
fi; \
sleep ${NSYS_TEARDOWN_GRACE}; exit \${NRL_BENCH_RC}"

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

# Generate the archiver the COMMAND tail invokes. Placed after the DRY_RUN
# gate so dry runs leave no stray helper in the log dir.
cat > "${ARCHIVE_HELPER}" <<'ARCHIVE_EOF'
#!/bin/bash
# Archive completed SWE-bench result batches into single .tar.gz files.
# Generated by run_rollout_benchmark_lyris.sh; safe to run standalone:
#   archive_swe_results.sh RESULTS_DIR [SINCE_EPOCH] [QUIESCE_SECS]
#   SINCE_EPOCH   only touch batches whose dir mtime >= this (0 = all/backlog).
#   QUIESCE_SECS  a batch must be write-free this long before it is packed
#                 (default 120) — guards batches a concurrent live job owns.
# A batch is only removed after `tar -tzf` verifies the archive and its entry
# count matches the source tree. Nothing here deletes an unverified byte.
set -euo pipefail

RESULTS_DIR="${1:?usage: $0 RESULTS_DIR [SINCE_EPOCH] [QUIESCE_SECS]}"
SINCE_EPOCH="${2:-0}"
QUIESCE_SECS="${3:-120}"

[ -d "${RESULTS_DIR}" ] || { echo "[archive] no results dir at ${RESULTS_DIR}; nothing to do"; exit 0; }

archived=0 skipped=0 freed=0

for dir in "${RESULTS_DIR}"/swebench_results_*/; do
  [ -d "${dir}" ] || continue
  batch="${dir%/}"
  name="$(basename "${batch}")"

  if [ -e "${batch}.tar.gz" ]; then
    echo "[archive] skip ${name}: archive already exists"
    skipped=$((skipped + 1)); continue
  fi

  # Window filter: batch root mtime updates on each instance-dir creation, so
  # a batch produced by this job has mtime >= the job's start.
  dir_mtime=$(stat -c %Y "${batch}")
  if [ "${dir_mtime}" -lt "${SINCE_EPOCH}" ]; then
    echo "[archive] skip ${name}: predates this job (pass SINCE_EPOCH=0 to archive backlog)"
    skipped=$((skipped + 1)); continue
  fi

  # Quiesce filter: any write within QUIESCE_SECS means some run may still be
  # producing into this batch. The batch THIS job just produced always looks
  # young here (the driver aggregates and exits within seconds of the last
  # instance write), so on a too-young batch wait out the remainder and
  # re-check ONCE: our own batch goes quiet and passes; a batch a concurrent
  # live job is writing keeps getting fresh writes and stays skipped.
  # awk max instead of `sort | head -1`: head closing the pipe early raises
  # SIGPIPE in sort, which pipefail turns into rc=141 and set -e turns fatal.
  for attempt in 1 2; do
    newest=$(find "${batch}" -printf '%T@\n' 2>/dev/null \
             | awk 'BEGIN{m=0} $1>m{m=$1} END{printf "%d", m}')
    age=$(( $(date +%s) - newest ))
    [ "${age}" -ge "${QUIESCE_SECS}" ] && break
    if [ "${attempt}" -eq 1 ]; then
      wait_s=$(( QUIESCE_SECS - age + 5 ))
      echo "[archive] ${name}: written ${age}s ago; waiting ${wait_s}s to confirm it is quiescent..."
      sleep "${wait_s}"
    fi
  done
  if [ "${age}" -lt "${QUIESCE_SECS}" ]; then
    echo "[archive] skip ${name}: still being written (${age}s ago) — a live run owns it"
    skipped=$((skipped + 1)); continue
  fi

  # One archiver per batch across concurrent jobs.
  lock="${batch}.archiving.lock"
  exec 9>"${lock}"
  if ! flock -n 9; then
    echo "[archive] skip ${name}: another archiver holds the lock"
    skipped=$((skipped + 1)); continue
  fi

  src_count=$(find "${batch}" | wc -l)
  tmp="${batch}.tar.gz.tmp"
  echo "[archive] packing ${name} (${src_count} inodes)..."
  if ! tar -C "${RESULTS_DIR}" -czf "${tmp}" "${name}"; then
    echo "[archive] ERROR: tar failed for ${name}; source left untouched"
    rm -f "${tmp}" "${lock}"; continue
  fi
  # Verify readability and completeness before deleting anything.
  tar_count=$(tar -tzf "${tmp}" | wc -l)
  if [ "${tar_count}" -ne "${src_count}" ]; then
    echo "[archive] ERROR: ${name} entry mismatch (tar=${tar_count} src=${src_count}); source left untouched"
    rm -f "${tmp}" "${lock}"; continue
  fi

  mv "${tmp}" "${batch}.tar.gz"
  rm -rf "${batch}"
  rm -f "${lock}"
  echo "[archive] done ${name}: ${src_count} inodes -> 1"
  archived=$((archived + 1))
  freed=$((freed + src_count - 1))
done

echo "[archive] summary: ${archived} batch(es) archived, ${skipped} skipped, ~${freed} inodes freed"
ARCHIVE_EOF
chmod +x "${ARCHIVE_HELPER}"

SBATCH_OUTPUT="$(sbatch "${SBATCH_ARGS[@]}" "${RAY_SUB}")"
echo "${SBATCH_OUTPUT}"
JOB_ID="$(echo "${SBATCH_OUTPUT}" | grep -oE '[0-9]+' | tail -1)"
echo "Job ID: ${JOB_ID}"
chmod 750 "${RUN_LOG_DIR}" 2>/dev/null || true
