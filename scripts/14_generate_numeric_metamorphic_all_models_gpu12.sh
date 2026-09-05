#!/usr/bin/env bash
set -Eeuo pipefail

# Generate certified numeric-metamorphic teacher pools for GSM8K and SVAMP.
# Three base models run serially; within each model, two question shards run in
# parallel on physical GPUs 1 and 2.  Outputs are written directly into the
# existing per-model generated_data roots, touching only gsm8k/ and svamp/.
# The seven MCQ dataset directories are never modified.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/home/luorongchuan/workspace_135/models}"
QWEN_MODEL="${QWEN_MODEL:-${MODEL_ROOT}/Qwen2.5-7B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-${MODEL_ROOT}/Llama-3.1-8B-Instruct}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"

GPU_FIRST="${GPU_FIRST:-1}"
GPU_SECOND="${GPU_SECOND:-2}"
MAX_QUESTIONS="${MAX_QUESTIONS:-1000}"
DATASETS="${DATASETS:-gsm8k svamp}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/relacats_v1/outputs/generated_data}"
LOG_BASE="${LOG_BASE:-${ROOT_DIR}/relacats_v1/outputs/logs/numeric_metamorphic_gpu${GPU_FIRST}${GPU_SECOND}}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
QUESTION_BATCH_SIZE="${QUESTION_BATCH_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TEMPERATURE="${TEMPERATURE:-0.8}"
CONFIDENCE_TEMPERATURE="${CONFIDENCE_TEMPERATURE:-0.0}"
SEED="${SEED:-42}"

HF_HOME="${HF_HOME:-/home/luorongchuan/workspace_135/datasets/.hf_cache_relacats_v1}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python not found or not executable: ${PYTHON_BIN}"
[[ "${GPU_FIRST}" =~ ^[0-9]+$ && "${GPU_SECOND}" =~ ^[0-9]+$ ]] || \
  fail "GPU_FIRST/GPU_SECOND must be non-negative integers"
[[ "${GPU_FIRST}" != "${GPU_SECOND}" ]] || fail "GPU_FIRST and GPU_SECOND must differ"
[[ "${MAX_QUESTIONS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_QUESTIONS must be positive"

MODEL_TAGS=(
  qwen2_5_7b_instruct
  llama3_1_8b_instruct
  deepseek_r1_distill_qwen_1_5b
)
MODEL_PATHS=(
  "${QWEN_MODEL}"
  "${LLAMA_MODEL}"
  "${DEEPSEEK_MODEL}"
)

for model_path in "${MODEL_PATHS[@]}"; do
  [[ -f "${model_path}/config.json" ]] || fail "Local model not found: ${model_path}"
done

mkdir -p "${OUTPUT_BASE}" "${LOG_BASE}"
exec 9>"${LOG_BASE}/.numeric_metamorphic_all_models.lock"
flock -n 9 || fail "another numeric metamorphic launcher is using ${LOG_BASE}"

if [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  for gpu in "${GPU_FIRST}" "${GPU_SECOND}"; do
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    [[ -z "${pids}" ]] || fail "physical GPU ${gpu} is busy (PID(s): ${pids})"
  done
fi

PYTHON_ENV_BIN="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
export PATH="${PYTHON_ENV_BIN}:${PATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME HF_DATASETS_CACHE HF_ENDPOINT

log() {
  echo "[$(date '+%F %T %z')] $*"
}

run_worker() {
  local gpu="$1"
  local shard="$2"
  local model_path="$3"
  local output_root="$4"
  local log_file="$5"

  setsid env CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -m relacats_v1.data_creation.generate_numeric_metamorphic_data \
      --model-name "${model_path}" \
      --datasets ${DATASETS} \
      --split train \
      --max-questions "${MAX_QUESTIONS}" \
      --output-root "${output_root}" \
      --temperature "${TEMPERATURE}" \
      --confidence-temperature "${CONFIDENCE_TEMPERATURE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --question-batch-size "${QUESTION_BATCH_SIZE}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --tensor-parallel-size 1 \
      --seed "${SEED}" \
      --num-shards 2 \
      --shard-index "${shard}" \
      >"${log_file}" 2>&1 &
  echo $!
}

cleanup_pids=()
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if (( status != 0 )); then
    for pid in "${cleanup_pids[@]:-}"; do
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    done
  fi
  for pid in "${cleanup_pids[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

log "Numeric metamorphic generation starting"
log "GPUs=${GPU_FIRST},${GPU_SECOND}"
log "datasets=${DATASETS}"
log "output_base=${OUTPUT_BASE}"
log "identity-only archive is expected to remain separate and is not touched"

for index in "${!MODEL_TAGS[@]}"; do
  tag="${MODEL_TAGS[$index]}"
  model_path="${MODEL_PATHS[$index]}"
  output_root="${OUTPUT_BASE}/${tag}"
  model_log_root="${LOG_BASE}/${tag}"
  mkdir -p "${output_root}" "${model_log_root}"

  log "===== START ${tag} ====="
  log "model=${model_path}"
  log "output=${output_root}"

  pid0="$(run_worker "${GPU_FIRST}" 0 "${model_path}" "${output_root}" "${model_log_root}/shard0_gpu${GPU_FIRST}.log")"
  pid1="$(run_worker "${GPU_SECOND}" 1 "${model_path}" "${output_root}" "${model_log_root}/shard1_gpu${GPU_SECOND}.log")"
  cleanup_pids=("${pid0}" "${pid1}")

  set +e
  wait "${pid0}"; status0=$?
  wait "${pid1}"; status1=$?
  set -e
  cleanup_pids=()

  if (( status0 != 0 || status1 != 0 )); then
    log "FAILED ${tag}: shard0=${status0}, shard1=${status1}"
    log "See ${model_log_root}/shard0_gpu${GPU_FIRST}.log and shard1_gpu${GPU_SECOND}.log"
    exit 1
  fi

  # Lightweight postcondition: both dataset directories must now exist.
  for dataset in ${DATASETS}; do
    qdir="${output_root}/${dataset}/questions"
    [[ -d "${qdir}" ]] || fail "missing output directory after ${tag}: ${qdir}"
    count="$(find "${qdir}" -maxdepth 1 -type f -name '*.json' | wc -l)"
    log "${tag}/${dataset}: ${count} question JSON files"
  done

  log "===== COMPLETE ${tag} ====="
done

log "All three models complete."
