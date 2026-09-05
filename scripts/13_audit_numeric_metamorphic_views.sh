#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/luorongchuan/miniconda3/envs/FAD_OPD/bin/python}"
MAX_QUESTIONS="${MAX_QUESTIONS:-1000}"
PREVIEW_PER_DATASET="${PREVIEW_PER_DATASET:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/relacats_v1/outputs/numeric_metamorphic_candidates}"
DATASETS=( ${DATASETS:-gsm8k svamp} )

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python not found: ${PYTHON_BIN}" >&2; exit 1; }

export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -m relacats_v1.data_creation.build_numeric_metamorphic_candidates \
  --datasets "${DATASETS[@]}" \
  --split train \
  --max-questions "${MAX_QUESTIONS}" \
  --preview-per-dataset "${PREVIEW_PER_DATASET}" \
  --output-root "${OUTPUT_ROOT}"

echo
echo "Numeric metamorphic audit complete."
echo "Summary: ${OUTPUT_ROOT}/summary.json"
echo "Preview files:"
for dataset in "${DATASETS[@]}"; do
  echo "  ${OUTPUT_ROOT}/${dataset}/preview.jsonl"
done
