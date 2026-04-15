#!/bin/bash
# =============================================================================
# CXL interleave×2 — context × concurrency matrix
# =============================================================================
#
# Matrix (same lifecycle as run_cxl_rdma_nic_experiments.sh):
#   Context lengths (target-input-tokens): 16k, 32k, 64k, 128k
#   max-concurrency: 8, 16, 32, 48, 64
#
# Scheme: cxl_interleave_2 (HiSparse cxl with dev_paths + map_bytes_per_device)
#
# For each (context × concurrency): start server → /health → e2e_bench_fast
# → stop server (fresh KV / radix per case).
#
# Usage:
#   export MODEL_PATH=/path/to/DeepSeek-V3.2-AWQ
#   bash test/cxl_utils/run_cxl_interleave_concurrency_matrix.sh
#
# Optional env:
#   MODEL_CLIENT, PORT, TP, DP_SIZE, MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS
#   CXL_DEV_PATH, CXL_DEV_PATH_2, CXL_MAP_BYTES_PER_DEVICE
#   NUM_REQUESTS_CXL — if set, used for every concurrency (uniform override).
#     If unset: conc 8 → 128, conc 16 → 256, conc 32/48/64 → 512 (faster low-parallel runs).
#   NUM_UNIQUE_PROMPTS, OUTPUT_TOKENS
#   ROUND2_OUTPUT_MIN, ROUND2_OUTPUT_MAX, REQUEST_RATE, REPEAT_MODE, PROMPT_MODE
#   HF_ENDPOINT, RESULTS_DIR, SERVER_STARTUP_WAIT, SKIP_SERVER
#   SGLANG_ENV_SCRIPT, PYTHON_BIN
#
# =============================================================================

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data3/yt/models/DeepSeek-V3.2-AWQ}"
MODEL_CLIENT="${MODEL_CLIENT:-deepseek-ai/DeepSeek-V3.2}"
TP="${TP:-8}"
DP_SIZE="${DP_SIZE:-8}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-200000}"

CXL_DEV_PATH="${CXL_DEV_PATH:-/dev/dax0.0}"
CXL_DEV_PATH_2="${CXL_DEV_PATH_2:-/dev/dax1.0}"
CXL_MAP_BYTES_PER_DEVICE="${CXL_MAP_BYTES_PER_DEVICE:-274877906944}"

# Empty = tiered by concurrency (see header). Non-empty = same count for all cases.
NUM_REQUESTS_CXL="${NUM_REQUESTS_CXL:-}"
NUM_UNIQUE_PROMPTS="${NUM_UNIQUE_PROMPTS:-1}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
ROUND2_OUTPUT_MIN="${ROUND2_OUTPUT_MIN:-0}"
ROUND2_OUTPUT_MAX="${ROUND2_OUTPUT_MAX:-1024}"
REQUEST_RATE="${REQUEST_RATE:-0}"
REPEAT_MODE="${REPEAT_MODE:-interleaved}"
PROMPT_MODE="${PROMPT_MODE:-synthetic}"
SEED="${SEED:-42}"
PAUSE_BETWEEN_ROUNDS="${PAUSE_BETWEEN_ROUNDS:-3.0}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-600}"
SKIP_SERVER="${SKIP_SERVER:-0}"

SGLANG_ENV_SCRIPT="${SGLANG_ENV_SCRIPT:-${HOME}/env.sh}"
PYTHON_BIN="${PYTHON_BIN:-/data2/ljr/a/downloads/miniconda3/envs/sgl/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_FAST="${E2E_FAST:-${SCRIPT_DIR}/e2e_bench_fast.py}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_cxl_interleave_conc_matrix_${TIMESTAMP}}"

CONTEXT_LENGTHS=(16384 32768 65536 131072)
CONCURRENCIES=(8 16 32 48 64)

SCHEME="cxl_interleave_2"

# num-requests per max-concurrency: low parallelism → fewer requests (shorter runs).
cxl_num_requests_for_conc() {
  local conc="$1"
  if [[ -n "${NUM_REQUESTS_CXL}" ]]; then
    echo "${NUM_REQUESTS_CXL}"
    return
  fi
  case "${conc}" in
    8) echo 128 ;;
    16) echo 256 ;;
    *) echo 512 ;;
  esac
}

build_hisparse_cxl_interleave_2() {
  printf '%s' '{"top_k":2048,"device_buffer_size":4096,"cxl":{"enabled":true,"dev_paths":["'"${CXL_DEV_PATH}"'","'"${CXL_DEV_PATH_2}"'"],"map_bytes_per_device":'"${CXL_MAP_BYTES_PER_DEVICE}"'}}'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-client) MODEL_CLIENT="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --dp-size) DP_SIZE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --mem-fraction-static) MEM_FRACTION_STATIC="$2"; shift 2 ;;
    --max-total-tokens) MAX_TOTAL_TOKENS="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; mkdir -p "${RESULTS_DIR}"; shift 2 ;;
    --num-requests-cxl) NUM_REQUESTS_CXL="$2"; shift 2 ;;
    --num-unique-prompts) NUM_UNIQUE_PROMPTS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --round2-output-min) ROUND2_OUTPUT_MIN="$2"; shift 2 ;;
    --round2-output-max) ROUND2_OUTPUT_MAX="$2"; shift 2 ;;
    --cxl-dev) CXL_DEV_PATH="$2"; shift 2 ;;
    --cxl-dev-2) CXL_DEV_PATH_2="$2"; shift 2 ;;
    --cxl-map-bytes-per-device) CXL_MAP_BYTES_PER_DEVICE="$2"; shift 2 ;;
    --env-script) SGLANG_ENV_SCRIPT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,32p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -f "${SGLANG_ENV_SCRIPT}" ]]; then
  # shellcheck disable=SC1090
  source "${SGLANG_ENV_SCRIPT}"
fi

if ! "${PYTHON_BIN}" -c "import sglang" 2>/dev/null; then
  echo "ERROR: '${PYTHON_BIN}' cannot import sglang." >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"
META_FILE="${RESULTS_DIR}/run_meta.txt"

wait_for_server() {
  local url="http://127.0.0.1:${PORT}/health"
  local max_wait="${SERVER_STARTUP_WAIT}"
  local waited=0
  echo "  Waiting for server at ${url} ..."
  while ! curl -sSf "${url}" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if [[ "${waited}" -ge "${max_wait}" ]]; then
      echo "  ERROR: Server did not become healthy within ${max_wait}s"
      return 1
    fi
  done
  echo "  Server ready (waited ${waited}s)"
}

kill_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    echo "  Stopping server (PID=${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    unset SERVER_PID
    sleep 5
  fi
}

if [[ "${SKIP_SERVER}" != "1" ]]; then
  trap kill_server EXIT
fi

ctx_tag() {
  case "$1" in
    16384) echo "16k" ;;
    32768) echo "32k" ;;
    65536) echo "64k" ;;
    131072) echo "128k" ;;
    *) echo "in$1" ;;
  esac
}

start_server() {
  local log_file="$1"
  local cfg
  cfg="$(build_hisparse_cxl_interleave_2)"
  echo "  Launching server (log -> ${log_file##*/})"
  echo "  HiSparse config: ${cfg}"
  # shellcheck disable=SC2086
  SGLANG_ENABLE_JIT_DEEPGEMM=0 "${PYTHON_BIN}" -m sglang.launch_server \
    --model "${MODEL_PATH}" \
    --tp "${TP}" \
    --dp-size "${DP_SIZE}" \
    --enable-dp-attention \
    --trust-remote-code \
    --dtype bfloat16 \
    --enable-hisparse \
    --allow-auto-truncate \
    --hisparse-config "${cfg}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --port "${PORT}" \
    >"${log_file}" 2>&1 &
  SERVER_PID=$!
}

run_e2e_fast() {
  local out_json="$1"
  local target_tokens="$2"
  local num_requests="$3"
  local max_concurrency="$4"
  export HF_ENDPOINT
  # shellcheck disable=SC2086
  "${PYTHON_BIN}" "${E2E_FAST}" \
    --server-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL_CLIENT}" \
    --prompt-mode "${PROMPT_MODE}" \
    --target-input-tokens "${target_tokens}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --num-unique-prompts "${NUM_UNIQUE_PROMPTS}" \
    --num-requests "${num_requests}" \
    --repeat-mode "${REPEAT_MODE}" \
    --request-rate "${REQUEST_RATE}" \
    --max-concurrency "${max_concurrency}" \
    --seed "${SEED}" \
    --pause-between-rounds "${PAUSE_BETWEEN_ROUNDS}" \
    --round2-output-min "${ROUND2_OUTPUT_MIN}" \
    --round2-output-max "${ROUND2_OUTPUT_MAX}" \
    --output "${out_json}" \
    2>&1 | tee "${out_json%.json}.log"
}

{
  echo "run_cxl_interleave_concurrency_matrix.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "SCHEME=${SCHEME}"
  echo "MODEL_PATH=${MODEL_PATH} MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  echo "CONTEXT_LENGTHS=${CONTEXT_LENGTHS[*]}"
  echo "CONCURRENCIES=${CONCURRENCIES[*]}"
  if [[ -n "${NUM_REQUESTS_CXL}" ]]; then
    echo "NUM_REQUESTS_CXL=${NUM_REQUESTS_CXL} (uniform for all concurrencies)"
  else
    echo "NUM_REQUESTS_CXL=(unset) tiered: conc8=128 conc16=256 conc32/48/64=512"
  fi
  echo "CXL_DEV_PATH=${CXL_DEV_PATH} CXL_DEV_PATH_2=${CXL_DEV_PATH_2}"
  echo "CXL_MAP_BYTES_PER_DEVICE=${CXL_MAP_BYTES_PER_DEVICE}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
} > "${META_FILE}"

TOTAL_CASES=$(( ${#CONTEXT_LENGTHS[@]} * ${#CONCURRENCIES[@]} ))
CURRENT_CASE=0

echo "============================================================"
echo "CXL interleave×2 × context × concurrency → ${RESULTS_DIR}/"
echo "  Cases: ${TOTAL_CASES} (one server per case)"
echo "============================================================"

for tin in "${CONTEXT_LENGTHS[@]}"; do
  tag="$(ctx_tag "${tin}")"
  for conc in "${CONCURRENCIES[@]}"; do
    CURRENT_CASE=$((CURRENT_CASE + 1))
    num_req="$(cxl_num_requests_for_conc "${conc}")"
    base="${SCHEME}_ctx${tag}_conc${conc}"
    out_file="${RESULTS_DIR}/${base}.json"
    srv_log="${RESULTS_DIR}/server_${base}.log"

    echo ""
    echo "################################################################"
    echo "  CASE ${CURRENT_CASE}/${TOTAL_CASES}: ctx=${tin} (${tag})  max-concurrency=${conc}  num-requests=${num_req}"
    echo "################################################################"

    if [[ "${SKIP_SERVER}" == "1" ]]; then
      echo "  SKIP_SERVER=1 — client only"
    else
      kill_server 2>/dev/null || true
      start_server "${srv_log}"
      if ! wait_for_server; then
        echo "  ERROR: server failed to start — skipping"
        kill_server
        continue
      fi
    fi

    echo "  --- client -> ${out_file##*/} ---"
    run_e2e_fast "${out_file}" "${tin}" "${num_req}" "${conc}"

    if [[ "${SKIP_SERVER}" != "1" ]]; then
      kill_server
    fi
  done
done

echo ""
echo "Done. Results under: ${RESULTS_DIR}/"

export RESULTS_DIR
"${PYTHON_BIN}" << 'PY'
import json, os, glob, sys

results_dir = os.environ.get("RESULTS_DIR", "")
if not results_dir or not os.path.isdir(results_dir):
    sys.exit(0)

paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
print("\nSummary (Round 2 TTFT / TBT mean ms):\n")
print(f"{'file':<60} {'R2_TTFT_ms':>12} {'R2_TBT_ms':>12}")
print("-" * 86)
for p in paths:
    bn = os.path.basename(p)
    if bn == "run_meta.txt":
        continue
    try:
        with open(p) as f:
            d = json.load(f)
        r2 = d.get("round2", {}).get("metrics", {})
        tt = r2.get("ttft_mean_ms", float("nan"))
        tb = r2.get("tbt_mean_ms", float("nan"))
        print(f"{bn:<60} {tt:>12.1f} {tb:>12.1f}")
    except Exception as e:
        print(f"{bn:<60} {'ERR':>12} {str(e)[:12]:>12}")
PY
