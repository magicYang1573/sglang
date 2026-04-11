#!/bin/bash
# =============================================================================
# CXL (single + interleave×2) vs RDMA (1/2/4/8 NIC) — Full experiment matrix
# =============================================================================
#
# Matrix:
#   Context lengths : 16k, 32k, 64k, 128k  (target-input-tokens)
#   Pool / backend  : cxl, cxl_interleave_2,
#                     rdma_1nic_local0,
#                     rdma_2nic_local0, rdma_2nic_local50,
#                     rdma_4nic_local0, rdma_8nic_local0
#
# Per-scheme num-requests:
#   cxl / cxl_interleave_2 : 512 (default NUM_REQUESTS_CXL)
#   rdma_*                  : 256 (default NUM_REQUESTS_RDMA)
#   max-concurrency         : 64 for all schemes
#
# Lifecycle (isolated KV / radix state between tests):
#   For each (scheme × context length): start server → wait /health → one
#   e2e_bench_fast run → stop server.  No shared process between cases.
#
# Server (each case): DeepSeek-V3.2 AWQ, TP=8, dp-size=8, DPA, bfloat16,
#   mem-fraction-static 0.85, max-total-tokens 200000, HiSparse with scheme-specific JSON.
#
# Client: test/cxl_utils/e2e_bench_fast.py (two-round prefix-cache bench).
#   All JSON + per-run logs go under RESULTS_DIR (single folder).
#
# Usage:
#   export MODEL_PATH=/data3/yt/models/DeepSeek-V3.2-AWQ   # server weights
#   bash test/cxl_utils/run_cxl_rdma_nic_experiments.sh
#
# Python / conda (required so `python -m sglang.launch_server` finds sglang):
#   Default: source ${HOME}/env.sh if present (e.g. PATH to miniconda + activate sgl).
#   Override: SGLANG_ENV_SCRIPT=/path/to/env.sh  or  PYTHON_BIN=/path/to/sgl/bin/python
#
# Optional env overrides:
#   MODEL_CLIENT   — HF id for tokenizer/client (default: deepseek-ai/DeepSeek-V3.2)
#   PORT, TP, DP_SIZE, MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS
#   CXL_DEV_PATH, CXL_DEV_PATH_2, CXL_MAP_BYTES, CXL_MAP_BYTES_PER_DEVICE
#   IB_DEVICES (comma-separated, default: mlx5_0,...,mlx5_7)
#   NUM_REQUESTS_CXL  (default 512) — num-requests for cxl / cxl_interleave_2
#   NUM_REQUESTS_RDMA (default 256) — num-requests for rdma_* schemes
#   NUM_UNIQUE_PROMPTS, OUTPUT_TOKENS (Round 1 / synthetic)
#   ROUND2_OUTPUT_MIN (default 0), ROUND2_OUTPUT_MAX (default 1024) — Round 2 random max_tokens
#   MAX_CONCURRENCY (default 64), REPEAT_MODE
#   HF_ENDPOINT, RESULTS_DIR, SERVER_STARTUP_WAIT
#   SKIP_SERVER=1  — do not start/kill server (client-only)
#
# =============================================================================

set -euo pipefail

# --- defaults ---
MODEL_PATH="${MODEL_PATH:-/data3/yt/models/DeepSeek-V3.2-AWQ}"
MODEL_CLIENT="${MODEL_CLIENT:-deepseek-ai/DeepSeek-V3.2}"
TP="${TP:-8}"
DP_SIZE="${DP_SIZE:-8}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-200000}"

# CXL devices
CXL_DEV_PATH="${CXL_DEV_PATH:-/dev/dax0.0}"
CXL_DEV_PATH_2="${CXL_DEV_PATH_2:-/dev/dax1.0}"
# 274877906944 = 256 GiB; used as map_bytes (single) and map_bytes_per_device (interleave)
CXL_MAP_BYTES="${CXL_MAP_BYTES:-274877906944}"
CXL_MAP_BYTES_PER_DEVICE="${CXL_MAP_BYTES_PER_DEVICE:-274877906944}"

# RDMA IB devices (8 NICs available, subsets used for 1/2/4/8 NIC schemes)
IB_DEVICES="${IB_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
IFS=',' read -ra ALL_IB_DEVS <<< "${IB_DEVICES}"

NUM_REQUESTS_CXL="${NUM_REQUESTS_CXL:-512}"
NUM_REQUESTS_RDMA="${NUM_REQUESTS_RDMA:-256}"
NUM_UNIQUE_PROMPTS="${NUM_UNIQUE_PROMPTS:-1}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
ROUND2_OUTPUT_MIN="${ROUND2_OUTPUT_MIN:-0}"
ROUND2_OUTPUT_MAX="${ROUND2_OUTPUT_MAX:-1024}"
REQUEST_RATE="${REQUEST_RATE:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
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
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_cxl_rdma_nic_${TIMESTAMP}}"

# Context lengths (tokens): 16k / 32k / 64k / 128k
CONTEXT_LENGTHS=(16384 32768 65536 131072)

# Scheme order: CXL single, CXL interleave×2, RDMA 1-NIC, 2-NIC (local0 + local50), 4-NIC, 8-NIC
SCHEMES=(cxl cxl_interleave_2 rdma_1nic_local0 rdma_2nic_local0 rdma_2nic_local50 rdma_4nic_local0 rdma_8nic_local0)

# --- HiSparse JSON builders ---
# CXL single device: dev_path + map_bytes (consistent with run_all_experiments.sh)
build_hisparse_cxl() {
  printf '%s' '{"top_k":2048,"device_buffer_size":4096,"cxl":{"enabled":true,"dev_path":"'"${CXL_DEV_PATH}"'","map_bytes":'"${CXL_MAP_BYTES}"'}}'
}

# CXL interleave×2: dev_paths + map_bytes_per_device
build_hisparse_cxl_interleave_2() {
  printf '%s' '{"top_k":2048,"device_buffer_size":4096,"cxl":{"enabled":true,"dev_paths":["'"${CXL_DEV_PATH}"'","'"${CXL_DEV_PATH_2}"'"],"map_bytes_per_device":'"${CXL_MAP_BYTES_PER_DEVICE}"'}}'
}

build_hisparse_rdma() {
  local num_nics="$1"
  local local_ratio="${2:-0.0}"
  local devs_json=""
  for ((i = 0; i < num_nics; i++)); do
    if [[ $i -gt 0 ]]; then devs_json+=","; fi
    devs_json+='"'"${ALL_IB_DEVS[$i]}"'"'
  done
  printf '%s' '{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"local_ratio":'"${local_ratio}"',"ib_devs":['"${devs_json}"']}}'
}

# --- CLI overrides ---
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
    --num-requests-rdma) NUM_REQUESTS_RDMA="$2"; shift 2 ;;
    --num-unique-prompts) NUM_UNIQUE_PROMPTS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
    --round2-output-min) ROUND2_OUTPUT_MIN="$2"; shift 2 ;;
    --round2-output-max) ROUND2_OUTPUT_MAX="$2"; shift 2 ;;
    --ib-devices) IB_DEVICES="$2"; IFS=',' read -ra ALL_IB_DEVS <<< "${IB_DEVICES}"; shift 2 ;;
    --cxl-dev) CXL_DEV_PATH="$2"; shift 2 ;;
    --cxl-dev-2) CXL_DEV_PATH_2="$2"; shift 2 ;;
    --cxl-map-bytes) CXL_MAP_BYTES="$2"; shift 2 ;;
    --cxl-map-bytes-per-device) CXL_MAP_BYTES_PER_DEVICE="$2"; shift 2 ;;
    --env-script) SGLANG_ENV_SCRIPT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,42p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Load conda env
if [[ -f "${SGLANG_ENV_SCRIPT}" ]]; then
  echo "Sourcing: ${SGLANG_ENV_SCRIPT}"
  # shellcheck disable=SC1090
  source "${SGLANG_ENV_SCRIPT}"
else
  echo "Note: SGLANG_ENV_SCRIPT not found (${SGLANG_ENV_SCRIPT}); using current PATH." >&2
fi

if ! "${PYTHON_BIN}" -c "import sglang" 2>/dev/null; then
  echo "ERROR: '${PYTHON_BIN}' cannot import sglang (resolved: $(command -v "${PYTHON_BIN}" 2>/dev/null || echo 'not in PATH'))." >&2
  echo "  Fix: put Miniconda on PATH + activate sgl in ${SGLANG_ENV_SCRIPT}, or set PYTHON_BIN to that env's python." >&2
  exit 1
fi
echo "Python for server + client: $(command -v "${PYTHON_BIN}") ($("${PYTHON_BIN}" -c 'import sys; print(sys.executable)'))"

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

start_server_for_scheme() {
  local scheme="$1"
  local log_file="$2"
  local cfg=""
  case "${scheme}" in
    cxl)                cfg="$(build_hisparse_cxl)" ;;
    cxl_interleave_2)   cfg="$(build_hisparse_cxl_interleave_2)" ;;
    rdma_1nic_local0)   cfg="$(build_hisparse_rdma 1 0.0)" ;;
    rdma_2nic_local0)   cfg="$(build_hisparse_rdma 2 0.0)" ;;
    rdma_2nic_local50)  cfg="$(build_hisparse_rdma 2 0.5)" ;;
    rdma_4nic_local0)   cfg="$(build_hisparse_rdma 4 0.0)" ;;
    rdma_8nic_local0)   cfg="$(build_hisparse_rdma 8 0.0)" ;;
    *) echo "Unknown scheme: ${scheme}"; return 1 ;;
  esac

  echo "  Launching server: scheme=${scheme} (log -> ${log_file##*/})"
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
    --max-concurrency "${MAX_CONCURRENCY}" \
    --seed "${SEED}" \
    --pause-between-rounds "${PAUSE_BETWEEN_ROUNDS}" \
    --round2-output-min "${ROUND2_OUTPUT_MIN}" \
    --round2-output-max "${ROUND2_OUTPUT_MAX}" \
    --output "${out_json}" \
    2>&1 | tee "${out_json%.json}.log"
}

{
  echo "run_cxl_rdma_nic_experiments.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC} MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "CONTEXT_LENGTHS=${CONTEXT_LENGTHS[*]}"
  echo "SCHEMES=${SCHEMES[*]}"
  echo "CXL_DEV_PATH=${CXL_DEV_PATH} CXL_DEV_PATH_2=${CXL_DEV_PATH_2}"
  echo "CXL_MAP_BYTES=${CXL_MAP_BYTES} CXL_MAP_BYTES_PER_DEVICE=${CXL_MAP_BYTES_PER_DEVICE}"
  echo "IB_DEVICES=${IB_DEVICES}"
  echo "NUM_REQUESTS_CXL=${NUM_REQUESTS_CXL} NUM_REQUESTS_RDMA=${NUM_REQUESTS_RDMA} NUM_UNIQUE_PROMPTS=${NUM_UNIQUE_PROMPTS}"
  echo "OUTPUT_TOKENS=${OUTPUT_TOKENS} (Round 1 / synthetic)"
  echo "ROUND2_OUTPUT_MIN=${ROUND2_OUTPUT_MIN} ROUND2_OUTPUT_MAX=${ROUND2_OUTPUT_MAX}"
  echo "MAX_CONCURRENCY=${MAX_CONCURRENCY}"
  echo "server_lifecycle=one_start_stop_per_scheme_x_context (no shared KV between cases)"
  echo "SGLANG_ENV_SCRIPT=${SGLANG_ENV_SCRIPT}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "python_executable=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
} > "${META_FILE}"

if [[ "${SKIP_SERVER}" == "1" ]]; then
  echo "WARNING: SKIP_SERVER=1 — no server lifecycle; you must match server config"
  echo "         to each (scheme, context) yourself. Normal runs use one fresh server per case."
fi

echo "============================================================"
echo "Experiment matrix — results -> ${RESULTS_DIR}/"
echo "  (one server start + one client run per case, then server stopped)"
echo "============================================================"
echo "Model (server):     ${MODEL_PATH}"
echo "Model (client):     ${MODEL_CLIENT}"
echo "TP / DP:            ${TP} / ${DP_SIZE}"
echo "mem-fraction / max-total-tokens: ${MEM_FRACTION_STATIC} / ${MAX_TOTAL_TOKENS}"
echo "Context lengths:    ${CONTEXT_LENGTHS[*]}"
echo "Schemes:            ${SCHEMES[*]}"
echo "CXL devices:        ${CXL_DEV_PATH}, ${CXL_DEV_PATH_2}"
echo "IB devices:         ${IB_DEVICES}"
echo "Client: num-requests cxl=${NUM_REQUESTS_CXL} rdma=${NUM_REQUESTS_RDMA} max-concurrency=${MAX_CONCURRENCY}"
echo "Round 2 max_tokens: uniform [${ROUND2_OUTPUT_MIN}, ${ROUND2_OUTPUT_MAX}] (0→1)"
echo "e2e_bench_fast:     ${E2E_FAST}"
echo "SKIP_SERVER:        ${SKIP_SERVER}"
echo "============================================================"

TOTAL_CASES=$(( ${#SCHEMES[@]} * ${#CONTEXT_LENGTHS[@]} ))
CURRENT_CASE=0

for scheme in "${SCHEMES[@]}"; do
  # Per-scheme num-requests: 512 for CXL, 256 for RDMA
  case "${scheme}" in
    cxl*)  scheme_num_requests="${NUM_REQUESTS_CXL}" ;;
    rdma*) scheme_num_requests="${NUM_REQUESTS_RDMA}" ;;
    *)     scheme_num_requests="${NUM_REQUESTS_CXL}" ;;
  esac

  for tin in "${CONTEXT_LENGTHS[@]}"; do
    tag="$(ctx_tag "${tin}")"
    CURRENT_CASE=$((CURRENT_CASE + 1))
    out_file="${RESULTS_DIR}/${scheme}_ctx${tag}.json"
    srv_log="${RESULTS_DIR}/server_${scheme}_ctx${tag}.log"

    echo ""
    echo "################################################################"
    echo "  CASE ${CURRENT_CASE}/${TOTAL_CASES}: scheme=${scheme}  context=${tin} (${tag})  num-requests=${scheme_num_requests}"
    echo "################################################################"

    if [[ "${SKIP_SERVER}" == "1" ]]; then
      echo "  SKIP_SERVER=1 — run client only"
    else
      kill_server 2>/dev/null || true
      start_server_for_scheme "${scheme}" "${srv_log}"
      if ! wait_for_server; then
        echo "  ERROR: server failed to start — skipping this case"
        kill_server
        continue
      fi
    fi

    echo ""
    echo "  --- client -> ${out_file##*/} ---"
    run_e2e_fast "${out_file}" "${tin}" "${scheme_num_requests}"

    if [[ "${SKIP_SERVER}" != "1" ]]; then
      kill_server
    fi
  done
done

echo ""
echo "============================================================"
echo "Done. All JSON + *.log under: ${RESULTS_DIR}/"
echo "============================================================"

export RESULTS_DIR
"${PYTHON_BIN}" << 'PY'
import json, os, glob, sys

results_dir = os.environ.get("RESULTS_DIR", "")
if not results_dir or not os.path.isdir(results_dir):
    sys.exit(0)

paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
print("\nSummary (Round 2 TTFT mean ms, if present):\n")
print(f"{'file':<52} {'R2_TTFT_ms':>12} {'R2_TBT_ms':>12}")
print("-" * 78)
for p in paths:
    if os.path.basename(p) == "run_meta.txt":
        continue
    try:
        with open(p) as f:
            d = json.load(f)
        r2 = d.get("round2", {}).get("metrics", {})
        tt = r2.get("ttft_mean_ms", float("nan"))
        tb = r2.get("tbt_mean_ms", float("nan"))
        print(f"{os.path.basename(p):<52} {tt:>12.1f} {tb:>12.1f}")
    except Exception as e:
        print(f"{os.path.basename(p):<52} {'ERR':>12} {str(e)[:12]:>12}")
PY
