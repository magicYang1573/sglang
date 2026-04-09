#!/bin/bash
# =============================================================================
# CXL vs RDMA vs DRAM — Full experiment matrix (HiSparse + DPA)
# =============================================================================
#
# Matrix:
#   Context lengths: 16k, 32k, 64k, 128k  (target-input-tokens)
#   Pool / backend:  dram, cxl, rdma_local_0, rdma_local_50
#
# Lifecycle (isolated KV / radix state between tests):
#   For each (scheme × context length): start server → wait /health → one
#   e2e_bench_fast run → stop server.  No shared process between cases.
#
# Server (each case): DeepSeek-V3.2 AWQ, TP=8, dp-size=8, DPA, bfloat16,
#   mem-fraction-static 0.7, HiSparse with scheme-specific JSON.
#
# Client: test/cxl_utils/e2e_bench_fast.py (two-round prefix-cache bench).
#   All JSON + per-run logs go under RESULTS_DIR (single folder).
#
# Usage:
#   export MODEL_PATH=/data3/yt/models/DeepSeek-V3.2-AWQ   # server weights
#   bash test/cxl_utils/run_all_experiments.sh
#
# Optional env overrides:
#   MODEL_CLIENT   — HF id for tokenizer/client (default: deepseek-ai/DeepSeek-V3.2)
#   PORT, TP, DP_SIZE, IB_DEVICE, CXL_DEV_PATH, CXL_MAP_BYTES
#   NUM_REQUESTS (default 512), NUM_UNIQUE_PROMPTS, OUTPUT_TOKENS (Round 1 / synthetic)
#   ROUND2_OUTPUT_MIN (default 0), ROUND2_OUTPUT_MAX (default 1024) — Round 2 random max_tokens
#   MAX_CONCURRENCY (default 128), REPEAT_MODE
#   HF_ENDPOINT, RESULTS_DIR, SERVER_STARTUP_WAIT
#   SKIP_SERVER=1  — do not start/kill server (client-only; you must align server
#                    with each case manually — not recommended for full matrix)
#
# =============================================================================

set -euo pipefail

# --- defaults ---
MODEL_PATH="${MODEL_PATH:-/data3/yt/models/DeepSeek-V3.2-AWQ}"
MODEL_CLIENT="${MODEL_CLIENT:-deepseek-ai/DeepSeek-V3.2}"
TP="${TP:-8}"
DP_SIZE="${DP_SIZE:-8}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"

IB_DEVICE="${IB_DEVICE:-mlx5_0}"
CXL_DEV_PATH="${CXL_DEV_PATH:-/dev/dax0.0}"
CXL_MAP_BYTES="${CXL_MAP_BYTES:-137438953472}"

NUM_REQUESTS="${NUM_REQUESTS:-512}"
NUM_UNIQUE_PROMPTS="${NUM_UNIQUE_PROMPTS:-1}"
# Round 1 synthetic prompts + server warmup length (must be > 0)
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
# Round 2 only: uniform random max_tokens per request; 0 is clamped to 1 in client
ROUND2_OUTPUT_MIN="${ROUND2_OUTPUT_MIN:-0}"
ROUND2_OUTPUT_MAX="${ROUND2_OUTPUT_MAX:-1024}"
REQUEST_RATE="${REQUEST_RATE:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
REPEAT_MODE="${REPEAT_MODE:-interleaved}"
PROMPT_MODE="${PROMPT_MODE:-synthetic}"
SEED="${SEED:-42}"
PAUSE_BETWEEN_ROUNDS="${PAUSE_BETWEEN_ROUNDS:-3.0}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-300}"
SKIP_SERVER="${SKIP_SERVER:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_FAST="${E2E_FAST:-${SCRIPT_DIR}/e2e_bench_fast.py}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_matrix_${TIMESTAMP}}"

# Context lengths (tokens): 16k / 32k / 64k / 128k
CONTEXT_LENGTHS=(16384 32768 65536 131072)

# Scheme order for loops and summary
SCHEMES=(dram cxl rdma_local_0 rdma_local_50)

# --- HiSparse JSON (built at runtime so CLI/env overrides apply) ---
HISPARSE_DRAM='{"top_k":2048,"device_buffer_size":4096}'

build_hisparse_cxl() {
  printf '%s' '{"top_k":2048,"device_buffer_size":4096,"cxl":{"enabled":true,"dev_path":"'"${CXL_DEV_PATH}"'","map_bytes":'"${CXL_MAP_BYTES}"'}}'
}

build_hisparse_rdma_0() {
  printf '%s' '{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"'"${IB_DEVICE}"'","local_ratio":0.0}}'
}
build_hisparse_rdma_50() {
  printf '%s' '{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"'"${IB_DEVICE}"'","local_ratio":0.5}}'
}

# --- CLI overrides ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-client) MODEL_CLIENT="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --dp-size) DP_SIZE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; mkdir -p "${RESULTS_DIR}"; shift 2 ;;
    --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
    --num-unique-prompts) NUM_UNIQUE_PROMPTS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
    --round2-output-min) ROUND2_OUTPUT_MIN="$2"; shift 2 ;;
    --round2-output-max) ROUND2_OUTPUT_MAX="$2"; shift 2 ;;
    --ib-device) IB_DEVICE="$2"; shift 2 ;;
    --cxl-dev) CXL_DEV_PATH="$2"; shift 2 ;;
    --cxl-map-bytes) CXL_MAP_BYTES="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,35p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

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
    dram) cfg="${HISPARSE_DRAM}" ;;
    cxl) cfg="$(build_hisparse_cxl)" ;;
    rdma_local_0) cfg="$(build_hisparse_rdma_0)" ;;
    rdma_local_50) cfg="$(build_hisparse_rdma_50)" ;;
    *) echo "Unknown scheme: ${scheme}"; return 1 ;;
  esac

  echo "  Launching server: scheme=${scheme} (log -> ${log_file##*/})"
  # shellcheck disable=SC2086
  SGLANG_ENABLE_JIT_DEEPGEMM=0 python -m sglang.launch_server \
    --model "${MODEL_PATH}" \
    --tp "${TP}" \
    --dp-size "${DP_SIZE}" \
    --enable-dp-attention \
    --trust-remote-code \
    --dtype bfloat16 \
    --enable-hisparse \
    --hisparse-config "${cfg}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --port "${PORT}" \
    >"${log_file}" 2>&1 &
  SERVER_PID=$!
}

run_e2e_fast() {
  local out_json="$1"
  local target_tokens="$2"
  export HF_ENDPOINT
  # shellcheck disable=SC2086
  python "${E2E_FAST}" \
    --server-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL_CLIENT}" \
    --prompt-mode "${PROMPT_MODE}" \
    --target-input-tokens "${target_tokens}" \
    --output-tokens "${OUTPUT_TOKENS}" \
    --num-unique-prompts "${NUM_UNIQUE_PROMPTS}" \
    --num-requests "${NUM_REQUESTS}" \
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
  echo "run_all_experiments.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "CONTEXT_LENGTHS=${CONTEXT_LENGTHS[*]}"
  echo "SCHEMES=${SCHEMES[*]}"
  echo "NUM_REQUESTS=${NUM_REQUESTS} NUM_UNIQUE_PROMPTS=${NUM_UNIQUE_PROMPTS}"
  echo "OUTPUT_TOKENS=${OUTPUT_TOKENS} (Round 1 / synthetic)"
  echo "ROUND2_OUTPUT_MIN=${ROUND2_OUTPUT_MIN} ROUND2_OUTPUT_MAX=${ROUND2_OUTPUT_MAX}"
  echo "MAX_CONCURRENCY=${MAX_CONCURRENCY}"
  echo "server_lifecycle=one_start_stop_per_scheme_x_context (no shared KV between cases)"
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
echo "Context lengths:    ${CONTEXT_LENGTHS[*]}"
echo "Schemes:            ${SCHEMES[*]}"
echo "Client: num-requests=${NUM_REQUESTS} max-concurrency=${MAX_CONCURRENCY}"
echo "Round 2 max_tokens: uniform [${ROUND2_OUTPUT_MIN}, ${ROUND2_OUTPUT_MAX}] (0→1)"
echo "e2e_bench_fast:     ${E2E_FAST}"
echo "SKIP_SERVER:        ${SKIP_SERVER}"
echo "============================================================"

for scheme in "${SCHEMES[@]}"; do
  for tin in "${CONTEXT_LENGTHS[@]}"; do
    tag="$(ctx_tag "${tin}")"
    out_file="${RESULTS_DIR}/${scheme}_ctx${tag}.json"
    srv_log="${RESULTS_DIR}/server_${scheme}_ctx${tag}.log"

    echo ""
    echo "################################################################"
    echo "  CASE: scheme=${scheme}  context=${tin} (${tag})"
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
    run_e2e_fast "${out_file}" "${tin}"

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
python3 << 'PY'
import json, os, glob, sys

results_dir = os.environ.get("RESULTS_DIR", "")
if not results_dir or not os.path.isdir(results_dir):
    sys.exit(0)

paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
print("\nSummary (Round 2 TTFT mean ms, if present):\n")
print(f"{'file':<48} {'R2_TTFT_ms':>12} {'R2_TBT_ms':>12}")
print("-" * 74)
for p in paths:
    if os.path.basename(p) == "run_meta.txt":
        continue
    try:
        with open(p) as f:
            d = json.load(f)
        r2 = d.get("round2", {}).get("metrics", {})
        tt = r2.get("ttft_mean_ms", float("nan"))
        tb = r2.get("tbt_mean_ms", float("nan"))
        print(f"{os.path.basename(p):<48} {tt:>12.1f} {tb:>12.1f}")
    except Exception as e:
        print(f"{os.path.basename(p):<48} {'ERR':>12} {str(e)[:12]:>12}")
PY
