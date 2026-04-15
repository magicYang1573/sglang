#!/bin/bash
# =============================================================================
# RDMA-only experiment matrix
# =============================================================================
#
# Matrix:
#   Context lengths : 16k, 32k, 64k, 128k  (target-input-tokens)
#   RDMA schemes    :
#     rdma_1nic_local0   — 1 NIC,  local_ratio=0.0  (all remote)
#     rdma_2nic_local0   — 2 NICs, local_ratio=0.0  (all remote)
#     rdma_2nic_local50  — 2 NICs, local_ratio=0.5  (half local)
#     rdma_4nic_local0   — 4 NICs, local_ratio=0.0  (all remote)
#     rdma_8nic_local0   — 8 NICs, local_ratio=0.0  (all remote)
#
# Per-scheme: 128 requests (NUM_REQUESTS, overridable), max-concurrency 64.
#
# Lifecycle: For each (scheme × context length): start server → wait /health →
#   one e2e_bench_fast run → stop server.  No shared KV between cases.
#
# Server: DeepSeek-V3.2 AWQ, TP=8, dp-size=8, DPA, bfloat16,
#   mem-fraction-static 0.85, max-total-tokens 200000, HiSparse + RDMA config.
#
# Client: test/cxl_utils/e2e_bench_fast.py (two-round prefix-cache bench).
#   Single-request timeout is 3600 s (patched in e2e_bench.py).
#
# Usage:
#   export MODEL_PATH=/data3/yt/models/DeepSeek-V3.2-AWQ
#   bash test/cxl_utils/run_rdma_experiments.sh
#
# Optional env overrides (same as run_cxl_rdma_nic_experiments.sh):
#   MODEL_CLIENT, PORT, TP, DP_SIZE, MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS
#   IB_DEVICES (comma-separated, default: mlx5_0,...,mlx5_7)
#   NUM_REQUESTS (default 128), NUM_UNIQUE_PROMPTS, OUTPUT_TOKENS
#   ROUND2_OUTPUT_MIN (default 0), ROUND2_OUTPUT_MAX (default 1024)
#   MAX_CONCURRENCY (default 64), REPEAT_MODE, REQUEST_RATE
#   HF_ENDPOINT, RESULTS_DIR, SERVER_STARTUP_WAIT
#   SKIP_SERVER=1  — do not start/kill server (client-only)
#   SGLANG_ENV_SCRIPT, PYTHON_BIN
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

# RDMA IB devices (8 NICs available; subsets used per scheme)
IB_DEVICES="${IB_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
IFS=',' read -ra ALL_IB_DEVS <<< "${IB_DEVICES}"

NUM_REQUESTS="${NUM_REQUESTS:-128}"
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
RESULTS_DIR="${RESULTS_DIR:-exp/exp_rdma_${TIMESTAMP}}"

# Context lengths (tokens): 16k / 32k / 64k / 128k
CONTEXT_LENGTHS=(16384 32768 65536 131072)

# RDMA schemes: rank_interleave + request_striping
SCHEMES=(
  rdma_1nic_local0
  rdma_2nic_local0
  rdma_2nic_local50
  rdma_4nic_local0
  rdma_8nic_local0
  rdma_stripe_2nic_local0
  rdma_stripe_4nic_local0
  rdma_stripe_8nic_local0
)

# --- HiSparse RDMA JSON builders ---
# rank_interleave (backward-compatible, tp_rank % num_nics)
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

# request_striping (single request → multiple NICs in parallel)
build_hisparse_rdma_striping() {
  local num_nics="$1"
  local local_ratio="${2:-0.0}"
  local min_tokens_per_rnic="${3:-8192}"
  local devs_json=""
  for ((i = 0; i < num_nics; i++)); do
    if [[ $i -gt 0 ]]; then devs_json+=","; fi
    devs_json+='"'"${ALL_IB_DEVS[$i]}"'"'
  done
  printf '%s' '{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"local_ratio":'"${local_ratio}"',"ib_devs":['"${devs_json}"'],"mode":"request_striping","max_rnics_per_request":'"${num_nics}"',"min_tokens_per_rnic":'"${min_tokens_per_rnic}"'}}'
}

# --- CLI overrides ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)            MODEL_PATH="$2";            shift 2 ;;
    --model-client)          MODEL_CLIENT="$2";          shift 2 ;;
    --tp)                    TP="$2";                    shift 2 ;;
    --dp-size)               DP_SIZE="$2";               shift 2 ;;
    --port)                  PORT="$2";                  shift 2 ;;
    --mem-fraction-static)   MEM_FRACTION_STATIC="$2";  shift 2 ;;
    --max-total-tokens)      MAX_TOTAL_TOKENS="$2";      shift 2 ;;
    --results-dir)           RESULTS_DIR="$2"; mkdir -p "${RESULTS_DIR}"; shift 2 ;;
    --num-requests)          NUM_REQUESTS="$2";          shift 2 ;;
    --num-unique-prompts)    NUM_UNIQUE_PROMPTS="$2";    shift 2 ;;
    --output-tokens)         OUTPUT_TOKENS="$2";         shift 2 ;;
    --max-concurrency)       MAX_CONCURRENCY="$2";       shift 2 ;;
    --round2-output-min)     ROUND2_OUTPUT_MIN="$2";     shift 2 ;;
    --round2-output-max)     ROUND2_OUTPUT_MAX="$2";     shift 2 ;;
    --ib-devices)            IB_DEVICES="$2"; IFS=',' read -ra ALL_IB_DEVS <<< "${IB_DEVICES}"; shift 2 ;;
    --env-script)            SGLANG_ENV_SCRIPT="$2";     shift 2 ;;
    --python)                PYTHON_BIN="$2";            shift 2 ;;
    -h|--help)
      sed -n '1,38p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# --- Load conda env ---
if [[ -f "${SGLANG_ENV_SCRIPT}" ]]; then
  echo "Sourcing: ${SGLANG_ENV_SCRIPT}"
  # shellcheck disable=SC1090
  source "${SGLANG_ENV_SCRIPT}"
else
  echo "Note: SGLANG_ENV_SCRIPT not found (${SGLANG_ENV_SCRIPT}); using current PATH." >&2
fi

if ! "${PYTHON_BIN}" -c "import sglang" 2>/dev/null; then
  echo "ERROR: '${PYTHON_BIN}' cannot import sglang." >&2
  exit 1
fi
echo "Python: $(command -v "${PYTHON_BIN}") ($("${PYTHON_BIN}" -c 'import sys; print(sys.executable)'))"

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
    16384)  echo "16k"  ;;
    32768)  echo "32k"  ;;
    65536)  echo "64k"  ;;
    131072) echo "128k" ;;
    *)      echo "in$1" ;;
  esac
}

start_server_for_scheme() {
  local scheme="$1"
  local log_file="$2"
  local cfg=""
  case "${scheme}" in
    rdma_1nic_local0)           cfg="$(build_hisparse_rdma 1 0.0)" ;;
    rdma_2nic_local0)           cfg="$(build_hisparse_rdma 2 0.0)" ;;
    rdma_2nic_local50)          cfg="$(build_hisparse_rdma 2 0.5)" ;;
    rdma_4nic_local0)           cfg="$(build_hisparse_rdma 4 0.0)" ;;
    rdma_8nic_local0)           cfg="$(build_hisparse_rdma 8 0.0)" ;;
    rdma_stripe_2nic_local0)    cfg="$(build_hisparse_rdma_striping 2 0.0)" ;;
    rdma_stripe_4nic_local0)    cfg="$(build_hisparse_rdma_striping 4 0.0)" ;;
    rdma_stripe_8nic_local0)    cfg="$(build_hisparse_rdma_striping 8 0.0)" ;;
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
  echo "run_rdma_experiments.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC} MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "CONTEXT_LENGTHS=${CONTEXT_LENGTHS[*]}"
  echo "SCHEMES=${SCHEMES[*]}"
  echo "IB_DEVICES=${IB_DEVICES}"
  echo "NUM_REQUESTS=${NUM_REQUESTS} NUM_UNIQUE_PROMPTS=${NUM_UNIQUE_PROMPTS}"
  echo "OUTPUT_TOKENS=${OUTPUT_TOKENS} (Round 1 / synthetic)"
  echo "ROUND2_OUTPUT_MIN=${ROUND2_OUTPUT_MIN} ROUND2_OUTPUT_MAX=${ROUND2_OUTPUT_MAX}"
  echo "MAX_CONCURRENCY=${MAX_CONCURRENCY}"
  echo "single_request_timeout=3600s (patched in e2e_bench.py)"
  echo "server_lifecycle=one_start_stop_per_scheme_x_context"
  echo "SGLANG_ENV_SCRIPT=${SGLANG_ENV_SCRIPT}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "python_executable=$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
} > "${META_FILE}"

if [[ "${SKIP_SERVER}" == "1" ]]; then
  echo "WARNING: SKIP_SERVER=1 — no server lifecycle."
fi

echo "============================================================"
echo "RDMA experiment matrix — results -> ${RESULTS_DIR}/"
echo "  (one server start + one client run per case, then server stopped)"
echo "============================================================"
echo "Model (server):     ${MODEL_PATH}"
echo "Model (client):     ${MODEL_CLIENT}"
echo "TP / DP:            ${TP} / ${DP_SIZE}"
echo "mem-fraction / max-total-tokens: ${MEM_FRACTION_STATIC} / ${MAX_TOTAL_TOKENS}"
echo "Context lengths:    ${CONTEXT_LENGTHS[*]}"
echo "Schemes:            ${SCHEMES[*]}"
echo "IB devices:         ${IB_DEVICES}"
echo "num-requests=${NUM_REQUESTS}  max-concurrency=${MAX_CONCURRENCY}"
echo "Round 2 max_tokens: uniform [${ROUND2_OUTPUT_MIN}, ${ROUND2_OUTPUT_MAX}] (0→1)"
echo "single-request timeout: 3600 s"
echo "e2e_bench_fast:     ${E2E_FAST}"
echo "SKIP_SERVER:        ${SKIP_SERVER}"
echo "============================================================"
echo ""
echo "Test groups (scheme x context_length):"
for scheme in "${SCHEMES[@]}"; do
  for tin in "${CONTEXT_LENGTHS[@]}"; do
    echo "  ${scheme}_ctx$(ctx_tag "${tin}")"
  done
done
echo "============================================================"

TOTAL_CASES=$(( ${#SCHEMES[@]} * ${#CONTEXT_LENGTHS[@]} ))
CURRENT_CASE=0

for scheme in "${SCHEMES[@]}"; do
  for tin in "${CONTEXT_LENGTHS[@]}"; do
    tag="$(ctx_tag "${tin}")"
    CURRENT_CASE=$((CURRENT_CASE + 1))
    out_file="${RESULTS_DIR}/${scheme}_ctx${tag}.json"
    srv_log="${RESULTS_DIR}/server_${scheme}_ctx${tag}.log"

    echo ""
    echo "################################################################"
    echo "  CASE ${CURRENT_CASE}/${TOTAL_CASES}: scheme=${scheme}  context=${tin} (${tag})  num-requests=${NUM_REQUESTS}"
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
    run_e2e_fast "${out_file}" "${tin}" "${NUM_REQUESTS}"

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
print("\nSummary (Round 2):\n")
print(f"{'file':<48} {'R2_success':>12} {'R2_TTFT_ms':>12} {'R2_TBT_ms':>12}")
print("-" * 86)
for p in paths:
    if os.path.basename(p) == "run_meta.txt":
        continue
    try:
        with open(p) as f:
            d = json.load(f)
        r2m = d.get("round2", {}).get("metrics", {})
        tt  = r2m.get("ttft_mean_ms",  float("nan"))
        tb  = r2m.get("tbt_mean_ms",   float("nan"))
        ns  = r2m.get("num_success",   -1)
        nr  = r2m.get("num_requests",  -1)
        print(f"{os.path.basename(p):<48} {f'{ns}/{nr}':>12} {tt:>12.1f} {tb:>12.1f}")
    except Exception as e:
        print(f"{os.path.basename(p):<48} {'ERR':>12} {str(e)[:24]:>12}")
PY
