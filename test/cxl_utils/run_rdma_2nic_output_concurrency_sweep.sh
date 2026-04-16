#!/bin/bash
# =============================================================================
# RDMA 2-NIC — output length × concurrency sweep (extends local0 matrix script)
# =============================================================================
#
# Three phases (each case: fresh server → /health → e2e_bench_fast → stop):
#
#   Phase A — local_ratio sweep @ concurrency=64, Round-2 max_tokens ~ U[0,1024]
#     Context: 16k, 32k, 64k, 128k
#     RDMA NIC×2: local_ratio=0.0 ("local 0") and 0.5 ("local 50")
#
#   Phase B — RDMA NIC×2 @ local_ratio=0.0, concurrency=64, fixed Round-2 length
#     Context: 128k and 32k only
#     Round-2 output (fixed): 1024, 2048, 4096, 8192 tokens (1k / 2k / 4k / 8k)
#
#   Phase C — RDMA NIC×2 @ local_ratio=0.0, Round-2 max_tokens ~ U[0,1024]
#     Context: 16k, 32k, 64k, 128k
#     max-concurrency: 80, 96
#
# Scheme JSON matches run_cxl_rdma_nic_experiments.sh: rdma_pool with first two
# entries of IB_DEVICES, rank_interleave mode (default).
#
# Usage:
#   export MODEL_PATH=/path/to/DeepSeek-V3.2-AWQ
#   bash test/cxl_utils/run_rdma_2nic_output_concurrency_sweep.sh
#
# Optional env (same style as run_rdma_2nic_local0_concurrency_matrix.sh):
#   MODEL_CLIENT, PORT, TP, DP_SIZE, MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS
#   IB_DEVICES, NUM_REQUESTS_RDMA, NUM_UNIQUE_PROMPTS, OUTPUT_TOKENS
#   REQUEST_RATE, REPEAT_MODE, PROMPT_MODE, SEED, PAUSE_BETWEEN_ROUNDS
#   HF_ENDPOINT, RESULTS_DIR, SERVER_STARTUP_WAIT, SKIP_SERVER
#   SGLANG_ENV_SCRIPT, PYTHON_BIN
#
# Phase B fixed decode sets OUTPUT_TOKENS = round2 length for synthetic Round 1.
# Phases A/C use ORIG_OUTPUT_TOKENS (after CLI / env defaults).
#
# Resume: existing per-case JSON under RESULTS_DIR is skipped unless FORCE_RERUN=1.
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

IB_DEVICES="${IB_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
IFS=',' read -ra ALL_IB_DEVS <<< "${IB_DEVICES}"

NUM_REQUESTS_RDMA="${NUM_REQUESTS_RDMA:-256}"
NUM_UNIQUE_PROMPTS="${NUM_UNIQUE_PROMPTS:-1}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
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
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_rdma_2nic_output_conc_sweep_${TIMESTAMP}}"

# Mutable per-case (defaults = Phase A / C random Round-2)
ROUND2_OUTPUT_MIN="${ROUND2_OUTPUT_MIN:-0}"
ROUND2_OUTPUT_MAX="${ROUND2_OUTPUT_MAX:-1024}"

build_hisparse_rdma_2nic() {
  local local_ratio="$1"
  local devs_json=""
  for ((i = 0; i < 2; i++)); do
    if [[ $i -gt 0 ]]; then devs_json+=","; fi
    devs_json+='"'"${ALL_IB_DEVS[$i]}"'"'
  done
  printf '%s' '{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"local_ratio":'"${local_ratio}"',"ib_devs":['"${devs_json}"']}}'
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
    --num-requests-rdma) NUM_REQUESTS_RDMA="$2"; shift 2 ;;
    --num-unique-prompts) NUM_UNIQUE_PROMPTS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --ib-devices) IB_DEVICES="$2"; IFS=',' read -ra ALL_IB_DEVS <<< "${IB_DEVICES}"; shift 2 ;;
    --env-script) SGLANG_ENV_SCRIPT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,40p' "$0"
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

# Round-1 synthetic decode length (e2e_bench_fast requires >0); restored after Phase B.
ORIG_OUTPUT_TOKENS="${OUTPUT_TOKENS}"

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

# Human tag for fixed decode filenames (1024 → o1k)
out_fix_tag() {
  case "$1" in
    1024) echo "1k" ;;
    2048) echo "2k" ;;
    4096) echo "4k" ;;
    8192) echo "8k" ;;
    *) echo "o$1" ;;
  esac
}

start_server_rdma_2nic() {
  local log_file="$1"
  local local_ratio="$2"
  local cfg
  cfg="$(build_hisparse_rdma_2nic "${local_ratio}")"
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

TOTAL_CASES=$(( 8 + 8 + 8 ))
CURRENT_CASE=0

{
  echo "run_rdma_2nic_output_concurrency_sweep.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "MODEL_PATH=${MODEL_PATH} MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  echo "NUM_REQUESTS_RDMA=${NUM_REQUESTS_RDMA}"
  echo "IB_DEVICES=${IB_DEVICES}"
  echo "TOTAL_CASES=${TOTAL_CASES} (phaseA=8 phaseB=8 phaseC=8)"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
} > "${META_FILE}"

echo "============================================================"
echo "RDMA 2-NIC output × concurrency sweep → ${RESULTS_DIR}/"
echo "  Total cases: ${TOTAL_CASES}"
echo "============================================================"

# --- Phase A: ctx × (local0, local50) @ conc=64, R2 random [0,1024] ---
PHASE_A_CTX=(16384 32768 65536 131072)
PHASE_A_LR=(0.0 0.5)
for lr in "${PHASE_A_LR[@]}"; do
  lr_tag="l0"
  [[ "${lr}" == "0.5" ]] && lr_tag="l50"
  for tin in "${PHASE_A_CTX[@]}"; do
    CURRENT_CASE=$((CURRENT_CASE + 1))
    tag="$(ctx_tag "${tin}")"
    base="pA_rdma2nic_${lr_tag}_ctx${tag}_conc64_r2rand0_1024"
    out_file="${RESULTS_DIR}/${base}.json"
    srv_log="${RESULTS_DIR}/server_${base}.log"
    OUTPUT_TOKENS="${ORIG_OUTPUT_TOKENS}"
    ROUND2_OUTPUT_MIN=0
    ROUND2_OUTPUT_MAX=1024

    if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
      echo ""
      echo "################################################################"
      echo "  SKIP ${CURRENT_CASE}/${TOTAL_CASES}: existing ${out_file##*/}"
      echo "################################################################"
      continue
    fi

    echo ""
    echo "################################################################"
    echo "  PHASE A ${CURRENT_CASE}/${TOTAL_CASES}: local_ratio=${lr} (${lr_tag}) ctx=${tin} (${tag}) conc=64 R2∈[0,1024] num-req=${NUM_REQUESTS_RDMA}"
    echo "################################################################"

    if [[ "${SKIP_SERVER}" == "1" ]]; then
      echo "  SKIP_SERVER=1 — client only"
    else
      kill_server 2>/dev/null || true
      start_server_rdma_2nic "${srv_log}" "${lr}"
      if ! wait_for_server; then
        echo "  ERROR: server failed to start — skipping"
        kill_server
        continue
      fi
    fi

    echo "  --- client -> ${out_file##*/} ---"
    run_e2e_fast "${out_file}" "${tin}" "${NUM_REQUESTS_RDMA}" 64

    if [[ "${SKIP_SERVER}" != "1" ]]; then
      kill_server
    fi
  done
done

# --- Phase B: 128k + 32k × fixed R2 @ conc=64, local_ratio=0.0 ---
PHASE_B_CTX=(131072 32768)
PHASE_B_OUT=(1024 2048 4096 8192)
for tin in "${PHASE_B_CTX[@]}"; do
  tag="$(ctx_tag "${tin}")"
  for olen in "${PHASE_B_OUT[@]}"; do
    CURRENT_CASE=$((CURRENT_CASE + 1))
    ot="$(out_fix_tag "${olen}")"
    base="pB_rdma2nic_l0_ctx${tag}_conc64_r2fix${ot}"
    out_file="${RESULTS_DIR}/${base}.json"
    srv_log="${RESULTS_DIR}/server_${base}.log"
    OUTPUT_TOKENS="${olen}"
    ROUND2_OUTPUT_MIN="${olen}"
    ROUND2_OUTPUT_MAX="${olen}"

    if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
      echo ""
      echo "################################################################"
      echo "  SKIP ${CURRENT_CASE}/${TOTAL_CASES}: existing ${out_file##*/}"
      echo "################################################################"
      continue
    fi

    echo ""
    echo "################################################################"
    echo "  PHASE B ${CURRENT_CASE}/${TOTAL_CASES}: local_ratio=0.0 ctx=${tin} (${tag}) conc=64 R2=${olen} (fixed) num-req=${NUM_REQUESTS_RDMA}"
    echo "################################################################"

    if [[ "${SKIP_SERVER}" == "1" ]]; then
      echo "  SKIP_SERVER=1 — client only"
    else
      kill_server 2>/dev/null || true
      start_server_rdma_2nic "${srv_log}" "0.0"
      if ! wait_for_server; then
        echo "  ERROR: server failed to start — skipping"
        kill_server
        continue
      fi
    fi

    echo "  --- client -> ${out_file##*/} ---"
    run_e2e_fast "${out_file}" "${tin}" "${NUM_REQUESTS_RDMA}" 64

    if [[ "${SKIP_SERVER}" != "1" ]]; then
      kill_server
    fi
  done
done

# Restore Round-1 output length for Phase C
OUTPUT_TOKENS="${ORIG_OUTPUT_TOKENS}"
ROUND2_OUTPUT_MIN=0
ROUND2_OUTPUT_MAX=1024

# --- Phase C: ctx × conc (80,96) @ local0, R2 random [0,1024] ---
PHASE_C_CTX=(16384 32768 65536 131072)
PHASE_C_CONC=(80 96)
for conc in "${PHASE_C_CONC[@]}"; do
  for tin in "${PHASE_C_CTX[@]}"; do
    CURRENT_CASE=$((CURRENT_CASE + 1))
    tag="$(ctx_tag "${tin}")"
    base="pC_rdma2nic_l0_ctx${tag}_conc${conc}_r2rand0_1024"
    out_file="${RESULTS_DIR}/${base}.json"
    srv_log="${RESULTS_DIR}/server_${base}.log"

    if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
      echo ""
      echo "################################################################"
      echo "  SKIP ${CURRENT_CASE}/${TOTAL_CASES}: existing ${out_file##*/}"
      echo "################################################################"
      continue
    fi

    echo ""
    echo "################################################################"
    echo "  PHASE C ${CURRENT_CASE}/${TOTAL_CASES}: local_ratio=0.0 ctx=${tin} (${tag}) conc=${conc} R2∈[0,1024] num-req=${NUM_REQUESTS_RDMA}"
    echo "################################################################"

    if [[ "${SKIP_SERVER}" == "1" ]]; then
      echo "  SKIP_SERVER=1 — client only"
    else
      kill_server 2>/dev/null || true
      start_server_rdma_2nic "${srv_log}" "0.0"
      if ! wait_for_server; then
        echo "  ERROR: server failed to start — skipping"
        kill_server
        continue
      fi
    fi

    echo "  --- client -> ${out_file##*/} ---"
    run_e2e_fast "${out_file}" "${tin}" "${NUM_REQUESTS_RDMA}" "${conc}"

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
