#!/bin/bash
# =============================================================================
# RDMA 2-NIC (first two IB_DEVICES) — three phased sweeps
# =============================================================================
#
# Phase 1 — local_ratio=0.0 ("NIC-2 local 0"):
#   Context: 16k, 32k, 64k, 128k
#   Round-2 output: fixed 1k (1024 tokens)
#   max-concurrency: 64
#
# Phase 2 — RDMA NIC×2 @ local_ratio=0.0:
#   Context: 16k, 32k, 64k, 128k
#   Round-2 output: fixed 1k
#   max-concurrency: 8, 16, 32, 48, 64, 80, 96
#   num-requests: max_concurrency × PH2_REQUESTS_MULT (default 8)
#
# Phase 3 — RDMA NIC×2 @ local_ratio=0.0:
#   Context: 16k, 32k, 64k, 128k
#   Round-2 output: fixed 2k / 4k / 8k (2048 / 4096 / 8192)
#   max-concurrency: 64
#   num-requests: 256 (2k), 256 (4k), 128 (8k) — overridable via env below
#
# Per case: start server → /health → e2e_bench_fast → stop server.
# Result JSON + logs: filenames start with p{phase}_{seq}_ for sorting.
#
# Usage:
#   export MODEL_PATH=/path/to/DeepSeek-V3.2-AWQ
#   bash test/cxl_utils/run_rdma_2nic_phased_req512.sh
#
# Optional env (same family as run_rdma_2nic_output_concurrency_sweep.sh):
#   MODEL_CLIENT, PORT, TP, DP_SIZE, MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS
#   IB_DEVICES, NUM_REQUESTS_RDMA (default 512 for Phase 1 only), NUM_UNIQUE_PROMPTS
#   PH2_REQUESTS_MULT (default 8) — Phase 2 num-requests = max_concurrency × this
#   NUM_REQUESTS_RDMA_P3_2K (default 256), NUM_REQUESTS_RDMA_P3_4K (default 256),
#   NUM_REQUESTS_RDMA_P3_8K (default 128) — Phase 3 only
#   REQUEST_RATE, REPEAT_MODE, PROMPT_MODE, SEED, PAUSE_BETWEEN_ROUNDS
#   HF_ENDPOINT, RESULTS_DIR, SERVER_STARTUP_WAIT, SKIP_SERVER
#   SGLANG_ENV_SCRIPT, PYTHON_BIN
#
# Resume: existing per-case JSON is skipped unless FORCE_RERUN=1.
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

NUM_REQUESTS_RDMA="${NUM_REQUESTS_RDMA:-512}"
PH2_REQUESTS_MULT="${PH2_REQUESTS_MULT:-8}"
NUM_REQUESTS_RDMA_P3_2K="${NUM_REQUESTS_RDMA_P3_2K:-256}"
NUM_REQUESTS_RDMA_P3_4K="${NUM_REQUESTS_RDMA_P3_4K:-256}"
NUM_REQUESTS_RDMA_P3_8K="${NUM_REQUESTS_RDMA_P3_8K:-128}"
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
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_rdma_2nic_phased_req512_${TIMESTAMP}}"

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
      sed -n '1,42p' "$0"
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

ORIG_OUTPUT_TOKENS="${OUTPUT_TOKENS}"

mkdir -p "${RESULTS_DIR}"
META_FILE="${RESULTS_DIR}/run_meta.txt"

CTX_ALL=(16384 32768 65536 131072)
PH2_CONC=(8 16 32 48 64 80 96)
PH3_OUT=(2048 4096 8192)
TOTAL_CASES=$(( ${#CTX_ALL[@]} + ${#CTX_ALL[@]} * ${#PH2_CONC[@]} + ${#CTX_ALL[@]} * ${#PH3_OUT[@]} ))

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

out_fix_tag() {
  case "$1" in
    1024) echo "1k" ;;
    2048) echo "2k" ;;
    4096) echo "4k" ;;
    8192) echo "8k" ;;
    *) echo "o$1" ;;
  esac
}

# Phase-3 num-requests by fixed Round-2 length (tokens).
p3_num_requests_for_olen() {
  case "$1" in
    2048) printf '%s' "${NUM_REQUESTS_RDMA_P3_2K}" ;;
    4096) printf '%s' "${NUM_REQUESTS_RDMA_P3_4K}" ;;
    8192) printf '%s' "${NUM_REQUESTS_RDMA_P3_8K}" ;;
    *) printf '%s' "${NUM_REQUESTS_RDMA}" ;;
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

{
  echo "run_rdma_2nic_phased_req512.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "MODEL_PATH=${MODEL_PATH} MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  echo "NUM_REQUESTS_RDMA=${NUM_REQUESTS_RDMA} PH2_REQUESTS_MULT=${PH2_REQUESTS_MULT}"
  echo "NUM_REQUESTS_RDMA_P3_2K=${NUM_REQUESTS_RDMA_P3_2K} NUM_REQUESTS_RDMA_P3_4K=${NUM_REQUESTS_RDMA_P3_4K} NUM_REQUESTS_RDMA_P3_8K=${NUM_REQUESTS_RDMA_P3_8K}"
  echo "IB_DEVICES=${IB_DEVICES}"
  echo "TOTAL_CASES=${TOTAL_CASES} (p1=${#CTX_ALL[@]} p2=$((${#CTX_ALL[@]} * ${#PH2_CONC[@]})) p3=$((${#CTX_ALL[@]} * ${#PH3_OUT[@]})))"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
} > "${META_FILE}"

GLOBAL_IDX=0

echo "============================================================"
echo "RDMA 2-NIC phased bench (P1 req=${NUM_REQUESTS_RDMA}; P2 req=conc×${PH2_REQUESTS_MULT}; P3 2k/4k=${NUM_REQUESTS_RDMA_P3_2K}/${NUM_REQUESTS_RDMA_P3_4K}, 8k=${NUM_REQUESTS_RDMA_P3_8K}) → ${RESULTS_DIR}/"
echo "  Total cases: ${TOTAL_CASES}"
echo "============================================================"

# --- Phase 1: ctx × conc=64, R2 fixed 1k, local_ratio=0.0 ---
P1_SEQ=0
OUTPUT_TOKENS=1024
ROUND2_OUTPUT_MIN=1024
ROUND2_OUTPUT_MAX=1024
for tin in "${CTX_ALL[@]}"; do
  P1_SEQ=$((P1_SEQ + 1))
  GLOBAL_IDX=$((GLOBAL_IDX + 1))
  tag="$(ctx_tag "${tin}")"
  seq="$(printf '%02d' "${P1_SEQ}")"
  base="p1_${seq}_rdma2nic_l0_ctx${tag}_conc64_r2fix1k_n${NUM_REQUESTS_RDMA}"
  out_file="${RESULTS_DIR}/${base}.json"
  srv_log="${RESULTS_DIR}/server_${base}.log"

  if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
    echo ""
    echo "################################################################"
    echo "  SKIP ${GLOBAL_IDX}/${TOTAL_CASES} Phase1: existing ${out_file##*/}"
    echo "################################################################"
    continue
  fi

  echo ""
  echo "################################################################"
  echo "  Phase1 ${GLOBAL_IDX}/${TOTAL_CASES}: local_ratio=0.0 ctx=${tin} (${tag}) conc=64 R2=1024 num-req=${NUM_REQUESTS_RDMA}"
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

# --- Phase 2: ctx × conc, R2 fixed 1k, local_ratio=0.0 ---
P2_SEQ=0
OUTPUT_TOKENS=1024
ROUND2_OUTPUT_MIN=1024
ROUND2_OUTPUT_MAX=1024
for tin in "${CTX_ALL[@]}"; do
  tag="$(ctx_tag "${tin}")"
  for conc in "${PH2_CONC[@]}"; do
    P2_SEQ=$((P2_SEQ + 1))
    GLOBAL_IDX=$((GLOBAL_IDX + 1))
    nreq_p2=$(( conc * PH2_REQUESTS_MULT ))
    seq="$(printf '%02d' "${P2_SEQ}")"
    base="p2_${seq}_rdma2nic_l0_ctx${tag}_conc${conc}_r2fix1k_n${nreq_p2}"
    out_file="${RESULTS_DIR}/${base}.json"
    srv_log="${RESULTS_DIR}/server_${base}.log"

    if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
      echo ""
      echo "################################################################"
      echo "  SKIP ${GLOBAL_IDX}/${TOTAL_CASES} Phase2: existing ${out_file##*/}"
      echo "################################################################"
      continue
    fi

    echo ""
    echo "################################################################"
    echo "  Phase2 ${GLOBAL_IDX}/${TOTAL_CASES}: local_ratio=0.0 ctx=${tin} (${tag}) conc=${conc} R2=1024 num-req=${nreq_p2} (=conc×${PH2_REQUESTS_MULT})"
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
    run_e2e_fast "${out_file}" "${tin}" "${nreq_p2}" "${conc}"

    if [[ "${SKIP_SERVER}" != "1" ]]; then
      kill_server
    fi
  done
done

# --- Phase 3: ctx × R2 fixed 2k/4k/8k, conc=64, local_ratio=0.0 ---
P3_SEQ=0
for tin in "${CTX_ALL[@]}"; do
  tag="$(ctx_tag "${tin}")"
  for olen in "${PH3_OUT[@]}"; do
    P3_SEQ=$((P3_SEQ + 1))
    GLOBAL_IDX=$((GLOBAL_IDX + 1))
    nreq_p3="$(p3_num_requests_for_olen "${olen}")"
    ot="$(out_fix_tag "${olen}")"
    seq="$(printf '%02d' "${P3_SEQ}")"
    base="p3_${seq}_rdma2nic_l0_ctx${tag}_conc64_r2fix${ot}_n${nreq_p3}"
    out_file="${RESULTS_DIR}/${base}.json"
    srv_log="${RESULTS_DIR}/server_${base}.log"
    OUTPUT_TOKENS="${olen}"
    ROUND2_OUTPUT_MIN="${olen}"
    ROUND2_OUTPUT_MAX="${olen}"

    if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
      echo ""
      echo "################################################################"
      echo "  SKIP ${GLOBAL_IDX}/${TOTAL_CASES} Phase3: existing ${out_file##*/}"
      echo "################################################################"
      continue
    fi

    echo ""
    echo "################################################################"
    echo "  Phase3 ${GLOBAL_IDX}/${TOTAL_CASES}: local_ratio=0.0 ctx=${tin} (${tag}) conc=64 R2=${olen} (fixed) num-req=${nreq_p3}"
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
    run_e2e_fast "${out_file}" "${tin}" "${nreq_p3}" 64

    if [[ "${SKIP_SERVER}" != "1" ]]; then
      kill_server
    fi
  done
done

OUTPUT_TOKENS="${ORIG_OUTPUT_TOKENS}"

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
print(f"{'file':<72} {'R2_TTFT_ms':>12} {'R2_TBT_ms':>12}")
print("-" * 98)
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
        print(f"{bn:<72} {tt:>12.1f} {tb:>12.1f}")
    except Exception as e:
        print(f"{bn:<72} {'ERR':>12} {str(e)[:12]:>12}")
PY
