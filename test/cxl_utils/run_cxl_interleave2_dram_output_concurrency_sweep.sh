#!/bin/bash
# =============================================================================
# CXL interleave×2 vs DRAM — output × concurrency sweep
# =============================================================================
#
# Two phases (each case: fresh server → /health → e2e_bench_fast → stop):
#
#   Phase 1 — Round-2 max_tokens ~ U[0,1024], max-concurrency 80 and 96
#     Context: 16k, 32k, 64k, 128k
#     Order: CXL interleave×2 first, then DRAM (HiSparse host DRAM pool; no CXL/RDMA)
#
#   Phase 2 — Fixed Round-2 decode length (1k / 2k / 4k / 8k tokens)
#     Context: 128k and 32k only
#     Order: DRAM first, then CXL interleave×2
#     max-concurrency: MAX_CONCURRENCY_FIXED (default 64; set env to override)
#
# CXL JSON matches run_cxl_interleave_concurrency_matrix.sh.
# DRAM JSON matches run_all_experiments.sh (HISPARSE_DRAM).
#
# num-requests: if NUM_REQUESTS_CXL is set, used for all cases; else tiered by
#   max-concurrency (same helper as run_cxl_interleave_concurrency_matrix.sh).
#
# Usage:
#   export MODEL_PATH=/path/to/DeepSeek-V3.2-AWQ
#   bash test/cxl_utils/run_cxl_interleave2_dram_output_concurrency_sweep.sh
#
# Optional env:
#   MODEL_CLIENT, PORT, TP, DP_SIZE, MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS
#   CXL_DEV_PATH, CXL_DEV_PATH_2, CXL_MAP_BYTES_PER_DEVICE
#   NUM_REQUESTS_CXL, NUM_UNIQUE_PROMPTS, OUTPUT_TOKENS (Round 1 / synthetic)
#   OUTPUT_TOKENS_PHASE1 — if set, overrides OUTPUT_TOKENS for Phase 1 only
#   MAX_CONCURRENCY_FIXED (default 64) — Phase 2 only
#   REQUEST_RATE, REPEAT_MODE, PROMPT_MODE, SEED, PAUSE_BETWEEN_ROUNDS
#   HF_ENDPOINT, RESULTS_DIR, SERVER_STARTUP_WAIT, SKIP_SERVER
#   SGLANG_ENV_SCRIPT, PYTHON_BIN
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

CXL_DEV_PATH="${CXL_DEV_PATH:-/dev/dax0.0}"
CXL_DEV_PATH_2="${CXL_DEV_PATH_2:-/dev/dax1.0}"
CXL_MAP_BYTES_PER_DEVICE="${CXL_MAP_BYTES_PER_DEVICE:-274877906944}"

NUM_REQUESTS_CXL="${NUM_REQUESTS_CXL:-}"
NUM_UNIQUE_PROMPTS="${NUM_UNIQUE_PROMPTS:-1}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-512}"
REQUEST_RATE="${REQUEST_RATE:-0}"
REPEAT_MODE="${REPEAT_MODE:-interleaved}"
PROMPT_MODE="${PROMPT_MODE:-synthetic}"
SEED="${SEED:-42}"
PAUSE_BETWEEN_ROUNDS="${PAUSE_BETWEEN_ROUNDS:-3.0}"

MAX_CONCURRENCY_FIXED="${MAX_CONCURRENCY_FIXED:-64}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-600}"
SKIP_SERVER="${SKIP_SERVER:-0}"

SGLANG_ENV_SCRIPT="${SGLANG_ENV_SCRIPT:-${HOME}/env.sh}"
PYTHON_BIN="${PYTHON_BIN:-/data2/ljr/a/downloads/miniconda3/envs/sgl/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_FAST="${E2E_FAST:-${SCRIPT_DIR}/e2e_bench_fast.py}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_cxl2_dram_output_conc_sweep_${TIMESTAMP}}"

HISPARSE_DRAM='{"top_k":2048,"device_buffer_size":4096}'

ROUND2_OUTPUT_MIN="${ROUND2_OUTPUT_MIN:-0}"
ROUND2_OUTPUT_MAX="${ROUND2_OUTPUT_MAX:-1024}"

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
    --max-concurrency-fixed) MAX_CONCURRENCY_FIXED="$2"; shift 2 ;;
    --cxl-dev) CXL_DEV_PATH="$2"; shift 2 ;;
    --cxl-dev-2) CXL_DEV_PATH_2="$2"; shift 2 ;;
    --cxl-map-bytes-per-device) CXL_MAP_BYTES_PER_DEVICE="$2"; shift 2 ;;
    --env-script) SGLANG_ENV_SCRIPT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,38p' "$0"
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

start_server_for_scheme() {
  local scheme="$1"
  local log_file="$2"
  local cfg=""
  case "${scheme}" in
    cxl_interleave_2) cfg="$(build_hisparse_cxl_interleave_2)" ;;
    dram) cfg="${HISPARSE_DRAM}" ;;
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

TOTAL_CASES=$(( 16 + 16 ))
CURRENT_CASE=0

{
  echo "run_cxl_interleave2_dram_output_concurrency_sweep.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "MODEL_PATH=${MODEL_PATH} MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  if [[ -n "${NUM_REQUESTS_CXL}" ]]; then
    echo "NUM_REQUESTS_CXL=${NUM_REQUESTS_CXL} (uniform)"
  else
    echo "NUM_REQUESTS_CXL=(unset) tiered by concurrency"
  fi
  echo "MAX_CONCURRENCY_FIXED=${MAX_CONCURRENCY_FIXED} (phase 2)"
  echo "CXL_DEV_PATH=${CXL_DEV_PATH} CXL_DEV_PATH_2=${CXL_DEV_PATH_2}"
  echo "CXL_MAP_BYTES_PER_DEVICE=${CXL_MAP_BYTES_PER_DEVICE}"
  echo "TOTAL_CASES=${TOTAL_CASES} (phase1=16 phase2=16)"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
} > "${META_FILE}"

echo "============================================================"
echo "CXL interleave×2 vs DRAM — output × concurrency → ${RESULTS_DIR}/"
echo "  Total cases: ${TOTAL_CASES}"
echo "============================================================"

# --- Phase 1: CXL interleave×2 then DRAM; conc 80,96; R2 random [0,1024] ---
PHASE1_SCHEMES=(cxl_interleave_2 dram)
PHASE1_CTX=(16384 32768 65536 131072)
PHASE1_CONC=(80 96)

OUTPUT_TOKENS="${OUTPUT_TOKENS_PHASE1:-${ORIG_OUTPUT_TOKENS}}"
ROUND2_OUTPUT_MIN=0
ROUND2_OUTPUT_MAX=1024

for conc in "${PHASE1_CONC[@]}"; do
  num_req="$(cxl_num_requests_for_conc "${conc}")"
  for tin in "${PHASE1_CTX[@]}"; do
    tag="$(ctx_tag "${tin}")"
    for scheme in "${PHASE1_SCHEMES[@]}"; do
      CURRENT_CASE=$((CURRENT_CASE + 1))
      base="p1_${scheme}_ctx${tag}_conc${conc}_r2rand0_1024"
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
      echo "  PHASE 1 ${CURRENT_CASE}/${TOTAL_CASES}: scheme=${scheme} ctx=${tin} (${tag}) conc=${conc} R2∈[0,1024] num-req=${num_req}"
      echo "################################################################"

      if [[ "${SKIP_SERVER}" == "1" ]]; then
        echo "  SKIP_SERVER=1 — client only"
      else
        kill_server 2>/dev/null || true
        start_server_for_scheme "${scheme}" "${srv_log}"
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
done

# --- Phase 2: DRAM then CXL; 128k+32k × fixed R2; conc=MAX_CONCURRENCY_FIXED ---
PHASE2_SCHEMES=(dram cxl_interleave_2)
PHASE2_CTX=(131072 32768)
PHASE2_OUT=(1024 2048 4096 8192)
conc2="${MAX_CONCURRENCY_FIXED}"
num_req2="$(cxl_num_requests_for_conc "${conc2}")"

for scheme in "${PHASE2_SCHEMES[@]}"; do
  for tin in "${PHASE2_CTX[@]}"; do
    tag="$(ctx_tag "${tin}")"
    for olen in "${PHASE2_OUT[@]}"; do
      CURRENT_CASE=$((CURRENT_CASE + 1))
      ot="$(out_fix_tag "${olen}")"
      base="p2_${scheme}_ctx${tag}_conc${conc2}_r2fix${ot}"
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
      echo "  PHASE 2 ${CURRENT_CASE}/${TOTAL_CASES}: scheme=${scheme} ctx=${tin} (${tag}) conc=${conc2} R2=${olen} (fixed) num-req=${num_req2}"
      echo "################################################################"

      if [[ "${SKIP_SERVER}" == "1" ]]; then
        echo "  SKIP_SERVER=1 — client only"
      else
        kill_server 2>/dev/null || true
        start_server_for_scheme "${scheme}" "${srv_log}"
        if ! wait_for_server; then
          echo "  ERROR: server failed to start — skipping"
          kill_server
          continue
        fi
      fi

      echo "  --- client -> ${out_file##*/} ---"
      run_e2e_fast "${out_file}" "${tin}" "${num_req2}" "${conc2}"

      if [[ "${SKIP_SERVER}" != "1" ]]; then
        kill_server
      fi
    done
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
