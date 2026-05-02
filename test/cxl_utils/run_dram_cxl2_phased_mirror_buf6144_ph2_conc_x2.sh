#!/bin/bash
# =============================================================================
# Variant of run_dram_cxl2_phased_mirror_req512.sh (dc_0417 experiment driver):
#   - HiSparse device_buffer_size: 6144 (was 4096)
#   - Phase 2: num-requests = concurrency × 2 (default PH2_REQUESTS_MULT=2; was 8)
#
# Same phase matrix / filenames otherwise; P2 output JSON basename uses n{conc*2}
# (e.g. conc8 → n16). Round 2 in e2e_bench_fast.py is the measured steady batch and
# equals --num-requests for that run.
# =============================================================================
#
# For each backend in {dram, cxl_interleave_2}, run phases 1–3; phase 4 CXL1/CXL2.
#
#   Phase 1 — ctx 16k/32k/64k/128k, R2 fixed 1k, conc=64, num-requests=NUM_REQUESTS
#   Phase 2 — same ctx, R2 fixed 1k, conc 8..96, num-requests=conc×PH2_REQUESTS_MULT (default 2)
#   Phase 3 — same ctx, R2 fixed 2k/4k/8k, conc=64, tiered num-requests
#   Phase 4 — cxl1 vs cxl2, ctx sweep, conc=64, num-requests=NUM_REQUESTS
#
# Usage:
#   export MODEL_PATH=/path/to/DeepSeek-V3.2-AWQ
#   bash test/cxl_utils/run_dram_cxl2_phased_mirror_buf6144_ph2_conc_x2.sh
#
# Optional env: same as parent script; override PH2_REQUESTS_MULT to scale P2 request count.
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
CXL_MAP_BYTES="${CXL_MAP_BYTES:-274877906944}"
CXL_MAP_BYTES_PER_DEVICE="${CXL_MAP_BYTES_PER_DEVICE:-274877906944}"

NUM_REQUESTS="${NUM_REQUESTS:-512}"
PH2_REQUESTS_MULT="${PH2_REQUESTS_MULT:-2}"
NUM_REQUESTS_P3_2K="${NUM_REQUESTS_P3_2K:-256}"
NUM_REQUESTS_P3_4K="${NUM_REQUESTS_P3_4K:-256}"
NUM_REQUESTS_P3_8K="${NUM_REQUESTS_P3_8K:-128}"
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
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_dram_cxl2_phased_buf6144_ph2c2_${TIMESTAMP}}"

ROUND2_OUTPUT_MIN="${ROUND2_OUTPUT_MIN:-0}"
ROUND2_OUTPUT_MAX="${ROUND2_OUTPUT_MAX:-1024}"

HISPARSE_DEVICE_BUFFER_SIZE="${HISPARSE_DEVICE_BUFFER_SIZE:-6144}"
HISPARSE_DRAM='{"top_k":2048,"device_buffer_size":'"${HISPARSE_DEVICE_BUFFER_SIZE}"'}'

build_hisparse_cxl() {
  printf '%s' '{"top_k":2048,"device_buffer_size":'"${HISPARSE_DEVICE_BUFFER_SIZE}"',"cxl":{"enabled":true,"dev_path":"'"${CXL_DEV_PATH}"'","map_bytes":'"${CXL_MAP_BYTES}"'}}'
}

build_hisparse_cxl_interleave_2() {
  printf '%s' '{"top_k":2048,"device_buffer_size":'"${HISPARSE_DEVICE_BUFFER_SIZE}"',"cxl":{"enabled":true,"dev_paths":["'"${CXL_DEV_PATH}"'","'"${CXL_DEV_PATH_2}"'"],"map_bytes_per_device":'"${CXL_MAP_BYTES_PER_DEVICE}"'}}'
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
    --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
    --num-unique-prompts) NUM_UNIQUE_PROMPTS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --cxl-dev) CXL_DEV_PATH="$2"; shift 2 ;;
    --cxl-dev-2) CXL_DEV_PATH_2="$2"; shift 2 ;;
    --cxl-map-bytes) CXL_MAP_BYTES="$2"; shift 2 ;;
    --cxl-map-bytes-per-device) CXL_MAP_BYTES_PER_DEVICE="$2"; shift 2 ;;
    --env-script) SGLANG_ENV_SCRIPT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,35p' "$0"
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

CASES_P123=$(( ${#CTX_ALL[@]} + ${#CTX_ALL[@]} * ${#PH2_CONC[@]} + ${#CTX_ALL[@]} * ${#PH3_OUT[@]} ))
CASES_P4=$(( ${#CTX_ALL[@]} * 2 ))
TOTAL_CASES=$(( CASES_P123 * 2 + CASES_P4 ))

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

p3_num_requests_for_olen() {
  case "$1" in
    2048) printf '%s' "${NUM_REQUESTS_P3_2K}" ;;
    4096) printf '%s' "${NUM_REQUESTS_P3_4K}" ;;
    8192) printf '%s' "${NUM_REQUESTS_P3_8K}" ;;
    *) printf '%s' "${NUM_REQUESTS}" ;;
  esac
}

start_server_for_scheme() {
  local scheme="$1"
  local log_file="$2"
  local cfg=""
  case "${scheme}" in
    dram) cfg="${HISPARSE_DRAM}" ;;
    cxl) cfg="$(build_hisparse_cxl)" ;;
    cxl_interleave_2) cfg="$(build_hisparse_cxl_interleave_2)" ;;
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

{
  echo "run_dram_cxl2_phased_mirror_buf6144_ph2_conc_x2.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "HISPARSE_DEVICE_BUFFER_SIZE=${HISPARSE_DEVICE_BUFFER_SIZE}"
  echo "MODEL_PATH=${MODEL_PATH} MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  echo "NUM_REQUESTS=${NUM_REQUESTS} PH2_REQUESTS_MULT=${PH2_REQUESTS_MULT}"
  echo "NUM_REQUESTS_P3_2K=${NUM_REQUESTS_P3_2K} NUM_REQUESTS_P3_4K=${NUM_REQUESTS_P3_4K} NUM_REQUESTS_P3_8K=${NUM_REQUESTS_P3_8K}"
  echo "CXL_DEV_PATH=${CXL_DEV_PATH} CXL_DEV_PATH_2=${CXL_DEV_PATH_2}"
  echo "CXL_MAP_BYTES=${CXL_MAP_BYTES} CXL_MAP_BYTES_PER_DEVICE=${CXL_MAP_BYTES_PER_DEVICE}"
  echo "TOTAL_CASES=${TOTAL_CASES} (dram+cxl2 mirror p1-3=${CASES_P123} each; p4=${CASES_P4})"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
} > "${META_FILE}"

GLOBAL_IDX=0

echo "============================================================"
echo "DRAM + CXL2 phased (buf=${HISPARSE_DEVICE_BUFFER_SIZE}, P2 req=conc×${PH2_REQUESTS_MULT}) → ${RESULTS_DIR}/"
echo "  Total cases: ${TOTAL_CASES}"
echo "  P1/P4 num-req=${NUM_REQUESTS}; P2 num-req=conc×${PH2_REQUESTS_MULT}; P3 2k/4k/8k=${NUM_REQUESTS_P3_2K}/${NUM_REQUESTS_P3_4K}/${NUM_REQUESTS_P3_8K}"
echo "============================================================"

# --- Phases 1–3 for dram, then for cxl_interleave_2 ---
for BACKEND_PAIR in "dram:dram" "cxl_interleave_2:cxl2"; do
  IFS=':' read -r SCHEME FILETAG <<< "${BACKEND_PAIR}"

  # Phase 1
  P1_SEQ=0
  OUTPUT_TOKENS=1024
  ROUND2_OUTPUT_MIN=1024
  ROUND2_OUTPUT_MAX=1024
  for tin in "${CTX_ALL[@]}"; do
    P1_SEQ=$((P1_SEQ + 1))
    GLOBAL_IDX=$((GLOBAL_IDX + 1))
    tag="$(ctx_tag "${tin}")"
    seq="$(printf '%02d' "${P1_SEQ}")"
    base="p1_${seq}_${FILETAG}_ctx${tag}_conc64_r2fix1k_n${NUM_REQUESTS}"
    out_file="${RESULTS_DIR}/${base}.json"
    srv_log="${RESULTS_DIR}/server_${base}.log"

    if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
      echo ""
      echo "################################################################"
      echo "  SKIP ${GLOBAL_IDX}/${TOTAL_CASES}: existing ${out_file##*/}"
      echo "################################################################"
      continue
    fi

    echo ""
    echo "################################################################"
    echo "  P1 ${GLOBAL_IDX}/${TOTAL_CASES} scheme=${SCHEME} (${FILETAG}) ctx=${tin} (${tag}) conc=64 R2=1024 num-req=${NUM_REQUESTS}"
    echo "################################################################"

    if [[ "${SKIP_SERVER}" == "1" ]]; then
      echo "  SKIP_SERVER=1 — client only"
    else
      kill_server 2>/dev/null || true
      start_server_for_scheme "${SCHEME}" "${srv_log}"
      if ! wait_for_server; then
        echo "  ERROR: server failed to start — skipping"
        kill_server
        continue
      fi
    fi

    echo "  --- client -> ${out_file##*/} ---"
    run_e2e_fast "${out_file}" "${tin}" "${NUM_REQUESTS}" 64

    if [[ "${SKIP_SERVER}" != "1" ]]; then
      kill_server
    fi
  done

  # Phase 2
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
      base="p2_${seq}_${FILETAG}_ctx${tag}_conc${conc}_r2fix1k_n${nreq_p2}"
      out_file="${RESULTS_DIR}/${base}.json"
      srv_log="${RESULTS_DIR}/server_${base}.log"

      if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
        echo ""
        echo "################################################################"
        echo "  SKIP ${GLOBAL_IDX}/${TOTAL_CASES}: existing ${out_file##*/}"
        echo "################################################################"
        continue
      fi

      echo ""
      echo "################################################################"
      echo "  P2 ${GLOBAL_IDX}/${TOTAL_CASES} scheme=${SCHEME} (${FILETAG}) ctx=${tin} (${tag}) conc=${conc} R2=1024 num-req=${nreq_p2} (=conc×${PH2_REQUESTS_MULT})"
      echo "################################################################"

      if [[ "${SKIP_SERVER}" == "1" ]]; then
        echo "  SKIP_SERVER=1 — client only"
      else
        kill_server 2>/dev/null || true
        start_server_for_scheme "${SCHEME}" "${srv_log}"
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

  # Phase 3
  P3_SEQ=0
  for tin in "${CTX_ALL[@]}"; do
    tag="$(ctx_tag "${tin}")"
    for olen in "${PH3_OUT[@]}"; do
      P3_SEQ=$((P3_SEQ + 1))
      GLOBAL_IDX=$((GLOBAL_IDX + 1))
      nreq_p3="$(p3_num_requests_for_olen "${olen}")"
      ot="$(out_fix_tag "${olen}")"
      seq="$(printf '%02d' "${P3_SEQ}")"
      base="p3_${seq}_${FILETAG}_ctx${tag}_conc64_r2fix${ot}_n${nreq_p3}"
      out_file="${RESULTS_DIR}/${base}.json"
      srv_log="${RESULTS_DIR}/server_${base}.log"
      OUTPUT_TOKENS="${olen}"
      ROUND2_OUTPUT_MIN="${olen}"
      ROUND2_OUTPUT_MAX="${olen}"

      if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
        echo ""
        echo "################################################################"
        echo "  SKIP ${GLOBAL_IDX}/${TOTAL_CASES}: existing ${out_file##*/}"
        echo "################################################################"
        continue
      fi

      echo ""
      echo "################################################################"
      echo "  P3 ${GLOBAL_IDX}/${TOTAL_CASES} scheme=${SCHEME} (${FILETAG}) ctx=${tin} (${tag}) conc=64 R2=${olen} num-req=${nreq_p3}"
      echo "################################################################"

      if [[ "${SKIP_SERVER}" == "1" ]]; then
        echo "  SKIP_SERVER=1 — client only"
      else
        kill_server 2>/dev/null || true
        start_server_for_scheme "${SCHEME}" "${srv_log}"
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

  OUTPUT_TOKENS=1024
  ROUND2_OUTPUT_MIN=1024
  ROUND2_OUTPUT_MAX=1024
done

# --- Phase 4: CXL-1 vs CXL interleave×2 @ ctx×1k @ conc=64 ---
P4_SEQ=0
OUTPUT_TOKENS=1024
ROUND2_OUTPUT_MIN=1024
ROUND2_OUTPUT_MAX=1024
for BACKEND_PAIR in "cxl:cxl1" "cxl_interleave_2:cxl2"; do
  IFS=':' read -r SCHEME FILETAG <<< "${BACKEND_PAIR}"
  for tin in "${CTX_ALL[@]}"; do
    P4_SEQ=$((P4_SEQ + 1))
    GLOBAL_IDX=$((GLOBAL_IDX + 1))
    tag="$(ctx_tag "${tin}")"
    seq="$(printf '%02d' "${P4_SEQ}")"
    base="p4_${seq}_${FILETAG}_ctx${tag}_conc64_r2fix1k_n${NUM_REQUESTS}"
    out_file="${RESULTS_DIR}/${base}.json"
    srv_log="${RESULTS_DIR}/server_${base}.log"

    if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
      echo ""
      echo "################################################################"
      echo "  SKIP ${GLOBAL_IDX}/${TOTAL_CASES}: existing ${out_file##*/}"
      echo "################################################################"
      continue
    fi

    echo ""
    echo "################################################################"
    echo "  P4 ${GLOBAL_IDX}/${TOTAL_CASES} scheme=${SCHEME} (${FILETAG}) ctx=${tin} (${tag}) conc=64 R2=1024 num-req=${NUM_REQUESTS}"
    echo "################################################################"

    if [[ "${SKIP_SERVER}" == "1" ]]; then
      echo "  SKIP_SERVER=1 — client only"
    else
      kill_server 2>/dev/null || true
      start_server_for_scheme "${SCHEME}" "${srv_log}"
      if ! wait_for_server; then
        echo "  ERROR: server failed to start — skipping"
        kill_server
        continue
      fi
    fi

    echo "  --- client -> ${out_file##*/} ---"
    run_e2e_fast "${out_file}" "${tin}" "${NUM_REQUESTS}" 64

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
print(f"{'file':<80} {'R2_TTFT_ms':>12} {'R2_TBT_ms':>12}")
print("-" * 106)
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
        print(f"{bn:<80} {tt:>12.1f} {tb:>12.1f}")
    except Exception as e:
        print(f"{bn:<80} {'ERR':>12} {str(e)[:12]:>12}")
PY
