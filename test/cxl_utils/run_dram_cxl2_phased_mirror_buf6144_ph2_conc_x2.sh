#!/bin/bash
# =============================================================================
# Variant: HiSparse device_buffer_size=6144; num-requests = concurrency × REQ_PER_CONC_MULT
# (default mult=2) for every phase; P1–P3 interleave DRAM then CXL×2 per test point.
#
# Order (vs parent run_dram_cxl2_phased_mirror_req512.sh):
#   Parent: all DRAM (P1→P2→P3), then all CXL2 (P1→P2→P3).
#   Here:   each (ctx[, conc[, olen]]) runs dram → cxl2 before advancing.
#
# Phase 4 has no DRAM — per ctx: cxl1 → cxl2 (single-device vs interleave×2).
# =============================================================================
#
# Usage:
#   export MODEL_PATH=/path/to/DeepSeek-V3.2-AWQ
#   bash test/cxl_utils/run_dram_cxl2_phased_mirror_buf6144_ph2_conc_x2.sh
#
# Optional env:
#   REQ_PER_CONC_MULT (default 2) — num-requests = max_concurrency × this, all phases
#   P134_CONC (default 64) — concurrency for phase 1, 3, and 4
#   HISPARSE_DEVICE_BUFFER_SIZE (default 6144)
#   Plus same server/client env as parent (MODEL_CLIENT, PORT, CXL_*, etc.)
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

REQ_PER_CONC_MULT="${REQ_PER_CONC_MULT:-2}"
P134_CONC="${P134_CONC:-64}"

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
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_dram_cxl2_buf6144_req2xconc_alt_${TIMESTAMP}}"

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
    --num-unique-prompts) NUM_UNIQUE_PROMPTS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --cxl-dev) CXL_DEV_PATH="$2"; shift 2 ;;
    --cxl-dev-2) CXL_DEV_PATH_2="$2"; shift 2 ;;
    --cxl-map-bytes) CXL_MAP_BYTES="$2"; shift 2 ;;
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

ORIG_OUTPUT_TOKENS="${OUTPUT_TOKENS}"

mkdir -p "${RESULTS_DIR}"
META_FILE="${RESULTS_DIR}/run_meta.txt"

NREQ_P134=$(( P134_CONC * REQ_PER_CONC_MULT ))

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
  echo "REQ_PER_CONC_MULT=${REQ_PER_CONC_MULT} P134_CONC=${P134_CONC} NREQ_P134=${NREQ_P134}"
  echo "P1-P3 order: interleaved dram then cxl2 per (ctx[, conc[, olen]]); P4 per ctx: cxl1 then cxl2"
  echo "MODEL_PATH=${MODEL_PATH} MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  echo "CXL_DEV_PATH=${CXL_DEV_PATH} CXL_DEV_PATH_2=${CXL_DEV_PATH_2}"
  echo "CXL_MAP_BYTES=${CXL_MAP_BYTES} CXL_MAP_BYTES_PER_DEVICE=${CXL_MAP_BYTES_PER_DEVICE}"
  echo "TOTAL_CASES=${TOTAL_CASES} (dram+cxl2 mirror p1-3=${CASES_P123} each; p4=${CASES_P4})"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
} > "${META_FILE}"

GLOBAL_IDX=0

echo "============================================================"
echo "buf=${HISPARSE_DEVICE_BUFFER_SIZE} | num-req = conc × ${REQ_PER_CONC_MULT} | DRAM↔CXL2 interleaved (P1–P3)"
echo "  → ${RESULTS_DIR}/"
echo "  Total cases: ${TOTAL_CASES}"
echo "  P1/P3/P4 conc=${P134_CONC} → num-req=${NREQ_P134}; P2 num-req=conc×${REQ_PER_CONC_MULT}"
echo "============================================================"

# --- Phase 1: per ctx — dram, then cxl2 ---
P1_SEQ=0
OUTPUT_TOKENS=1024
ROUND2_OUTPUT_MIN=1024
ROUND2_OUTPUT_MAX=1024
for tin in "${CTX_ALL[@]}"; do
  tag="$(ctx_tag "${tin}")"
  for BACKEND_PAIR in "dram:dram" "cxl_interleave_2:cxl2"; do
    IFS=':' read -r SCHEME FILETAG <<< "${BACKEND_PAIR}"
    P1_SEQ=$((P1_SEQ + 1))
    GLOBAL_IDX=$((GLOBAL_IDX + 1))
    seq="$(printf '%02d' "${P1_SEQ}")"
    base="p1_${seq}_${FILETAG}_ctx${tag}_conc${P134_CONC}_r2fix1k_n${NREQ_P134}"
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
    echo "  P1 ${GLOBAL_IDX}/${TOTAL_CASES} scheme=${SCHEME} (${FILETAG}) ctx=${tin} (${tag}) conc=${P134_CONC} R2=1024 num-req=${NREQ_P134}"
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
    run_e2e_fast "${out_file}" "${tin}" "${NREQ_P134}" "${P134_CONC}"

    if [[ "${SKIP_SERVER}" != "1" ]]; then
      kill_server
    fi
  done
done

# --- Phase 2: per (ctx, conc) — dram, then cxl2 ---
P2_SEQ=0
OUTPUT_TOKENS=1024
ROUND2_OUTPUT_MIN=1024
ROUND2_OUTPUT_MAX=1024
for tin in "${CTX_ALL[@]}"; do
  tag="$(ctx_tag "${tin}")"
  for conc in "${PH2_CONC[@]}"; do
    nreq_p2=$(( conc * REQ_PER_CONC_MULT ))
    for BACKEND_PAIR in "dram:dram" "cxl_interleave_2:cxl2"; do
      IFS=':' read -r SCHEME FILETAG <<< "${BACKEND_PAIR}"
      P2_SEQ=$((P2_SEQ + 1))
      GLOBAL_IDX=$((GLOBAL_IDX + 1))
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
      echo "  P2 ${GLOBAL_IDX}/${TOTAL_CASES} scheme=${SCHEME} (${FILETAG}) ctx=${tin} (${tag}) conc=${conc} R2=1024 num-req=${nreq_p2} (=conc×${REQ_PER_CONC_MULT})"
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
done

# --- Phase 3: per (ctx, R2 len) — dram, then cxl2 ---
P3_SEQ=0
for tin in "${CTX_ALL[@]}"; do
  tag="$(ctx_tag "${tin}")"
  for olen in "${PH3_OUT[@]}"; do
    ot="$(out_fix_tag "${olen}")"
    OUTPUT_TOKENS="${olen}"
    ROUND2_OUTPUT_MIN="${olen}"
    ROUND2_OUTPUT_MAX="${olen}"
    for BACKEND_PAIR in "dram:dram" "cxl_interleave_2:cxl2"; do
      IFS=':' read -r SCHEME FILETAG <<< "${BACKEND_PAIR}"
      P3_SEQ=$((P3_SEQ + 1))
      GLOBAL_IDX=$((GLOBAL_IDX + 1))
      seq="$(printf '%02d' "${P3_SEQ}")"
      base="p3_${seq}_${FILETAG}_ctx${tag}_conc${P134_CONC}_r2fix${ot}_n${NREQ_P134}"
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
      echo "  P3 ${GLOBAL_IDX}/${TOTAL_CASES} scheme=${SCHEME} (${FILETAG}) ctx=${tin} (${tag}) conc=${P134_CONC} R2=${olen} num-req=${NREQ_P134}"
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
      run_e2e_fast "${out_file}" "${tin}" "${NREQ_P134}" "${P134_CONC}"

      if [[ "${SKIP_SERVER}" != "1" ]]; then
        kill_server
      fi
    done
  done
done

OUTPUT_TOKENS=1024
ROUND2_OUTPUT_MIN=1024
ROUND2_OUTPUT_MAX=1024

# --- Phase 4: per ctx — cxl1 then cxl2 (no DRAM); num-req = P134_CONC × mult ---
P4_SEQ=0
OUTPUT_TOKENS=1024
ROUND2_OUTPUT_MIN=1024
ROUND2_OUTPUT_MAX=1024
for tin in "${CTX_ALL[@]}"; do
  tag="$(ctx_tag "${tin}")"
  for BACKEND_PAIR in "cxl:cxl1" "cxl_interleave_2:cxl2"; do
    IFS=':' read -r SCHEME FILETAG <<< "${BACKEND_PAIR}"
    P4_SEQ=$((P4_SEQ + 1))
    GLOBAL_IDX=$((GLOBAL_IDX + 1))
    seq="$(printf '%02d' "${P4_SEQ}")"
    base="p4_${seq}_${FILETAG}_ctx${tag}_conc${P134_CONC}_r2fix1k_n${NREQ_P134}"
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
    echo "  P4 ${GLOBAL_IDX}/${TOTAL_CASES} scheme=${SCHEME} (${FILETAG}) ctx=${tin} (${tag}) conc=${P134_CONC} R2=1024 num-req=${NREQ_P134}"
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
    run_e2e_fast "${out_file}" "${tin}" "${NREQ_P134}" "${P134_CONC}"

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
