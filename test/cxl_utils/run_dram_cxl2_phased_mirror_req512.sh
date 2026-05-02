#!/bin/bash
# =============================================================================
# DRAM + CXL interleave×2 — mirror run_rdma_2nic_phased_req512 client matrix
# =============================================================================
#
# For each backend in {dram, cxl_interleave_2}, run the same three phases as
# test/cxl_utils/run_rdma_2nic_phased_req512.sh (client params only; HiSparse
# pool differs — host DRAM vs CXL dev_paths[2]):
#
#   Phase 1 — ctx 16k/32k/64k/128k, R2 fixed 1k, conc=64, num-requests=NUM_REQUESTS
#   Phase 2 — same ctx, R2 fixed 1k, conc 8..96, num-requests=conc×PH2_REQUESTS_MULT (default 8)
#   Phase 3 — same ctx, R2 fixed 2k/4k/8k, conc=64, tiered num-requests
#
# Phase 4 (extra) — CXL single device ("cxl1") and CXL interleave×2 ("cxl2"):
#   ctx 16k/32k/64k/128k, R2 fixed 1k, conc=64, num-requests=NUM_REQUESTS
#   (cxl2 here matches Phase-1 cxl2 client settings; separate fresh runs / filenames.)
#
# Per case: start server → /health → e2e_bench_fast → stop server.
# Filenames: p{1..3}_{seq}_{dram|cxl2}_... ; Phase4: p4_{seq}_{cxl1|cxl2}_...
#
# Usage:
#   export MODEL_PATH=/path/to/DeepSeek-V3.2-AWQ
#   bash test/cxl_utils/run_dram_cxl2_phased_mirror_req512.sh
#
# Optional env:
#   MODEL_CLIENT, PORT, TP, DP_SIZE, MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS
#   CXL_DEV_PATH, CXL_DEV_PATH_2, CXL_MAP_BYTES, CXL_MAP_BYTES_PER_DEVICE
#   NUM_REQUESTS (default 512) — Phase 1 and Phase 4 only
#   PH2_REQUESTS_MULT (default 8) — Phase 2 num-requests = max_concurrency × this
#   NUM_REQUESTS_P3_2K (256), NUM_REQUESTS_P3_4K (256), NUM_REQUESTS_P3_8K (128)
#   NUM_UNIQUE_PROMPTS, OUTPUT_TOKENS, REQUEST_RATE, REPEAT_MODE, PROMPT_MODE
#   SEED, PAUSE_BETWEEN_ROUNDS, HF_ENDPOINT, RESULTS_DIR, SERVER_STARTUP_WAIT (default 1800)
#   SERVER_POST_HEALTH_SLEEP_SEC (default 300) — sleep after /health for CUDA graph / warmup; 0 to skip
#   SKIP_SERVER, FORCE_RERUN, SGLANG_ENV_SCRIPT, PYTHON_BIN
#   SERVER_TEARDOWN_WAIT_SEC (90), SERVER_POST_KILL_SLEEP_SEC (8) — teardown between cases
#
# DRAM HiSparse JSON matches run_cxl_interleave2_dram_output_concurrency_sweep.sh.
# CXL JSON matches run_cxl_rdma_nic_experiments.sh.
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
PH2_REQUESTS_MULT="${PH2_REQUESTS_MULT:-8}"
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
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-1800}"
SERVER_POST_HEALTH_SLEEP_SEC="${SERVER_POST_HEALTH_SLEEP_SEC:-300}"
SKIP_SERVER="${SKIP_SERVER:-0}"
# Between cases: tear down full launch_server tree + wait until HTTP port is free (avoids rpc_port leaks).
SERVER_TEARDOWN_WAIT_SEC="${SERVER_TEARDOWN_WAIT_SEC:-90}"
SERVER_POST_KILL_SLEEP_SEC="${SERVER_POST_KILL_SLEEP_SEC:-8}"

SGLANG_ENV_SCRIPT="${SGLANG_ENV_SCRIPT:-${HOME}/env.sh}"
PYTHON_BIN="${PYTHON_BIN:-/data2/ljr/a/downloads/miniconda3/envs/sgl/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_FAST="${E2E_FAST:-${SCRIPT_DIR}/e2e_bench_fast.py}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_dram_cxl2_phased_mirror_${TIMESTAMP}}"

ROUND2_OUTPUT_MIN="${ROUND2_OUTPUT_MIN:-0}"
ROUND2_OUTPUT_MAX="${ROUND2_OUTPUT_MAX:-1024}"

HISPARSE_DRAM='{"top_k":2048,"device_buffer_size":4096}'

build_hisparse_cxl() {
  printf '%s' '{"top_k":2048,"device_buffer_size":4096,"cxl":{"enabled":true,"dev_path":"'"${CXL_DEV_PATH}"'","map_bytes":'"${CXL_MAP_BYTES}"'}}'
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
      sed -n '1,45p' "$0"
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
  if [[ "${SERVER_POST_HEALTH_SLEEP_SEC:-0}" -gt 0 ]]; then
    echo "  Sleeping ${SERVER_POST_HEALTH_SLEEP_SEC}s after /health (CUDA graph / server warmup)..."
    sleep "${SERVER_POST_HEALTH_SLEEP_SEC}"
  fi
}

# Recursive kill when setsid is unavailable (children-first).
kill_tree_term() {
  local pid=$1
  local c
  for c in $(pgrep -P "${pid}" 2>/dev/null || true); do
    kill_tree_term "${c}"
  done
  kill -TERM "${pid}" 2>/dev/null || true
}

kill_tree_kill() {
  local pid=$1
  local c
  for c in $(pgrep -P "${pid}" 2>/dev/null || true); do
    kill_tree_kill "${c}"
  done
  kill -KILL "${pid}" 2>/dev/null || true
}

wait_http_port_down() {
  local url="http://127.0.0.1:${PORT}/health"
  local max_wait="${1:-${SERVER_TEARDOWN_WAIT_SEC:-90}}"
  local waited=0
  while curl -sSf --connect-timeout 1 "${url}" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if [[ $((waited % 15)) -eq 0 ]]; then
      echo "  ... still waiting for ${PORT} to go down (${waited}s)"
    fi
    if [[ "${waited}" -ge "${max_wait}" ]]; then
      return 1
    fi
  done
  return 0
}

kill_server() {
  if [[ -z "${SERVER_PID:-}" ]]; then
    return 0
  fi
  local pid="${SERVER_PID}"
  echo "  Stopping server (PID=${pid}, PORT=${PORT})..."
  if command -v setsid >/dev/null 2>&1; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  else
    kill_tree_term "${pid}"
  fi

  local max_wait="${SERVER_TEARDOWN_WAIT_SEC:-90}"
  if ! wait_http_port_down "${max_wait}"; then
    echo "  WARN: HTTP still up after ${max_wait}s — sending SIGKILL to process group / tree"
    if command -v setsid >/dev/null 2>&1; then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
    else
      kill_tree_kill "${pid}"
    fi
    wait_http_port_down "${max_wait}" || echo "  WARN: ${PORT} may still be bound; next start may fail"
  fi

  wait "${pid}" 2>/dev/null || true
  unset SERVER_PID
  sleep "${SERVER_POST_KILL_SLEEP_SEC:-8}"
}

ensure_http_down_before_launch() {
  [[ "${SKIP_SERVER}" == "1" ]] && return 0
  local url="http://127.0.0.1:${PORT}/health"
  if ! curl -sSf --connect-timeout 1 "${url}" >/dev/null 2>&1; then
    return 0
  fi
  echo "  PORT ${PORT} still has a listener — waiting up to ${SERVER_TEARDOWN_WAIT_SEC:-90}s before launch..."
  wait_http_port_down "${SERVER_TEARDOWN_WAIT_SEC:-90}"
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
  ensure_http_down_before_launch || return 1
  # setsid → new session + process group so kill_server can signal the whole launch_server tree.
  # shellcheck disable=SC2086
  if command -v setsid >/dev/null 2>&1; then
    setsid env SGLANG_ENABLE_JIT_DEEPGEMM=0 "${PYTHON_BIN}" -m sglang.launch_server \
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
  else
    env SGLANG_ENABLE_JIT_DEEPGEMM=0 "${PYTHON_BIN}" -m sglang.launch_server \
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
  fi
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
  echo "run_dram_cxl2_phased_mirror_req512.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "MODEL_PATH=${MODEL_PATH} MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  echo "SERVER_STARTUP_WAIT=${SERVER_STARTUP_WAIT:-1800} SERVER_POST_HEALTH_SLEEP_SEC=${SERVER_POST_HEALTH_SLEEP_SEC:-300}"
  echo "SERVER_TEARDOWN_WAIT_SEC=${SERVER_TEARDOWN_WAIT_SEC:-90} SERVER_POST_KILL_SLEEP_SEC=${SERVER_POST_KILL_SLEEP_SEC:-8}"
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
echo "DRAM + CXL2 phased mirror + CXL1/CXL2 sweep → ${RESULTS_DIR}/"
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
      if ! start_server_for_scheme "${SCHEME}" "${srv_log}"; then
        echo "  ERROR: could not free PORT=${PORT} — skipping case"
        continue
      fi
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
        if ! start_server_for_scheme "${SCHEME}" "${srv_log}"; then
          echo "  ERROR: could not free PORT=${PORT} — skipping case"
          continue
        fi
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
        if ! start_server_for_scheme "${SCHEME}" "${srv_log}"; then
          echo "  ERROR: could not free PORT=${PORT} — skipping case"
          continue
        fi
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
      if ! start_server_for_scheme "${SCHEME}" "${srv_log}"; then
        echo "  ERROR: could not free PORT=${PORT} — skipping case"
        continue
      fi
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
