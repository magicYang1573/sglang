#!/bin/bash
# =============================================================================
# HiSparse: DRAM vs CXL (single-device) vs RDMA (2-NIC local0)
#   Context: 32k, 64k, 128k  |  max-concurrency: 128, 160, 192
#
# Lifecycle matches run_dram_cxl2_phased_mirror_buf6144_ph2_conc_x2.sh:
#   each case → fresh server → /health → e2e_bench_fast → tear down.
#
# Order: for each backend (dram, cxl, rdma), for each context length, for each
# concurrency (27 cases total).
#
# Usage:
#   export MODEL_PATH=/path/to/DeepSeek-V3.2-AWQ
#   bash test/cxl_utils/run_dram_cxl_rdma_ctx32k_64k_128k_conc128_160_192.sh
#
# Optional env:
#   MODEL_CLIENT, PORT, TP, DP_SIZE, MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS
#   MAX_RUNNING_REQUESTS — passed as --max-running-requests when set
#   HISPARSE_DEVICE_BUFFER_SIZE (default 6144) — used in dram/cxl JSON; rdma uses
#     RDMA_DEVICE_BUFFER_SIZE if set, else same value
#   REQ_PER_CONC_MULT (default 2) — num-requests = max_concurrency × this
#   NUM_REQUESTS — if set, overrides per-case num-requests for all cases
#   CXL_DEV_PATH, CXL_MAP_BYTES — single-device CXL (see build_hisparse_cxl)
#   IB_DEVICES (comma-separated), RDMA_NUM_IB_DEVS (default 2) — first N names
#     become ib_devs[] in rdma_pool JSON
#   NUM_UNIQUE_PROMPTS, OUTPUT_TOKENS, ROUND2_OUTPUT_MIN/MAX, REQUEST_RATE,
#     REPEAT_MODE, PROMPT_MODE, SEED, PAUSE_BETWEEN_ROUNDS
#   HF_ENDPOINT, RESULTS_DIR, SERVER_STARTUP_WAIT, SKIP_SERVER, FORCE_RERUN
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
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"

CXL_DEV_PATH="${CXL_DEV_PATH:-/dev/dax0.0}"
CXL_MAP_BYTES="${CXL_MAP_BYTES:-274877906944}"

HISPARSE_DEVICE_BUFFER_SIZE="${HISPARSE_DEVICE_BUFFER_SIZE:-6144}"
RDMA_DEVICE_BUFFER_SIZE="${RDMA_DEVICE_BUFFER_SIZE:-${HISPARSE_DEVICE_BUFFER_SIZE}}"

IB_DEVICES="${IB_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"
RDMA_NUM_IB_DEVS="${RDMA_NUM_IB_DEVS:-2}"
IFS=',' read -ra ALL_IB_DEVS <<< "${IB_DEVICES}"

REQ_PER_CONC_MULT="${REQ_PER_CONC_MULT:-2}"
NUM_REQUESTS="${NUM_REQUESTS:-}"

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
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-1800}"
SKIP_SERVER="${SKIP_SERVER:-0}"
SERVER_TEARDOWN_WAIT_SEC="${SERVER_TEARDOWN_WAIT_SEC:-90}"
SERVER_POST_KILL_SLEEP_SEC="${SERVER_POST_KILL_SLEEP_SEC:-8}"

SGLANG_ENV_SCRIPT="${SGLANG_ENV_SCRIPT:-${HOME}/env.sh}"
PYTHON_BIN="${PYTHON_BIN:-/data2/ljr/a/downloads/miniconda3/envs/sgl/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_FAST="${E2E_FAST:-${SCRIPT_DIR}/e2e_bench_fast.py}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_dram_cxl_rdma_ctx32_64_128k_conc128_160_192_${TIMESTAMP}}"

CONTEXT_LENGTHS=(32768 65536 131072)
CONCURRENCIES=(128 160 192)
SCHEMES=(dram cxl rdma)

build_hisparse_dram() {
  printf '%s' '{"top_k":2048,"device_buffer_size":'"${HISPARSE_DEVICE_BUFFER_SIZE}"'}'
}

build_hisparse_cxl() {
  printf '%s' '{"top_k":2048,"device_buffer_size":'"${HISPARSE_DEVICE_BUFFER_SIZE}"',"cxl":{"enabled":true,"dev_path":"'"${CXL_DEV_PATH}"'","map_bytes":'"${CXL_MAP_BYTES}"'}}'
}

build_hisparse_rdma() {
  local n="${RDMA_NUM_IB_DEVS}"
  local devs_json=""
  local i
  for ((i = 0; i < n; i++)); do
    if [[ $i -ge ${#ALL_IB_DEVS[@]} ]]; then
      echo "ERROR: IB_DEVICES has fewer than ${n} entries (need ${n} for rdma)." >&2
      return 1
    fi
    if [[ $i -gt 0 ]]; then devs_json+=","; fi
    devs_json+='"'"${ALL_IB_DEVS[$i]}"'"'
  done
  printf '%s' '{"top_k":2048,"device_buffer_size":'"${RDMA_DEVICE_BUFFER_SIZE}"',"rdma_pool":{"enabled":true,"local_ratio":0.0,"ib_devs":['"${devs_json}"']}}'
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
    --max-running-requests) MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; mkdir -p "${RESULTS_DIR}"; shift 2 ;;
    --hisparse-device-buffer-size) HISPARSE_DEVICE_BUFFER_SIZE="$2"; shift 2 ;;
    --rdma-device-buffer-size) RDMA_DEVICE_BUFFER_SIZE="$2"; shift 2 ;;
    --req-per-conc-mult) REQ_PER_CONC_MULT="$2"; shift 2 ;;
    --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
    --num-unique-prompts) NUM_UNIQUE_PROMPTS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --round2-output-min) ROUND2_OUTPUT_MIN="$2"; shift 2 ;;
    --round2-output-max) ROUND2_OUTPUT_MAX="$2"; shift 2 ;;
    --cxl-dev) CXL_DEV_PATH="$2"; shift 2 ;;
    --cxl-map-bytes) CXL_MAP_BYTES="$2"; shift 2 ;;
    --ib-devices) IB_DEVICES="$2"; IFS=',' read -ra ALL_IB_DEVS <<< "${IB_DEVICES}"; shift 2 ;;
    --rdma-num-ib-devs) RDMA_NUM_IB_DEVS="$2"; shift 2 ;;
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

mkdir -p "${RESULTS_DIR}"
META_FILE="${RESULTS_DIR}/run_meta.txt"

TOTAL_CASES=$(( ${#SCHEMES[@]} * ${#CONTEXT_LENGTHS[@]} * ${#CONCURRENCIES[@]} ))
CURRENT_CASE=0

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
    32768) echo "32k" ;;
    65536) echo "64k" ;;
    131072) echo "128k" ;;
    *) echo "in$1" ;;
  esac
}

hisparse_config_for_scheme() {
  local scheme="$1"
  case "${scheme}" in
    dram) build_hisparse_dram ;;
    cxl) build_hisparse_cxl ;;
    rdma) build_hisparse_rdma ;;
    *) echo "Unknown scheme: ${scheme}" >&2; return 1 ;;
  esac
}

start_server_for_scheme() {
  local scheme="$1"
  local log_file="$2"
  local cfg
  cfg="$(hisparse_config_for_scheme "${scheme}")" || return 1

  local max_run_args=()
  if [[ -n "${MAX_RUNNING_REQUESTS}" ]]; then
    max_run_args=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
  fi

  echo "  Launching server: scheme=${scheme} (log -> ${log_file##*/})"
  echo "  HiSparse config: ${cfg}"
  ensure_http_down_before_launch || return 1

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
      "${max_run_args[@]}" \
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
      "${max_run_args[@]}" \
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
  echo "run_dram_cxl_rdma_ctx32k_64k_128k_conc128_160_192.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "SCHEMES=${SCHEMES[*]}"
  echo "CONTEXT_LENGTHS=${CONTEXT_LENGTHS[*]}"
  echo "CONCURRENCIES=${CONCURRENCIES[*]}"
  echo "REQ_PER_CONC_MULT=${REQ_PER_CONC_MULT} NUM_REQUESTS=${NUM_REQUESTS:-}"
  echo "HISPARSE_DEVICE_BUFFER_SIZE=${HISPARSE_DEVICE_BUFFER_SIZE} RDMA_DEVICE_BUFFER_SIZE=${RDMA_DEVICE_BUFFER_SIZE}"
  echo "RDMA_NUM_IB_DEVS=${RDMA_NUM_IB_DEVS} IB_DEVICES=${IB_DEVICES}"
  echo "CXL_DEV_PATH=${CXL_DEV_PATH} CXL_MAP_BYTES=${CXL_MAP_BYTES}"
  echo "MODEL_PATH=${MODEL_PATH} MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT} MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-}"
  echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC} MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS}"
  echo "TOTAL_CASES=${TOTAL_CASES}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
} > "${META_FILE}"

echo "============================================================"
echo "DRAM / CXL / RDMA × ctx(32k,64k,128k) × conc(128,160,192)"
echo "  → ${RESULTS_DIR}/"
echo "  Cases: ${TOTAL_CASES} (one server per case)"
echo "============================================================"

for scheme in "${SCHEMES[@]}"; do
  for tin in "${CONTEXT_LENGTHS[@]}"; do
    tag="$(ctx_tag "${tin}")"
    for conc in "${CONCURRENCIES[@]}"; do
      CURRENT_CASE=$((CURRENT_CASE + 1))
      if [[ -n "${NUM_REQUESTS}" ]]; then
        nreq="${NUM_REQUESTS}"
      else
        nreq=$(( conc * REQ_PER_CONC_MULT ))
      fi
      seq="$(printf '%02d' "${CURRENT_CASE}")"
      base="${seq}_${scheme}_ctx${tag}_conc${conc}_n${nreq}"
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
      echo "  CASE ${CURRENT_CASE}/${TOTAL_CASES}: scheme=${scheme} ctx=${tin} (${tag}) conc=${conc} num-req=${nreq}"
      echo "################################################################"

      if [[ "${SKIP_SERVER}" == "1" ]]; then
        echo "  SKIP_SERVER=1 — client only"
      else
        kill_server 2>/dev/null || true
        if ! start_server_for_scheme "${scheme}" "${srv_log}"; then
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
      run_e2e_fast "${out_file}" "${tin}" "${nreq}" "${conc}"

      if [[ "${SKIP_SERVER}" != "1" ]]; then
        kill_server
      fi
    done
  done
done

echo ""
echo "Done. Results under: ${RESULTS_DIR}/"
