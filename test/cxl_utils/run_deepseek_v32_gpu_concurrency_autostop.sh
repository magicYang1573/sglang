#!/bin/bash
# =============================================================================
# DeepSeek-V3.2 full-GPU (no HiSparse) — context × concurrency throughput sweep
# =============================================================================
#
# Goal:
#   - Do NOT use HiSparse / RDMA pool.
#   - Benchmark context lengths: 16k, 32k, 64k, 128k
#   - Benchmark max-concurrency: 8,16,32,48,64,80,96
#   - For each context length, stop running higher concurrencies when throughput
#     gain is negligible ("almost no improvement").
#
# Notes:
#   - Radix cache remains enabled by default (we do NOT pass --disable-radix-cache).
#   - Per case: start server -> /health -> e2e_bench_fast -> stop server.
#   - Resume supported: existing JSON under RESULTS_DIR is reused unless
#     FORCE_RERUN=1.
#
# Usage:
#   export MODEL_PATH=/path/to/DeepSeek-V3.2-AWQ
#   bash test/cxl_utils/run_deepseek_v32_gpu_concurrency_autostop.sh
#
# Optional env:
#   MODEL_CLIENT, PORT, TP, DP_SIZE, MEM_FRACTION_STATIC, MAX_TOTAL_TOKENS
#   NUM_REQUESTS_MULT (num-requests = max_concurrency x this, default 8)
#   NUM_UNIQUE_PROMPTS, OUTPUT_TOKENS
#   REQUEST_RATE, REPEAT_MODE, PROMPT_MODE, SEED, PAUSE_BETWEEN_ROUNDS
#   ROUND2_OUTPUT_MIN, ROUND2_OUTPUT_MAX (defaults fixed 1k)
#   EARLY_STOP_ENABLE (default 1)
#   EARLY_STOP_MIN_IMPROVEMENT (default 0.05, i.e. <5% gain => stop)
#   EARLY_STOP_CHECK_FROM_CONC (default 32)
#   HF_ENDPOINT, RESULTS_DIR, SERVER_STARTUP_WAIT, SKIP_SERVER
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

NUM_REQUESTS_MULT="${NUM_REQUESTS_MULT:-8}"
NUM_UNIQUE_PROMPTS="${NUM_UNIQUE_PROMPTS:-1}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1024}"
REQUEST_RATE="${REQUEST_RATE:-0}"
REPEAT_MODE="${REPEAT_MODE:-interleaved}"
PROMPT_MODE="${PROMPT_MODE:-synthetic}"
SEED="${SEED:-42}"
PAUSE_BETWEEN_ROUNDS="${PAUSE_BETWEEN_ROUNDS:-3.0}"

# Fixed 1k Round-2 decode by default (same style as phased req512 scripts).
ROUND2_OUTPUT_MIN="${ROUND2_OUTPUT_MIN:-1024}"
ROUND2_OUTPUT_MAX="${ROUND2_OUTPUT_MAX:-1024}"

EARLY_STOP_ENABLE="${EARLY_STOP_ENABLE:-1}"
EARLY_STOP_MIN_IMPROVEMENT="${EARLY_STOP_MIN_IMPROVEMENT:-0.05}"
EARLY_STOP_CHECK_FROM_CONC="${EARLY_STOP_CHECK_FROM_CONC:-32}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-600}"
SKIP_SERVER="${SKIP_SERVER:-0}"

SGLANG_ENV_SCRIPT="${SGLANG_ENV_SCRIPT:-${HOME}/env.sh}"
PYTHON_BIN="${PYTHON_BIN:-/data2/ljr/a/downloads/miniconda3/envs/sgl/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_FAST="${E2E_FAST:-${SCRIPT_DIR}/e2e_bench_fast.py}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results_deepseek_v32_gpu_conc_autostop_${TIMESTAMP}}"

CONTEXT_LENGTHS=(16384 32768 65536 131072)
CONCURRENCIES=(8 16 32 48 64 80 96)
MAX_CASES=$(( ${#CONTEXT_LENGTHS[@]} * ${#CONCURRENCIES[@]} ))

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
    --num-requests-mult) NUM_REQUESTS_MULT="$2"; shift 2 ;;
    --num-unique-prompts) NUM_UNIQUE_PROMPTS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --round2-output-min) ROUND2_OUTPUT_MIN="$2"; shift 2 ;;
    --round2-output-max) ROUND2_OUTPUT_MAX="$2"; shift 2 ;;
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

mkdir -p "${RESULTS_DIR}"
META_FILE="${RESULTS_DIR}/run_meta.txt"
EARLY_STOP_FILE="${RESULTS_DIR}/early_stop_notes.txt"

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

start_server_gpu() {
  local log_file="$1"
  echo "  Launching server (full GPU, no HiSparse) -> ${log_file##*/}"
  # shellcheck disable=SC2086
  SGLANG_ENABLE_JIT_DEEPGEMM=0 "${PYTHON_BIN}" -m sglang.launch_server \
    --model "${MODEL_PATH}" \
    --tp "${TP}" \
    --dp-size "${DP_SIZE}" \
    --enable-dp-attention \
    --trust-remote-code \
    --dtype bfloat16 \
    --allow-auto-truncate \
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

extract_r2_output_throughput() {
  local json_file="$1"
  "${PYTHON_BIN}" - "${json_file}" <<'PY'
import json, math, sys
path = sys.argv[1]
try:
    with open(path) as f:
        d = json.load(f)
    v = d.get("round2", {}).get("metrics", {}).get("output_throughput_tok_s", float("nan"))
    v = float(v)
    if math.isfinite(v):
        print(f"{v:.6f}")
    else:
        print("nan")
except Exception:
    print("nan")
PY
}

calc_improvement_ratio() {
  local prev="$1"
  local curr="$2"
  "${PYTHON_BIN}" - "${prev}" "${curr}" <<'PY'
import math, sys
prev = float(sys.argv[1])
curr = float(sys.argv[2])
if (not math.isfinite(prev)) or prev <= 0.0 or (not math.isfinite(curr)):
    print("inf")
else:
    print(f"{(curr - prev) / prev:.6f}")
PY
}

{
  echo "run_deepseek_v32_gpu_concurrency_autostop.sh"
  echo "timestamp=${TIMESTAMP}"
  echo "MODEL_PATH=${MODEL_PATH} MODEL_CLIENT=${MODEL_CLIENT}"
  echo "TP=${TP} DP_SIZE=${DP_SIZE} PORT=${PORT}"
  echo "CONTEXT_LENGTHS=${CONTEXT_LENGTHS[*]}"
  echo "CONCURRENCIES=${CONCURRENCIES[*]}"
  echo "NUM_REQUESTS_MULT=${NUM_REQUESTS_MULT}"
  echo "ROUND2_OUTPUT_MIN=${ROUND2_OUTPUT_MIN} ROUND2_OUTPUT_MAX=${ROUND2_OUTPUT_MAX}"
  echo "EARLY_STOP_ENABLE=${EARLY_STOP_ENABLE}"
  echo "EARLY_STOP_MIN_IMPROVEMENT=${EARLY_STOP_MIN_IMPROVEMENT}"
  echo "EARLY_STOP_CHECK_FROM_CONC=${EARLY_STOP_CHECK_FROM_CONC}"
  echo "MAX_CASES=${MAX_CASES}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "NOTE=Radix cache is enabled by default (no --disable-radix-cache)."
} > "${META_FILE}"

: > "${EARLY_STOP_FILE}"

echo "============================================================"
echo "DeepSeek-V3.2 full-GPU context × concurrency sweep -> ${RESULTS_DIR}/"
echo "  No HiSparse. Radix cache uses default (enabled)."
echo "  Max possible cases: ${MAX_CASES}"
echo "============================================================"

CASE_IDX=0
RAN_CASES=0

for tin in "${CONTEXT_LENGTHS[@]}"; do
  tag="$(ctx_tag "${tin}")"
  prev_tput="nan"
  prev_conc=0

  echo ""
  echo "######################## Context ${tag} ########################"
  for conc in "${CONCURRENCIES[@]}"; do
    CASE_IDX=$((CASE_IDX + 1))
    nreq=$(( conc * NUM_REQUESTS_MULT ))
    base="gpu_full_v32_ctx${tag}_conc${conc}_r2_${ROUND2_OUTPUT_MIN}_${ROUND2_OUTPUT_MAX}_n${nreq}"
    out_file="${RESULTS_DIR}/${base}.json"
    srv_log="${RESULTS_DIR}/server_${base}.log"

    if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${out_file}" ]]; then
      echo ""
      echo "################################################################"
      echo "  SKIP ${CASE_IDX}/${MAX_CASES}: existing ${out_file##*/}"
      echo "################################################################"
    else
      echo ""
      echo "################################################################"
      echo "  CASE ${CASE_IDX}/${MAX_CASES}: ctx=${tin} (${tag}) conc=${conc} num-req=${nreq} (=conc×${NUM_REQUESTS_MULT})"
      echo "################################################################"

      if [[ "${SKIP_SERVER}" == "1" ]]; then
        echo "  SKIP_SERVER=1 -> client only"
      else
        kill_server 2>/dev/null || true
        start_server_gpu "${srv_log}"
        if ! wait_for_server; then
          echo "  ERROR: server failed to start, skip this case"
          kill_server
          continue
        fi
      fi

      echo "  --- client -> ${out_file##*/} ---"
      run_e2e_fast "${out_file}" "${tin}" "${nreq}" "${conc}"
      RAN_CASES=$((RAN_CASES + 1))

      if [[ "${SKIP_SERVER}" != "1" ]]; then
        kill_server
      fi
    fi

    curr_tput="$(extract_r2_output_throughput "${out_file}")"
    echo "  Round2 output throughput: ${curr_tput} tok/s"

    if [[ "${EARLY_STOP_ENABLE}" == "1" ]] && [[ "${prev_conc}" -gt 0 ]] && [[ "${conc}" -ge "${EARLY_STOP_CHECK_FROM_CONC}" ]]; then
      improve_ratio="$(calc_improvement_ratio "${prev_tput}" "${curr_tput}")"
      echo "  Throughput gain vs conc=${prev_conc}: ${improve_ratio}"
      if awk -v imp="${improve_ratio}" -v thr="${EARLY_STOP_MIN_IMPROVEMENT}" 'BEGIN { exit !(imp < thr) }'; then
        note="ctx=${tag}: stop after conc=${conc}; gain_vs_${prev_conc}=${improve_ratio} < threshold=${EARLY_STOP_MIN_IMPROVEMENT}"
        echo "  EARLY STOP: ${note}"
        echo "${note}" | tee -a "${EARLY_STOP_FILE}"
        break
      fi
    fi

    prev_tput="${curr_tput}"
    prev_conc="${conc}"
  done
done

echo ""
echo "Done. Results under: ${RESULTS_DIR}/"
echo "Executed new client runs: ${RAN_CASES}"
if [[ -s "${EARLY_STOP_FILE}" ]]; then
  echo "Early-stop notes:"
  cat "${EARLY_STOP_FILE}"
else
  echo "Early-stop notes: (none)"
fi

export RESULTS_DIR
"${PYTHON_BIN}" << 'PY'
import glob, json, os, re, sys

results_dir = os.environ.get("RESULTS_DIR", "")
if not results_dir or not os.path.isdir(results_dir):
    sys.exit(0)

pat = re.compile(r"gpu_full_v32_ctx([0-9]+k)_conc([0-9]+)_")
rows = []
for p in glob.glob(os.path.join(results_dir, "*.json")):
    bn = os.path.basename(p)
    m = pat.search(bn)
    if not m:
        continue
    ctx, conc = m.group(1), int(m.group(2))
    try:
        with open(p) as f:
            d = json.load(f)
        r2 = d.get("round2", {}).get("metrics", {})
        rows.append(
            (
                ctx,
                conc,
                float(r2.get("output_throughput_tok_s", float("nan"))),
                float(r2.get("request_throughput_req_s", float("nan"))),
                float(r2.get("ttft_mean_ms", float("nan"))),
                float(r2.get("tbt_mean_ms", float("nan"))),
                bn,
            )
        )
    except Exception:
        rows.append((ctx, conc, float("nan"), float("nan"), float("nan"), float("nan"), bn))

def ctx_key(ctx: str):
    try:
        return int(ctx[:-1])
    except Exception:
        return 10**9

rows.sort(key=lambda x: (ctx_key(x[0]), x[1]))

print("\nSummary (Round2):\n")
print(f"{'ctx':>6} {'conc':>5} {'tok/s':>12} {'req/s':>10} {'TTFT(ms)':>10} {'TBT(ms)':>10}  file")
print("-" * 100)
for ctx, conc, tok_s, req_s, ttft, tbt, bn in rows:
    print(f"{ctx:>6} {conc:>5} {tok_s:>12.1f} {req_s:>10.2f} {ttft:>10.1f} {tbt:>10.1f}  {bn}")
PY
