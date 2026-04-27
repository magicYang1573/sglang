#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_ROOT="${RUN_ROOT:-/data2/ljr/a/sglang/exp-topk}"
MODEL_PATH="${MODEL_PATH:-/data3/yt/models/DeepSeek-V3.2-AWQ}"
PYTHON_BIN="${PYTHON_BIN:-/data1/yt/miniconda3/envs/sgl/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-26666}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"

LENGTHS="${LENGTHS:-16384,32768,65536,131072}"
REQUESTS_PER_LENGTH="${REQUESTS_PER_LENGTH:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-1000}"
SEED="${SEED:-0}"

DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-${RUN_ROOT}/dataset_cache}"
run_dir="${RUN_ROOT}/deepseek_v32_$(date +%Y%m%d_%H%M%S)"
SERVER_PID=""

mkdir -p "${DATASET_CACHE_DIR}" "${run_dir}/topk_logs" "${run_dir}/analysis"

cleanup() {
  if [[ -n "${SERVER_PID}" && "${KEEP_SERVER:-0}" != "1" ]]; then
    echo "Stopping SGLang server pid=${SERVER_PID}"
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  local url="${1}"
  local timeout="${2:-1800}"
  "${PYTHON_BIN}" - "$url" "$timeout" <<'PY'
import sys
import time
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
timeout = int(sys.argv[2])
start = time.time()
while time.time() - start < timeout:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if 200 <= resp.status < 300:
                print(f"Server ready: {url}")
                raise SystemExit(0)
    except Exception:
        time.sleep(5)
print(f"Timed out waiting for {url}", file=sys.stderr)
raise SystemExit(1)
PY
}

gen_extra=()
if [[ "${FORCE_REGENERATE_DATASET:-0}" == "1" ]]; then
  gen_extra+=(--force)
fi

echo "=== dataset step (before server) $(date) ==="
echo "Dataset cache: ${DATASET_CACHE_DIR}"
echo "This run logs: ${run_dir}"
{
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/nsa_topk_kv_coverage/generate_mixed_sharegpt_dataset.py \
    --model "${MODEL_PATH}" \
    --output-dir "${DATASET_CACHE_DIR}" \
    --lengths "${LENGTHS}" \
    --requests-per-length "${REQUESTS_PER_LENGTH}" \
    --output-len "${OUTPUT_LEN}" \
    --seed "${SEED}" \
    --sharegpt-path /data3/yt/sharegpt.json \
    --skip-if-params-match \
    "${gen_extra[@]}"
} 2>&1 | tee "${run_dir}/generate_dataset.log"

echo "Run directory: ${run_dir}"
echo "Top-k logs:    ${run_dir}/topk_logs"

if [[ "${START_SERVER:-1}" == "1" ]]; then
  echo "Starting SGLang DeepSeek-V3.2 server on ${BASE_URL}"
  SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  SGLANG_NSA_TOPK_LOG_DIR="${run_dir}/topk_logs" \
  SGLANG_NSA_TOPK_LOG_MODE=decode \
  SGLANG_NSA_TOPK_LOG_EVERY_DECODE=1 \
  "${PYTHON_BIN}" -m sglang.launch_server \
    --model "${MODEL_PATH}" \
    --tp 8 \
    --dp-size 8 \
    --enable-dp-attention \
    --trust-remote-code \
    --dtype bfloat16 \
    --disable-cuda-graph \
    --allow-auto-truncate \
    --max-total-tokens 500000 \
    --mem-fraction-static 0.85 \
    --max-running-requests 512 \
    --port "${PORT}" \
    >"${run_dir}/server.log" 2>&1 &
  SERVER_PID="$!"
  echo "${SERVER_PID}" >"${run_dir}/server.pid"
  wait_for_server "${BASE_URL}" 3600
else
  echo "START_SERVER=0: use an already-running server at ${BASE_URL}; logs must go to ${run_dir}/topk_logs"
fi

echo "Sending mixed-length requests"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/nsa_topk_kv_coverage/run_mixed_length_requests.py \
  --base-url "${BASE_URL}" \
  --model "${MODEL_PATH}" \
  --dataset-jsonl "${DATASET_CACHE_DIR}/dataset.jsonl" \
  --output-jsonl "${run_dir}/request_outputs.jsonl" \
  --concurrency "${CONCURRENCY:-64}" \
  2>&1 | tee "${run_dir}/run_requests.log"

echo "Analyzing decode-time KV coverage"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/nsa_topk_kv_coverage/analyze_decode_kv_coverage.py \
  --log-dir "${run_dir}/topk_logs" \
  --manifest "${DATASET_CACHE_DIR}/manifest.json" \
  --output-dir "${run_dir}/analysis" \
  2>&1 | tee "${run_dir}/analysis.log"

echo
echo "Experiment complete."
echo "Run directory: ${run_dir}"
echo "Dataset cache (reused when params match): ${DATASET_CACHE_DIR}"
echo "Summary (all lengths):"
echo "  ${run_dir}/analysis/summary_by_length.csv"
echo "  ${run_dir}/analysis/summary_by_length_layer.csv"
echo "Per-length slices:"
echo "  ${run_dir}/analysis/by_length/len_16k/  len_32k/  len_64k/  len_128k/"
