#!/bin/bash
# =============================================================================
# CXL vs RDMA Prefix Cache — Full Experiment Runner
# =============================================================================
#
# Runs all four scheme configurations (A: DRAM, B-0: RDMA 0%, B-30: RDMA 30%,
# C: CXL) collecting TTFT/TBT/throughput.
#
# Both benchmark rounds replay the EXACT SAME N requests sampled from the
# ShareGPT dataset. Each of the N requests has unique content (different
# conversation). Round 1 populates the prefix cache; Round 2 hits it.
#
# Usage:
#   bash run_all_experiments.sh [--model MODEL] [--tp TP] [--port PORT] ...
#
# Requires:
#   - SGLang installed with HiSparse support
#   - For scheme B: RDMA extension built (make rdma in test/cxl_utils/)
#   - For scheme C: CXL DAX device available
# =============================================================================

set -euo pipefail

# Defaults
MODEL="${MODEL:-deepseek-ai/DeepSeek-V3.2}"
TP="${TP:-8}"
PORT="${PORT:-30000}"
NUM_REQUESTS="${NUM_REQUESTS:-16}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-64}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
REQUEST_RATE="${REQUEST_RATE:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-0}"
DATASET_PATH="${DATASET_PATH:-}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-180}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_BENCH="${SCRIPT_DIR}/e2e_bench.py"

# Parse command line overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
        --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
        --min-prompt-tokens) MIN_PROMPT_TOKENS="$2"; shift 2 ;;
        --max-prompt-tokens) MAX_PROMPT_TOKENS="$2"; shift 2 ;;
        --request-rate) REQUEST_RATE="$2"; shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Scheme definitions
SCHEMES=("dram" "rdma_0" "rdma_30" "cxl")
declare -A CONFIGS
CONFIGS[dram]='{"top_k":2048,"device_buffer_size":4096}'
CONFIGS[rdma_0]='{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"mlx5_0","local_ratio":0.0}}'
CONFIGS[rdma_30]='{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"mlx5_0","local_ratio":0.3}}'
CONFIGS[cxl]='{"top_k":2048,"device_buffer_size":4096,"cxl":{"enabled":true,"dev_path":"/dev/dax0.0","map_bytes":68719476736}}'

mkdir -p "${RESULTS_DIR}"

wait_for_server() {
    local url="http://localhost:${PORT}/health"
    local max_wait=$SERVER_STARTUP_WAIT
    local waited=0
    echo "  Waiting for server at ${url} ..."
    while ! curl -s "${url}" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            echo "  ERROR: Server did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "  Server ready (waited ${waited}s)"
}

kill_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "  Stopping server (PID=${SERVER_PID})..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        unset SERVER_PID
        sleep 5
    fi
}

trap kill_server EXIT

echo "============================================================"
echo "CXL vs RDMA Experiment Runner"
echo "============================================================"
echo "Model:            ${MODEL}"
echo "TP:               ${TP}"
echo "Port:             ${PORT}"
echo "Requests/round:   ${NUM_REQUESTS}  (same requests in both rounds)"
echo "Output tokens:    ${OUTPUT_TOKENS}  (0 = dataset ground truth)"
echo "Prompt range:     ${MIN_PROMPT_TOKENS} - ${MAX_PROMPT_TOKENS} tokens"
echo "Request rate:     ${REQUEST_RATE} req/s"
echo "Results dir:      ${RESULTS_DIR}"
echo "============================================================"

# Build optional dataset-path arg
DATASET_ARG=""
if [ -n "${DATASET_PATH}" ]; then
    DATASET_ARG="--dataset-path ${DATASET_PATH}"
fi

for scheme in "${SCHEMES[@]}"; do
    config="${CONFIGS[$scheme]}"
    echo ""
    echo "############################################################"
    echo "  SCHEME: ${scheme}"
    echo "  Config: ${config}"
    echo "############################################################"

    output_file="${RESULTS_DIR}/${scheme}.json"

    # Start server
    echo "  Starting server..."
    python -m sglang.launch_server \
        --model "${MODEL}" \
        --tp "${TP}" \
        --enable-hisparse \
        --hisparse-config "${config}" \
        --port "${PORT}" \
        > "${RESULTS_DIR}/${scheme}_server.log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "  Skipping scheme ${scheme} (server failed to start)"
        kill_server
        continue
    fi

    echo "  Running two-round benchmark (both rounds = same ${NUM_REQUESTS} requests)..."
    # shellcheck disable=SC2086
    python "${E2E_BENCH}" \
        --server-url "http://localhost:${PORT}" \
        --model "${MODEL}" \
        --num-requests "${NUM_REQUESTS}" \
        --output-tokens "${OUTPUT_TOKENS}" \
        --min-prompt-tokens "${MIN_PROMPT_TOKENS}" \
        --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
        --request-rate "${REQUEST_RATE}" \
        --max-concurrency "${MAX_CONCURRENCY}" \
        ${DATASET_ARG} \
        --output "${output_file}" \
        2>&1 | tee "${RESULTS_DIR}/${scheme}.log"

    # Collect system metrics
    echo "  Collecting system metrics..."
    free -h > "${RESULTS_DIR}/${scheme}_mem.txt" 2>/dev/null || true
    nvidia-smi > "${RESULTS_DIR}/${scheme}_gpu.txt" 2>/dev/null || true

    kill_server
done

echo ""
echo "============================================================"
echo "All experiments complete. Results in ${RESULTS_DIR}/"
echo "============================================================"

# Print summary comparison table: TTFT and TBT for round1 vs round2
echo ""
echo "Summary: Round 1 → Round 2 improvements"
echo "--------------------------------------------------------------------"
printf "%-12s | %10s %10s %10s | %10s %10s\n" \
    "Scheme" "R1-TTFT" "R2-TTFT" "TTFT-x" "R1-TBT" "R2-TBT"
echo "--------------------------------------------------------------------"

for scheme in "${SCHEMES[@]}"; do
    f="${RESULTS_DIR}/${scheme}.json"
    if [ -f "$f" ]; then
        python3 - <<EOF
import json
d = json.load(open("$f"))
r1 = d["round1"]["metrics"]
r2 = d["round2"]["metrics"]
ttft_x = r1["ttft_mean_ms"] / r2["ttft_mean_ms"] if r2["ttft_mean_ms"] > 0 else float("nan")
print(f"{'${scheme}':<12s} | {r1['ttft_mean_ms']:>10.1f} {r2['ttft_mean_ms']:>10.1f} {ttft_x:>10.2f}x"
      f" | {r1['tbt_mean_ms']:>10.1f} {r2['tbt_mean_ms']:>10.1f}")
EOF
    else
        printf "%-12s | %10s %10s %10s | %10s %10s\n" \
            "${scheme}" "N/A" "N/A" "N/A" "N/A" "N/A"
    fi
done
