#!/bin/bash
# =============================================================================
# CXL vs RDMA Prefix Cache — Full Experiment Runner
# =============================================================================
#
# Runs five pool configurations:
#   dram, rdma_50 (50% local), rdma_25 (25% local), rdma_0 (0% local), cxl
#
# Default workload matches docs/plan/experiment_plan_deepseek_v32.md (long context):
#   --prompt-mode synthetic --target-input-tokens --output-tokens
#   --num-requests 512 --max-concurrency 32 --request-rate 0
#
# SWEEP=1 runs the full 4×3 input×output matrix (8K–64K × 1K–4K) per scheme.
#
# Client note: --num-requests is the number of HTTP calls per round; SGLang does not
# reduce that number. --max-concurrency only limits how many requests are in flight
# on the client; use 32 for long context to limit KV / DRAM pressure vs 128.
#
# Usage:
#   bash run_all_experiments.sh [--model MODEL] [--tp TP] [--port PORT] ...
#   SWEEP=1 bash run_all_experiments.sh
#
# Requires:
#   - SGLang with HiSparse
#   - RDMA schemes: extension built (make rdma in test/cxl_utils/)
#   - CXL scheme: DAX device available
# =============================================================================

set -euo pipefail

# Defaults (long-context e2e)
MODEL="${MODEL:-deepseek-ai/DeepSeek-V3.2}"
TP="${TP:-8}"
PORT="${PORT:-30000}"
NUM_REQUESTS="${NUM_REQUESTS:-512}"
TARGET_INPUT_TOKENS="${TARGET_INPUT_TOKENS:-8192}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1024}"
REQUEST_RATE="${REQUEST_RATE:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
PROMPT_MODE="${PROMPT_MODE:-synthetic}"
DATASET_PATH="${DATASET_PATH:-}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-180}"
# SWEEP=1: run all input×output pairs below for each scheme
SWEEP="${SWEEP:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_BENCH="${SCRIPT_DIR}/e2e_bench.py"

# Parse command line overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
        --target-input-tokens) TARGET_INPUT_TOKENS="$2"; shift 2 ;;
        --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
        --request-rate) REQUEST_RATE="$2"; shift 2 ;;
        --max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
        --prompt-mode) PROMPT_MODE="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Scheme definitions (local_ratio only applies to RDMA)
SCHEMES=(dram rdma_50 rdma_25 rdma_0 cxl)
declare -A CONFIGS
CONFIGS[dram]='{"top_k":2048,"device_buffer_size":4096}'
CONFIGS[rdma_50]='{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"mlx5_0","local_ratio":0.5}}'
CONFIGS[rdma_25]='{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"mlx5_0","local_ratio":0.25}}'
CONFIGS[rdma_0]='{"top_k":2048,"device_buffer_size":4096,"rdma_pool":{"enabled":true,"ib_dev":"mlx5_0","local_ratio":0.0}}'
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
echo "Model:              ${MODEL}"
echo "TP:                 ${TP}"
echo "Port:               ${PORT}"
echo "Prompt mode:        ${PROMPT_MODE}"
echo "Requests/round:     ${NUM_REQUESTS}  (same prompts in Round 1 & 2)"
echo "Target input tok:   ${TARGET_INPUT_TOKENS}  (synthetic)"
echo "Output tokens:      ${OUTPUT_TOKENS}"
echo "Max concurrency:    ${MAX_CONCURRENCY}"
echo "Request rate:       ${REQUEST_RATE}  (0 = no inter-request delay)"
echo "SWEEP:              ${SWEEP}  (1 = 8K–64K × 1K–4K matrix)"
echo "Results dir:        ${RESULTS_DIR}"
echo "============================================================"

DATASET_ARG=""
if [ -n "${DATASET_PATH}" ]; then
    DATASET_ARG="--dataset-path ${DATASET_PATH}"
fi

run_one_bench() {
    local out_path="$1"
    local tin="$2"
    local tout="$3"
    # shellcheck disable=SC2086
    python "${E2E_BENCH}" \
        --server-url "http://localhost:${PORT}" \
        --model "${MODEL}" \
        --prompt-mode "${PROMPT_MODE}" \
        --target-input-tokens "${tin}" \
        --num-requests "${NUM_REQUESTS}" \
        --output-tokens "${tout}" \
        --request-rate "${REQUEST_RATE}" \
        --max-concurrency "${MAX_CONCURRENCY}" \
        ${DATASET_ARG} \
        --output "${out_path}" \
        2>&1 | tee "${out_path%.json}.log"
}

for scheme in "${SCHEMES[@]}"; do
    config="${CONFIGS[$scheme]}"
    echo ""
    echo "############################################################"
    echo "  SCHEME: ${scheme}"
    echo "  Config: ${config}"
    echo "############################################################"

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

    if [ "${SWEEP}" = "1" ]; then
        echo "  SWEEP: running input×output matrix (12 runs)..."
        for tin in 8192 16384 32768 65536; do
            for tout in 1024 2048 4096; do
                tag="in${tin}_out${tout}"
                out_file="${RESULTS_DIR}/${scheme}_${tag}.json"
                echo "  --- ${tag} ---"
                run_one_bench "${out_file}" "${tin}" "${tout}"
            done
        done
    else
        out_file="${RESULTS_DIR}/${scheme}.json"
        echo "  Single configuration: in=${TARGET_INPUT_TOKENS} out=${OUTPUT_TOKENS}"
        run_one_bench "${out_file}" "${TARGET_INPUT_TOKENS}" "${OUTPUT_TOKENS}"
    fi

    echo "  Collecting system metrics..."
    free -h > "${RESULTS_DIR}/${scheme}_mem.txt" 2>/dev/null || true
    nvidia-smi > "${RESULTS_DIR}/${scheme}_gpu.txt" 2>/dev/null || true

    kill_server
done

echo ""
echo "============================================================"
echo "All experiments complete. Results in ${RESULTS_DIR}/"
echo "============================================================"

echo ""
echo "Summary: Round 1 → Round 2 (TTFT ratio, TBT)"
echo "--------------------------------------------------------------------"
printf "%-28s | %10s %10s %10s | %10s %10s\n" \
    "Run" "R1-TTFT" "R2-TTFT" "TTFT-x" "R1-TBT" "R2-TBT"
echo "--------------------------------------------------------------------"

if [ "${SWEEP}" = "1" ]; then
    for scheme in "${SCHEMES[@]}"; do
        for tin in 8192 16384 32768 65536; do
            for tout in 1024 2048 4096; do
                tag="in${tin}_out${tout}"
                f="${RESULTS_DIR}/${scheme}_${tag}.json"
                if [ -f "$f" ]; then
                    python3 - <<EOF
import json
label = "${scheme}_${tag}"
d = json.load(open("$f"))
r1 = d["round1"]["metrics"]
r2 = d["round2"]["metrics"]
ttft_x = r1["ttft_mean_ms"] / r2["ttft_mean_ms"] if r2["ttft_mean_ms"] > 0 else float("nan")
print(f"{label:<28s} | {r1['ttft_mean_ms']:>10.1f} {r2['ttft_mean_ms']:>10.1f} {ttft_x:>10.2f}x"
      f" | {r1['tbt_mean_ms']:>10.1f} {r2['tbt_mean_ms']:>10.1f}")
EOF
                fi
            done
        done
    done
else
    for scheme in "${SCHEMES[@]}"; do
        f="${RESULTS_DIR}/${scheme}.json"
        if [ -f "$f" ]; then
            python3 - <<EOF
import json
label = "${scheme}"
d = json.load(open("$f"))
r1 = d["round1"]["metrics"]
r2 = d["round2"]["metrics"]
ttft_x = r1["ttft_mean_ms"] / r2["ttft_mean_ms"] if r2["ttft_mean_ms"] > 0 else float("nan")
print(f"{label:<28s} | {r1['ttft_mean_ms']:>10.1f} {r2['ttft_mean_ms']:>10.1f} {ttft_x:>10.2f}x"
      f" | {r1['tbt_mean_ms']:>10.1f} {r2['tbt_mean_ms']:>10.1f}")
EOF
        fi
    done
fi
