#!/usr/bin/env bash
set -euo pipefail

cxx=${CXX:-g++}

$cxx -std=c++17 -O3 -Wall -Wextra -march=native -mclflushopt read_latency.cpp -o read_latency.exe
$cxx -std=c++17 -O3 -Wall -Wextra -march=native -mclflushopt write_latency.cpp -o write_latency.exe

echo "Built ./read_latency.exe and ./write_latency.exe"

# Example runs (uncomment to execute after build):
./read_latency.exe --dev=/dev/dax0.0 --read-bytes=16384 --iterations=1000
./write_latency.exe --dev=/dev/dax0.0 --write-bytes=16384 --iterations=1000
