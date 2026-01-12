g++ -O2 -std=c++17 -fopenmp kvcache_latency_cxl2dram.cpp -o kvcache_latency_cxl2dram.exe
./kvcache_latency_cxl2dram.exe --dax /dev/dax0.0 --token-total 2048 --tokens 16 \
--layers 64 --layers-total 64 --heads 8 --head-dim 128 --token-offset 0 \
--layer-offset 0 --warmup 3 --iterations 20 -v