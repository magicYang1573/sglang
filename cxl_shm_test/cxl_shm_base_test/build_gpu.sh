nvcc -O3 -Xcompiler -mclflushopt write_latency_gpu.cpp -o write_latency_gpu.exe -lcudart
nvcc -O3 -Xcompiler -mclflushopt read_latency_gpu.cpp -o read_latency_gpu.exe -lcudart
./write_latency_gpu.exe --dev=/dev/dax0.0 --write-bytes=16384 --iterations=1000 --gpu-id=0
./read_latency_gpu.exe --dev=/dev/dax0.0 --read-bytes=16384 --iterations=1000 --gpu-id=0