# 单层离散 Top-K KV Cache 读取延迟 Micro-Benchmark

**对比 DRAM / CXL / RDMA 三种内存后端在 DeepSeek-V3.2 NSA 稀疏访问模式下的 GPU 离散读取延迟**

---

## 1. 实验目标

在脱离 SGLang serving 栈的 micro-benchmark 层面，精确测量从 DRAM、CXL、RDMA 三种内存后端中，按照 DeepSeek-V3.2 NSA 的 **单层离散 top-k** 访问模式，将选中的 KV Cache 条目读取到 GPU VRAM 的延迟。

三种后端均测量的是 **相同的离散 top-k 读取操作**——GPU 按随机 indices 从远端内存中 scatter read 指定 token 的 KV 数据。区别仅在于数据源介质：

| 后端 | 数据源 | GPU 读取方式 |
|------|-------|-------------|
| **DRAM** | 本地 CPU pinned memory | GPU kernel scatter read（经 PCIe DMA） |
| **CXL** | CXL type-3 设备 mmap 区域 | GPU kernel scatter read（经 CXL 链路） |
| **RDMA** | 远端节点 DRAM | RDMA READ → 本地 staging → GPU kernel scatter read |

核心目标是验证：**在离散细粒度访问模式下，CXL 的 load/store 语义显著优于 RDMA 的消息语义**。

---

## 2. DeepSeek-V3.2 KV Cache 条目大小

### 2.1 SGLang 中的实际实现

在 SGLang 的 HiSparse 实现中，每个 token 的 KV Cache 条目大小由以下链路确定：

1. **Device pool** (`MLATokenToKVPool`) 的 `kv_buffer` shape 为 `(num_tokens, 1, kv_cache_dim)`，dtype 为 `store_dtype`
2. **Host pool** (`MLATokenToKVPoolHost`) 继承 device pool 的 `kv_cache_dim` 和 `store_dtype`
3. **JIT kernel** 接收的 `item_size_bytes` = `token_stride_size` = `kv_cache_dim × store_dtype.itemsize`

MLA 的 KV Cache 将 K 和 V 压缩到同一个 latent vector 中（不像 MHA 有独立的 K/V buffer），因此每 token 只存储一个向量。

### 2.2 具体数值计算

DeepSeek-V3.2 模型参数（来自 [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json)）：

```
kv_lora_rank = 512
qk_rope_head_dim = 64
```

SGLang 中对 NSA 模型的 `kv_cache_dim` 计算（`model_runner_kv_cache_mixin.py:calculate_mla_kv_cache_dim`）：

- **BF16 模式**：`kv_cache_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576`
  - `store_dtype = torch.bfloat16`（`itemsize = 2`）
  - **`item_size_bytes = 576 × 2 = 1152 bytes`**

- **FP8 packed 模式**（NSA 默认在高端 GPU 上）：`kv_cache_dim = kv_lora_rank + kv_lora_rank//128*4 + qk_rope_head_dim*2 = 512 + 16 + 128 = 656`
  - `store_dtype = torch.uint8`（FP8 以 uint8 存储）
  - **`item_size_bytes = 656 × 1 = 656 bytes`**

### 2.3 本实验选用

选取 **FP8 packed 模式**，因为它是 DeepSeek-V3.2 NSA + HiSparse 在 H20/H100 上的默认运行配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| **`item_size_bytes`** | **656 bytes** | 每 token KV 条目大小 |
| `kv_cache_dim` | 656 | FP8 nope + scale + BF16 rope packed |
| `store_dtype` | `torch.uint8` | FP8 以 uint8 存储 |
| `num_layers` | 61 | 但本实验仅测单层 |
| `index_topk` | 2048 | NSA 默认 top-k（本实验测 64-2048） |

---

## 3. 实验设计

### 3.1 通用设置

所有三种后端共享相同的 GPU 侧操作——scatter read kernel 按照离散 indices 从 host pointer 读取 top-k 个 token 的 KV 数据到 GPU VRAM 连续 buffer。这与 HiSparse 实际运行时 `hisparse.cuh` 中 `transfer_item_warp` 的行为一致。

```
host/remote memory                          GPU VRAM
┌───────────────────┐                      ┌──────────────┐
│ token 0:  656B    │                      │              │
│ token 1:  656B    │  scatter read        │ top-k tokens │
│ ...               │  (by random indices) │ contiguous   │
│ token N-1: 656B   │ ──────────────────→  │ 656B × top_k │
│ (N = seq_len)     │                      │              │
└───────────────────┘                      └──────────────┘
```

**GPU Scatter Read Kernel**（模拟 `hisparse.cuh` `transfer_item_warp`）：

```cuda
// grid=(top_k,), block=(256,) — 每个 block (8 warps) 处理一个 token
__global__ void scatter_read_kernel(
    const uint8_t* __restrict__ src,        // host KV pool base (pinned/CXL/staging)
    uint8_t* __restrict__ dst,              // GPU output buffer (contiguous)
    const int64_t* __restrict__ indices,    // top-k token positions
    int64_t item_size_bytes)
{
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const uint64_t* src64 = (const uint64_t*)(src + indices[token_idx] * item_size_bytes);
    uint64_t* dst64 = (uint64_t*)(dst + token_idx * item_size_bytes);
    const int total_u64 = item_size_bytes / sizeof(uint64_t);  // 656/8 = 82

    for (int j = tid; j < total_u64; j += blockDim.x) {
        uint64_t tmp;
        asm volatile("ld.global.nc.b64 %0,[%1];" : "=l"(tmp) : "l"(src64 + j));
        asm volatile("st.global.cg.b64 [%0],%1;" :: "l"(dst64 + j), "l"(tmp));
    }
}
```

### 3.2 实验 A：DRAM → GPU（基线）

- 分配 `seq_len × 656 bytes` 的 CPU pinned buffer（`cudaHostRegister`）
- 随机填充数据
- 生成随机 top-k indices
- 启动 scatter_read_kernel，由 GPU 直接从 pinned DRAM 离散读取
- CUDA event 计时

### 3.3 实验 B：CXL → GPU

- `cxl_init(dev_path, map_bytes, register_cuda=True, gpu_id=0)` 映射 CXL 设备并注册为 CUDA host memory
- 将 KV 数据写入 CXL mmap region
- 使用与 DRAM 完全相同的 scatter_read_kernel（GPU 通过注册后的 host pointer 经 CXL 链路读取）
- CUDA event 计时

### 3.4 实验 C：RDMA 离散读取 → GPU

**RDMA 路径模拟的是 per-layer 按需获取 top-k KV 的场景**——远端节点持有全量 KV，本地通过 RDMA 仅读取 top-k 对应的离散条目，然后 GPU 从本地 staging buffer scatter read。

**数据路径**：
```
本机 NIC loopback DRAM ──(N 次 RDMA READ, 每次 656B)──→ 本地 pinned staging ──(GPU scatter read)──→ GPU VRAM
```

**为什么不做全量预取**：本实验的目的是公平对比三种后端做 **相同的离散 top-k 读取** 的延迟。RDMA 路径也只读取 top-k 条目，以体现 RDMA 消息语义在细粒度离散访问下的固有劣势（per-message μs 级开销 × top-k 次）。

**实现（本机 NIC loopback）**：
1. 使用 `ibv_*` libibverbs API，在本机 RDMA 网卡上建立 RC QP loopback 连接（QP 连接到自身）
2. 分配两块独立内存区域并注册为 MR：
   - **"远端" MR**：存储全量单层 KV（模拟远端节点持有的 KV pool）
   - **本地 staging MR**：接收 RDMA READ 结果（`top_k × 656B`，pinned memory）
3. 发起 **top-k 次 RDMA READ WR**，每次从 "远端" MR 读取一个 token 的 656 bytes 到 staging MR 的对应位置
   - 批量 `ibv_post_send` 提交所有 WR，仅最后一个标记 `IBV_SEND_SIGNALED`
4. 轮询 CQ 等待所有 RDMA READ 完成
5. 启动 GPU scatter read kernel 从本地 staging 连续读取到 GPU
6. 计时 = RDMA 读取时间 + GPU scatter read 时间

### 3.5 关于本机 NIC Loopback 的说明

本实验使用 **本机 RDMA 网卡 loopback**（RC QP 两端均在本机同一网卡上）。Loopback 模式下数据经过 NIC 硬件处理但不经过物理网络线缆，因此：

| 方面 | 本机 NIC Loopback | 跨机真实 RDMA |
|------|-------------------|-------------|
| per-message 开销 | ~1-2 μs | ~2-5 μs |
| 数据路径 | CPU→NIC→NIC DMA→CPU（PCIe 回环） | CPU→NIC→线缆→NIC→CPU |
| 带宽 | 受 PCIe 与 NIC 处理能力限制 | 受线速限制（100/200 Gbps） |
| 拥塞 | 无 | 可能有 |

**Loopback 的有效性**：
- **per-message 开销**是 RDMA 离散读取的主要瓶颈，loopback 下该开销已经包含了完整的 NIC 硬件处理流程（WQE 解析、DMA、CQE 生成），与跨机场景在量级上一致（仅少了线缆传播延迟 ~0.5-1 μs）
- 因此 loopback 结果可以有效体现 **RDMA 消息语义在离散细粒度访问下的结构性劣势**
- 跨机场景的延迟只会更高（per-message 开销 ~2-5× loopback），loopback 数据是 RDMA 性能的 **下界**

**即使是 loopback，RDMA 在离散读取场景的劣势也会显著体现**：以 top-k=2048 为例，per-message 开销累计 = 2048 × ~1-2 μs = **2-4 ms**（loopback），远超 CXL/DRAM 的整体 scatter read 延迟（~50-100 μs）。

**实验报告中将注明**："RDMA 数据为本机 NIC loopback 模式，跨机场景延迟预计更高。"

---

## 4. 变量矩阵

| 变量 | 取值 | 说明 |
|------|------|------|
| **内存后端** | DRAM, CXL, RDMA | 核心对比维度 |
| **top-k** | 64, 128, 256, 512, 1024, 2048 | 离散读取条目数 |
| **seq_len** | 32768 (32K), 65536 (64K), 131072 (128K) | KV pool 大小（影响访问离散度） |
| **item_size_bytes** | 656 | 固定，FP8 packed MLA |
| **batch_size** | 1 | 单请求（消除 batch 干扰） |

> `item_size_bytes = 656` 固定不变。这是 DeepSeek-V3.2 FP8 NSA 模式下的真实 per-token KV 大小。

---

## 5. 数据量速查表

| seq_len | top-k | top-k 数据量 | sparse ratio | RDMA 消息数 |
|---------|-------|-------------|-------------|------------|
| 32K | 64 | 41 KB | 0.20% | 64 |
| 32K | 128 | 82 KB | 0.39% | 128 |
| 32K | 256 | 164 KB | 0.78% | 256 |
| 32K | 512 | 328 KB | 1.56% | 512 |
| 32K | 1024 | 656 KB | 3.13% | 1024 |
| 32K | 2048 | 1.31 MB | 6.25% | 2048 |
| 64K | 64 | 41 KB | 0.10% | 64 |
| 64K | 256 | 164 KB | 0.39% | 256 |
| 64K | 1024 | 656 KB | 1.56% | 1024 |
| 64K | 2048 | 1.31 MB | 3.13% | 2048 |
| 128K | 64 | 41 KB | 0.05% | 64 |
| 128K | 128 | 82 KB | 0.10% | 128 |
| 128K | 256 | 164 KB | 0.20% | 256 |
| 128K | 512 | 328 KB | 0.39% | 512 |
| 128K | 1024 | 656 KB | 0.78% | 1024 |
| 128K | 2048 | 1.31 MB | 1.56% | 2048 |

---

## 6. 测量方法

### 6.1 计时方式

- **DRAM / CXL**：CUDA Event timing（`cudaEventRecord` → `cudaEventSynchronize` → `cudaEventElapsedTime`）
- **RDMA**：
  - RDMA 阶段：`clock_gettime(CLOCK_MONOTONIC)` 包裹 `ibv_post_send` → `ibv_poll_cq` 完成
  - GPU 阶段：CUDA Event timing
  - 总延迟 = RDMA 阶段 + GPU 阶段

### 6.2 迭代次数

| 阶段 | 次数 |
|------|------|
| Warmup | 50 |
| Timed | 200 |

每次迭代重新生成随机 top-k indices，模拟真实 NSA 的 query-dependent 选择行为。

### 6.3 统计指标

| 指标 | 说明 |
|------|------|
| mean latency (μs) | 平均单层读取延迟 |
| std (μs) | 标准差 |
| p50 / p99 (μs) | 中位数和尾延迟 |
| effective BW (GB/s) | `top_k × item_size / mean_latency` |

---

## 7. 预期结果与分析

### 7.1 延迟估算（seq_len=128K, NIC loopback）

| 后端 | top-k=64 | top-k=256 | top-k=1024 | top-k=2048 |
|------|----------|-----------|------------|------------|
| **DRAM** | ~2-5 μs | ~5-15 μs | ~15-35 μs | ~25-50 μs |
| **CXL** | ~5-12 μs | ~12-30 μs | ~30-70 μs | ~50-100 μs |
| **RDMA** (loopback) | ~64-130 μs | ~256-510 μs | ~1000-2500 μs | ~2000-5000 μs |

> DRAM/CXL 延迟随 top-k 近似线性增长（数据量线性增加）；RDMA 延迟以 per-message 开销为主，与 top-k 严格线性。

### 7.2 RDMA 的结构性劣势

RDMA 的核心问题在于 **per-message 固定开销**：

```
RDMA 单层延迟 ≈ per_msg_overhead × top_k + data_transfer_time
            ≈ 1~2.5 μs × top_k + ~几十 μs (loopback)

示例 (top_k=2048, loopback):
    ≈ 1~2.5 μs × 2048 + ~50 μs
    ≈ 2000 ~ 5000 μs
```

相比之下，CXL 和 DRAM 通过单次 GPU kernel launch 完成所有 top-k 条目的读取，GPU 内部的 warp 级并行天然隐藏了离散访问延迟。

> 注：如果 RDMA 改为用 SG（Scatter-Gather）list 将多个离散读取合并为一个 WR，可以减少 per-msg 开销，但仍受限于单 WR 的 SG 条目数上限（通常 ~30），且远端内存的离散访问模式导致 NIC 硬件 gather 效率低。

### 7.3 预期图表

**图 1：单层延迟 vs top-k（seq_len=128K, NIC loopback）**

```
延迟 (μs, log scale)
    │
10⁴ │                                                ×──× RDMA
    │                                    ×───────────
    │                        ×───────────
10³ │            ×───────────
    │
    │
10² │        ×                                       ●──● CXL
    │                                    ●───────────
    │                        ●───────────
10¹ │    ●───                ○───────────○───────────○──○ DRAM
    │    ○───────○───────────
    │
    └────┬───────┬───────────┬───────────┬───────────┬──
         64     128         256         512        1024  2048
                              top-k
```

CXL 曲线紧跟 DRAM（~2-3× overhead），RDMA 即使在 loopback 下也高出 1-2 个数量级。

---

## 8. 实现方案

### 8.1 目录结构

```
test/cxl_utils/
├── sparse_kv_bench.py           # 主测试脚本
├── sparse_kv_kernel.cu          # GPU scatter read kernel (JIT compile)
├── rdma_scatter_read.c          # RDMA 离散读取 C 实现 (ibverbs)
├── rdma_scatter_read_py.cpp     # pybind11 wrapper
├── Makefile                     # 编译 RDMA 模块
└── README.md
```

### 8.2 DRAM 后端

```python
def init_dram_pool(seq_len, item_size):
    """Allocate pinned DRAM KV pool, return host tensor."""
    buf = torch.randint(0, 256, (seq_len, item_size), dtype=torch.uint8)
    torch.cuda.cudart().cudaHostRegister(buf.data_ptr(), buf.numel(), 0)
    return buf

def bench_dram(host_buf, device_buf, indices, item_size, warmup=50, iters=200):
    top_k = indices.shape[0]
    # warmup
    for _ in range(warmup):
        launch_scatter_kernel(host_buf.data_ptr(), device_buf.data_ptr(),
                              indices.data_ptr(), item_size, top_k)
        torch.cuda.synchronize()
    # timed
    latencies = []
    for _ in range(iters):
        indices = torch.randint(0, host_buf.shape[0], (top_k,),
                                dtype=torch.int64, device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch_scatter_kernel(host_buf.data_ptr(), device_buf.data_ptr(),
                              indices.data_ptr(), item_size, top_k)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000)  # ms → μs
    return latencies
```

### 8.3 CXL 后端

```python
def init_cxl_pool(seq_len, item_size, cxl_ext, dev_path="/dev/dax0.0", gpu_id=0):
    """Map CXL device, write KV data, return base offset."""
    total_bytes = seq_len * item_size
    ok = cxl_ext.cxl_init(dev_path=dev_path, map_bytes=total_bytes + 4096,
                           register_cuda=True, gpu_id=gpu_id)
    assert ok, "CXL init failed"

    # Write random data
    src = torch.randint(0, 256, (seq_len, item_size), dtype=torch.uint8)
    cxl_ext.tensor_to_cxl(src.view(-1), offset=0)
    # CXL base is now CUDA-registered; GPU kernel reads via registered pointer
    return cxl_ext.cxl_get_base_ptr(), total_bytes

def bench_cxl(cxl_base_ptr, device_buf, indices, item_size, warmup=50, iters=200):
    """Same kernel as DRAM — GPU scatter read from CXL registered pointer."""
    top_k = indices.shape[0]
    for _ in range(warmup):
        launch_scatter_kernel(cxl_base_ptr, device_buf.data_ptr(),
                              indices.data_ptr(), item_size, top_k)
        torch.cuda.synchronize()
    latencies = []
    for _ in range(iters):
        indices = torch.randint(0, device_buf.shape[0], (top_k,),
                                dtype=torch.int64, device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch_scatter_kernel(cxl_base_ptr, device_buf.data_ptr(),
                              indices.data_ptr(), item_size, top_k)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000)
    return latencies
```

### 8.4 RDMA 后端

RDMA 离散读取的 C 层核心逻辑：

```c
// rdma_scatter_read.c
// 使用 libibverbs API 实现离散 RDMA READ

struct rdma_context {
    struct ibv_context *ctx;
    struct ibv_pd      *pd;
    struct ibv_cq      *cq;
    struct ibv_qp      *qp;
    struct ibv_mr      *local_mr;    // 本地 staging buffer
    struct ibv_mr      *remote_mr;   // 远端 KV pool (loopback 时也是本地)
    uint64_t            remote_addr;
    uint32_t            remote_rkey;
};

// 初始化：创建 RC QP，loopback 连接到自己
int rdma_init(struct rdma_context *rctx,
              const char *ib_dev_name,
              void *local_buf, size_t local_size,
              void *remote_buf, size_t remote_size);

// 离散读取 top_k 个条目
// 发起 top_k 个 RDMA READ WR，每个读取 item_size bytes
double rdma_scatter_read(struct rdma_context *rctx,
                         const int64_t *indices,  // token positions
                         int top_k,
                         int item_size) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Post all RDMA READ requests
    for (int i = 0; i < top_k; i++) {
        struct ibv_sge sge = {
            .addr   = (uint64_t)rctx->local_mr->addr + i * item_size,
            .length = item_size,
            .lkey   = rctx->local_mr->lkey,
        };
        struct ibv_send_wr wr = {
            .wr_id      = i,
            .sg_list    = &sge,
            .num_sge    = 1,
            .opcode     = IBV_WR_RDMA_READ,
            .send_flags = (i == top_k - 1) ? IBV_SEND_SIGNALED : 0,
            .wr.rdma = {
                .remote_addr = rctx->remote_addr + indices[i] * item_size,
                .rkey        = rctx->remote_rkey,
            },
        };
        struct ibv_send_wr *bad_wr;
        ibv_post_send(rctx->qp, &wr, &bad_wr);
    }

    // Poll for completion
    struct ibv_wc wc;
    while (ibv_poll_cq(rctx->cq, 1, &wc) == 0) {}
    assert(wc.status == IBV_WC_SUCCESS);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_us = (t1.tv_sec - t0.tv_sec) * 1e6 +
                        (t1.tv_nsec - t0.tv_nsec) / 1e3;
    return elapsed_us;
}
```

Python 侧将 RDMA 阶段与 GPU 阶段串联：

```python
def bench_rdma(rdma_ctx, staging_buf, device_buf, seq_len, top_k, item_size,
               warmup=50, iters=200):
    """RDMA scatter read + GPU scatter read."""
    latencies_rdma = []
    latencies_gpu = []
    latencies_total = []

    for it in range(-warmup, iters):
        indices_cpu = torch.randint(0, seq_len, (top_k,), dtype=torch.int64)
        indices_gpu = indices_cpu.cuda()

        # Phase 1: RDMA scatter read (remote → local staging)
        rdma_us = rdma_ext.scatter_read(rdma_ctx, indices_cpu.data_ptr(),
                                         top_k, item_size)

        # Phase 2: GPU scatter read (local staging → GPU)
        # staging_buf 已经是 pinned, indices 是连续 0..top_k-1
        contiguous_indices = torch.arange(top_k, dtype=torch.int64, device="cuda")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch_scatter_kernel(staging_buf.data_ptr(), device_buf.data_ptr(),
                              contiguous_indices.data_ptr(), item_size, top_k)
        end.record()
        torch.cuda.synchronize()
        gpu_us = start.elapsed_time(end) * 1000

        if it >= 0:
            latencies_rdma.append(rdma_us)
            latencies_gpu.append(gpu_us)
            latencies_total.append(rdma_us + gpu_us)

    return latencies_rdma, latencies_gpu, latencies_total
```

---

## 9. 输出格式

```
=== Single-Layer Sparse KV Read Latency Benchmark ===
Model: DeepSeek-V3.2 (MLA FP8, item_size=656B, store_dtype=uint8)

[Config: seq_len=131072, top_k=2048, data=1.31MB]

Backend         | Mean (μs) | Std (μs) |  P50 (μs) |  P99 (μs) | Eff.BW (GB/s)
----------------|-----------|----------|-----------|-----------|---------------
DRAM            |    xxx.xx |    xx.xx |    xxx.xx |    xxx.xx |         xx.xx
CXL             |    xxx.xx |    xx.xx |    xxx.xx |    xxx.xx |         xx.xx
RDMA (total)    |   xxxx.xx |   xxx.xx |   xxxx.xx |   xxxx.xx |          x.xx
  └─ rdma read  |   xxxx.xx |   xxx.xx |           |           |
  └─ gpu copy   |    xxx.xx |    xx.xx |           |           |

CXL vs DRAM overhead:  +xx.x%
RDMA vs CXL overhead:  +xxxx.x%  (xx.x× slower)
RDMA vs DRAM overhead: +xxxx.x%  (xx.x× slower)

[Sweep: seq_len=131072, varying top-k]

top-k |  DRAM (μs) |   CXL (μs) |  RDMA (μs) | CXL/DRAM | RDMA/CXL
------|------------|------------|------------|----------|----------
   64 |     xxx.xx |     xxx.xx |    xxxx.xx |    x.xx× |    xx.x×
  128 |     xxx.xx |     xxx.xx |    xxxx.xx |    x.xx× |    xx.x×
  256 |     xxx.xx |     xxx.xx |    xxxx.xx |    x.xx× |    xx.x×
  512 |     xxx.xx |     xxx.xx |    xxxx.xx |    x.xx× |    xx.x×
 1024 |     xxx.xx |     xxx.xx |    xxxx.xx |    x.xx× |    xx.x×
 2048 |     xxx.xx |     xxx.xx |    xxxx.xx |    x.xx× |    xx.x×
```

---

## 10. 环境要求

### 10.1 必须

| 组件 | 要求 |
|------|------|
| GPU | 1× NVIDIA H20/H100/A100 |
| CXL | CXL 2.0 type-3 设备 ≥ 16GB, DAX 设备可用 |
| RDMA NIC | Mellanox ConnectX-5/6/7, 100/200Gbps IB 或 RoCE |
| OS | Linux 5.15+, `libibverbs-dev` 已安装 |
| CUDA | ≥ 12.0 |

### 10.2 RDMA 测试模式

- **默认模式：本机 NIC loopback**——使用单机 RDMA 网卡，QP 连接到自身，数据经 NIC 硬件处理。可有效体现 RDMA per-message 开销
- **可选扩展：跨机 RDMA**——如果具备两台机器 + IB 直连，可补充跨机数据作为对比。跨机结果将体现更高的 per-message 延迟

---

## 11. 运行命令

```bash
cd test/cxl_utils

# 编译 RDMA 模块
make rdma

# 运行完整 benchmark（RDMA 使用本机 NIC loopback）
python sparse_kv_bench.py \
    --backends dram,cxl,rdma \
    --cxl-dev /dev/dax0.0 \
    --ib-dev mlx5_0 \
    --rdma-mode loopback \
    --seq-lens 32768,65536,131072 \
    --top-ks 64,128,256,512,1024,2048 \
    --item-size 656 \
    --warmup 50 \
    --iters 200 \
    --output results/

# 仅运行 DRAM + CXL（无 RDMA 网卡时）
python sparse_kv_bench.py \
    --backends dram,cxl \
    --cxl-dev /dev/dax0.0 \
    --seq-lens 32768,65536,131072 \
    --top-ks 64,128,256,512,1024,2048 \
    --item-size 656 \
    --output results/
```
