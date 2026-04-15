/*
 * RDMA scatter read — header.
 *
 * Provides RC QP loopback setup and per-top-k discrete RDMA READ to measure
 * per-message overhead in a sparse KV cache access pattern.
 */

#ifndef RDMA_SCATTER_READ_H
#define RDMA_SCATTER_READ_H

#include <infiniband/verbs.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct rdma_context {
    struct ibv_context *ctx;
    struct ibv_pd      *pd;
    struct ibv_cq      *send_cq;
    struct ibv_cq      *recv_cq;
    struct ibv_qp      *qp;

    struct ibv_mr      *local_mr;
    struct ibv_mr      *remote_mr;

    void               *local_buf;
    size_t              local_size;
    void               *remote_buf;
    size_t              remote_size;

    uint64_t            remote_addr;
    uint32_t            remote_rkey;

    uint16_t            lid;
    uint8_t             port_num;
    int                 gid_index;
    union ibv_gid       gid;
    int                 max_send_wr;
};

/*
 * Initialise a loopback RDMA context:
 *   - Opens the named IB device (e.g. "mlx5_0")
 *   - Allocates PD, CQ, QP (RC)
 *   - Registers local staging buffer and "remote" KV pool as MRs
 *   - Transitions QP through INIT → RTR → RTS with loopback addressing
 *
 * Returns 0 on success, -1 on error.
 */
int rdma_init_loopback(struct rdma_context *rctx,
                       const char *ib_dev_name,
                       void *local_buf, size_t local_size,
                       void *remote_buf, size_t remote_size,
                       int max_topk);

/*
 * Perform discrete RDMA READs for top_k items.
 *
 * Each of the top_k indices triggers one RDMA READ of item_size bytes from
 * the "remote" MR into the local staging MR.  Only the last WR is signalled.
 *
 * Returns elapsed time in microseconds (CLOCK_MONOTONIC).
 */
double rdma_scatter_read(struct rdma_context *rctx,
                         const int64_t *indices,
                         int top_k,
                         int item_size);

/*
 * Perform a bulk contiguous RDMA READ of total_bytes from remote MR offset 0
 * into local MR offset 0.  Splits into max-message-size chunks internally.
 *
 * Used by RDMAMLATokenToKVPoolHost.prefetch_for_decode() to simulate
 * one-shot full-KV prefetch for a request entering decode.
 *
 * Returns elapsed time in microseconds (CLOCK_MONOTONIC).
 */
double rdma_bulk_read(struct rdma_context *rctx, size_t total_bytes);

/*
 * Offset-aware bulk RDMA READ: read `length` bytes starting at
 * `remote_offset` in the remote MR into `local_offset` in the local MR.
 *
 * Used by request_striping mode where each NIC reads a different shard
 * of the same request's KV data from a specific region of remote_buffer
 * into a specific region of landing_staging.
 *
 * Returns elapsed time in microseconds (CLOCK_MONOTONIC).
 */
double rdma_bulk_read_offset(struct rdma_context *rctx,
                             size_t remote_offset,
                             size_t local_offset,
                             size_t length);

/*
 * Tear down all RDMA resources.
 */
void rdma_destroy(struct rdma_context *rctx);

#ifdef __cplusplus
}
#endif

#endif /* RDMA_SCATTER_READ_H */
