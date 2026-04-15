/*
 * RDMA scatter read — implementation using libibverbs.
 *
 * Sets up a loopback RC QP on a single IB device and issues per-token
 * RDMA READ WRs to measure the per-message overhead of discrete reads
 * (the core bottleneck when fetching sparse KV cache over RDMA).
 */

#include "rdma_scatter_read.h"

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/*  helpers                                                            */
/* ------------------------------------------------------------------ */

static struct ibv_context *open_device(const char *name) {
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    if (!dev_list) {
        fprintf(stderr, "rdma: ibv_get_device_list failed\n");
        return NULL;
    }
    struct ibv_context *ctx = NULL;
    for (int i = 0; dev_list[i]; ++i) {
        if (name == NULL || strcmp(ibv_get_device_name(dev_list[i]), name) == 0) {
            ctx = ibv_open_device(dev_list[i]);
            break;
        }
    }
    ibv_free_device_list(dev_list);
    if (!ctx)
        fprintf(stderr, "rdma: device '%s' not found\n", name ? name : "(any)");
    return ctx;
}

static int modify_qp_to_init(struct ibv_qp *qp, uint8_t port) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state        = IBV_QPS_INIT;
    attr.port_num        = port;
    attr.pkey_index      = 0;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_READ |
                           IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_LOCAL_WRITE;
    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                         IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS);
}

static int modify_qp_to_rtr(struct ibv_qp *qp, uint8_t port,
                             uint32_t dest_qpn, uint16_t dest_lid,
                             union ibv_gid *dest_gid, int gid_index) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state              = IBV_QPS_RTR;
    attr.path_mtu              = IBV_MTU_4096;
    attr.dest_qp_num           = dest_qpn;
    attr.rq_psn                = 0;
    attr.max_dest_rd_atomic    = 16;
    attr.min_rnr_timer         = 12;
    attr.ah_attr.is_global     = 1;
    attr.ah_attr.grh.dgid      = *dest_gid;
    attr.ah_attr.grh.sgid_index = gid_index;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.dlid          = dest_lid;
    attr.ah_attr.sl            = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num      = port;
    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                         IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                         IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
}

static int modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state      = IBV_QPS_RTS;
    attr.timeout        = 14;
    attr.retry_cnt      = 7;
    attr.rnr_retry      = 7;
    attr.sq_psn         = 0;
    attr.max_rd_atomic  = 16;
    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                         IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                         IBV_QP_MAX_QP_RD_ATOMIC);
}

/* ------------------------------------------------------------------ */
/*  public API                                                         */
/* ------------------------------------------------------------------ */

int rdma_init_loopback(struct rdma_context *rctx,
                       const char *ib_dev_name,
                       void *local_buf, size_t local_size,
                       void *remote_buf, size_t remote_size,
                       int max_topk) {
    memset(rctx, 0, sizeof(*rctx));
    rctx->port_num = 1;
    rctx->gid_index = 0;

    /* Open device */
    rctx->ctx = open_device(ib_dev_name);
    if (!rctx->ctx) return -1;

    /* Query port for LID and GID */
    struct ibv_port_attr port_attr;
    if (ibv_query_port(rctx->ctx, rctx->port_num, &port_attr)) {
        fprintf(stderr, "rdma: ibv_query_port failed\n");
        goto err;
    }
    rctx->lid = port_attr.lid;
    if (ibv_query_gid(rctx->ctx, rctx->port_num, rctx->gid_index, &rctx->gid)) {
        fprintf(stderr, "rdma: ibv_query_gid failed\n");
        goto err;
    }

    /* Allocate PD */
    rctx->pd = ibv_alloc_pd(rctx->ctx);
    if (!rctx->pd) { fprintf(stderr, "rdma: ibv_alloc_pd failed\n"); goto err; }

    /* Create CQs (one for send, one for recv — recv CQ unused for RDMA READ) */
    int cq_size;
    cq_size = max_topk + 64;
    rctx->send_cq = ibv_create_cq(rctx->ctx, cq_size, NULL, NULL, 0);
    if (!rctx->send_cq) { fprintf(stderr, "rdma: ibv_create_cq(send) failed\n"); goto err; }
    rctx->recv_cq = ibv_create_cq(rctx->ctx, 16, NULL, NULL, 0);
    if (!rctx->recv_cq) { fprintf(stderr, "rdma: ibv_create_cq(recv) failed\n"); goto err; }

    /* Query device to get max_qp_wr */
    struct ibv_device_attr dev_attr;
    if (ibv_query_device(rctx->ctx, &dev_attr)) {
        fprintf(stderr, "rdma: ibv_query_device failed\n");
        goto err;
    }
    rctx->max_send_wr = dev_attr.max_qp_wr;
    if (rctx->max_send_wr > cq_size)
        rctx->max_send_wr = cq_size;

    /* Create QP (RC) */
    {
        struct ibv_qp_init_attr qp_init;
        memset(&qp_init, 0, sizeof(qp_init));
        qp_init.send_cq = rctx->send_cq;
        qp_init.recv_cq = rctx->recv_cq;
        qp_init.qp_type = IBV_QPT_RC;
        qp_init.cap.max_send_wr  = rctx->max_send_wr;
        qp_init.cap.max_recv_wr  = 1;
        qp_init.cap.max_send_sge = 1;
        qp_init.cap.max_recv_sge = 1;
        rctx->qp = ibv_create_qp(rctx->pd, &qp_init);
        if (!rctx->qp) { fprintf(stderr, "rdma: ibv_create_qp failed\n"); goto err; }
    }

    /* Register MRs */
    rctx->local_buf  = local_buf;
    rctx->local_size = local_size;
    rctx->remote_buf  = remote_buf;
    rctx->remote_size = remote_size;

    rctx->local_mr = ibv_reg_mr(rctx->pd, local_buf, local_size,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!rctx->local_mr) { fprintf(stderr, "rdma: ibv_reg_mr(local) failed: %s\n", strerror(errno)); goto err; }

    rctx->remote_mr = ibv_reg_mr(rctx->pd, remote_buf, remote_size,
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                                 IBV_ACCESS_REMOTE_WRITE);
    if (!rctx->remote_mr) { fprintf(stderr, "rdma: ibv_reg_mr(remote) failed: %s\n", strerror(errno)); goto err; }

    rctx->remote_addr = (uint64_t)(uintptr_t)remote_buf;
    rctx->remote_rkey = rctx->remote_mr->rkey;

    /* Transition QP: RESET → INIT → RTR → RTS (loopback to self) */
    if (modify_qp_to_init(rctx->qp, rctx->port_num)) {
        fprintf(stderr, "rdma: modify_qp_to_init failed\n"); goto err;
    }
    if (modify_qp_to_rtr(rctx->qp, rctx->port_num,
                          rctx->qp->qp_num, rctx->lid,
                          &rctx->gid, rctx->gid_index)) {
        fprintf(stderr, "rdma: modify_qp_to_rtr failed\n"); goto err;
    }
    if (modify_qp_to_rts(rctx->qp)) {
        fprintf(stderr, "rdma: modify_qp_to_rts failed\n"); goto err;
    }

    return 0;

err:
    rdma_destroy(rctx);
    return -1;
}

double rdma_scatter_read(struct rdma_context *rctx,
                         const int64_t *indices,
                         int top_k,
                         int item_size) {
    struct timespec t0, t1;

    /*
     * Batch strategy: post all WRs, signal every `signal_interval` WR.
     * This avoids overflowing the SQ if top_k > max_send_wr by draining
     * completions in batches.  For typical top_k <= 2048 and max_send_wr
     * >= 4096, a single batch suffices.
     */
    int signal_interval = 64;
    if (signal_interval > top_k)
        signal_interval = top_k;

    clock_gettime(CLOCK_MONOTONIC, &t0);

    int posted = 0;
    int completed = 0;
    int signals_expected = 0;

    while (posted < top_k) {
        int batch_end = posted + signal_interval;
        if (batch_end > top_k)
            batch_end = top_k;

        for (int i = posted; i < batch_end; ++i) {
            int is_last_in_batch = (i == batch_end - 1);

            struct ibv_sge sge;
            sge.addr   = (uint64_t)(uintptr_t)rctx->local_buf + (uint64_t)i * item_size;
            sge.length = (uint32_t)item_size;
            sge.lkey   = rctx->local_mr->lkey;

            struct ibv_send_wr wr;
            memset(&wr, 0, sizeof(wr));
            wr.wr_id      = (uint64_t)i;
            wr.sg_list    = &sge;
            wr.num_sge    = 1;
            wr.opcode     = IBV_WR_RDMA_READ;
            wr.send_flags = is_last_in_batch ? IBV_SEND_SIGNALED : 0;
            wr.wr.rdma.remote_addr = rctx->remote_addr +
                                     (uint64_t)indices[i] * item_size;
            wr.wr.rdma.rkey        = rctx->remote_rkey;

            struct ibv_send_wr *bad_wr = NULL;
            int ret = ibv_post_send(rctx->qp, &wr, &bad_wr);
            if (ret) {
                fprintf(stderr, "rdma: ibv_post_send failed at i=%d: %s\n",
                        i, strerror(ret));
                return -1.0;
            }
        }
        signals_expected++;
        posted = batch_end;

        /* Drain completions for this batch's signalled WR */
        struct ibv_wc wc;
        int ne;
        do {
            ne = ibv_poll_cq(rctx->send_cq, 1, &wc);
        } while (ne == 0);
        if (ne < 0 || wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr, "rdma: CQ poll error, status=%d\n",
                    ne < 0 ? -1 : (int)wc.status);
            return -1.0;
        }
        completed++;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_us = (double)(t1.tv_sec - t0.tv_sec) * 1e6 +
                        (double)(t1.tv_nsec - t0.tv_nsec) / 1e3;
    return elapsed_us;
}

double rdma_bulk_read(struct rdma_context *rctx, size_t total_bytes) {
    if (total_bytes == 0) return 0.0;

    /* Cap at available buffer sizes */
    if (total_bytes > rctx->remote_size)
        total_bytes = rctx->remote_size;
    if (total_bytes > rctx->local_size)
        total_bytes = rctx->local_size;

    /* Max message size per WR -- use 1MB chunks for efficiency */
    const size_t MAX_CHUNK = 1 * 1024 * 1024;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    size_t offset = 0;
    int signal_interval = 16;
    int wr_count = 0;
    int signals_expected = 0;

    while (offset < total_bytes) {
        size_t chunk = total_bytes - offset;
        if (chunk > MAX_CHUNK) chunk = MAX_CHUNK;

        wr_count++;
        int do_signal = (wr_count % signal_interval == 0) ||
                        (offset + chunk >= total_bytes);

        struct ibv_sge sge;
        sge.addr   = (uint64_t)(uintptr_t)rctx->local_buf + offset;
        sge.length = (uint32_t)chunk;
        sge.lkey   = rctx->local_mr->lkey;

        struct ibv_send_wr wr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id      = (uint64_t)wr_count;
        wr.sg_list    = &sge;
        wr.num_sge    = 1;
        wr.opcode     = IBV_WR_RDMA_READ;
        wr.send_flags = do_signal ? IBV_SEND_SIGNALED : 0;
        wr.wr.rdma.remote_addr = rctx->remote_addr + offset;
        wr.wr.rdma.rkey        = rctx->remote_rkey;

        struct ibv_send_wr *bad_wr = NULL;
        int ret = ibv_post_send(rctx->qp, &wr, &bad_wr);
        if (ret) {
            fprintf(stderr, "rdma_bulk_read: ibv_post_send failed at offset=%zu: %s\n",
                    offset, strerror(ret));
            return -1.0;
        }

        offset += chunk;

        if (do_signal) {
            signals_expected++;
            struct ibv_wc wc;
            int ne;
            do {
                ne = ibv_poll_cq(rctx->send_cq, 1, &wc);
            } while (ne == 0);
            if (ne < 0 || wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "rdma_bulk_read: CQ poll error, status=%d\n",
                        ne < 0 ? -1 : (int)wc.status);
                return -1.0;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_us = (double)(t1.tv_sec - t0.tv_sec) * 1e6 +
                        (double)(t1.tv_nsec - t0.tv_nsec) / 1e3;
    return elapsed_us;
}

double rdma_bulk_read_offset(struct rdma_context *rctx,
                             size_t remote_offset,
                             size_t local_offset,
                             size_t length) {
    if (length == 0) return 0.0;

    if (remote_offset + length > rctx->remote_size)
        length = rctx->remote_size - remote_offset;
    if (local_offset + length > rctx->local_size)
        length = rctx->local_size - local_offset;

    const size_t MAX_CHUNK = 1 * 1024 * 1024;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    size_t done = 0;
    int signal_interval = 16;
    int wr_count = 0;

    while (done < length) {
        size_t chunk = length - done;
        if (chunk > MAX_CHUNK) chunk = MAX_CHUNK;

        wr_count++;
        int do_signal = (wr_count % signal_interval == 0) ||
                        (done + chunk >= length);

        struct ibv_sge sge;
        sge.addr   = (uint64_t)(uintptr_t)rctx->local_buf + local_offset + done;
        sge.length = (uint32_t)chunk;
        sge.lkey   = rctx->local_mr->lkey;

        struct ibv_send_wr wr;
        memset(&wr, 0, sizeof(wr));
        wr.wr_id      = (uint64_t)wr_count;
        wr.sg_list    = &sge;
        wr.num_sge    = 1;
        wr.opcode     = IBV_WR_RDMA_READ;
        wr.send_flags = do_signal ? IBV_SEND_SIGNALED : 0;
        wr.wr.rdma.remote_addr = rctx->remote_addr + remote_offset + done;
        wr.wr.rdma.rkey        = rctx->remote_rkey;

        struct ibv_send_wr *bad_wr = NULL;
        int ret = ibv_post_send(rctx->qp, &wr, &bad_wr);
        if (ret) {
            fprintf(stderr, "rdma_bulk_read_offset: ibv_post_send failed at done=%zu: %s\n",
                    done, strerror(ret));
            return -1.0;
        }

        done += chunk;

        if (do_signal) {
            struct ibv_wc wc;
            int ne;
            do {
                ne = ibv_poll_cq(rctx->send_cq, 1, &wc);
            } while (ne == 0);
            if (ne < 0 || wc.status != IBV_WC_SUCCESS) {
                fprintf(stderr, "rdma_bulk_read_offset: CQ poll error, status=%d\n",
                        ne < 0 ? -1 : (int)wc.status);
                return -1.0;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_us = (double)(t1.tv_sec - t0.tv_sec) * 1e6 +
                        (double)(t1.tv_nsec - t0.tv_nsec) / 1e3;
    return elapsed_us;
}

void rdma_destroy(struct rdma_context *rctx) {
    if (!rctx) return;
    if (rctx->qp)       ibv_destroy_qp(rctx->qp);
    if (rctx->local_mr)  ibv_dereg_mr(rctx->local_mr);
    if (rctx->remote_mr) ibv_dereg_mr(rctx->remote_mr);
    if (rctx->send_cq)   ibv_destroy_cq(rctx->send_cq);
    if (rctx->recv_cq)   ibv_destroy_cq(rctx->recv_cq);
    if (rctx->pd)        ibv_dealloc_pd(rctx->pd);
    if (rctx->ctx)       ibv_close_device(rctx->ctx);
    memset(rctx, 0, sizeof(*rctx));
}
