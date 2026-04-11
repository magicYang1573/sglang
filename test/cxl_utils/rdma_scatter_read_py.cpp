/*
 * pybind11 wrapper for the RDMA scatter read C library.
 *
 * Provides two APIs:
 *
 * 1. Global singleton (legacy, used by sparse_kv_bench.py):
 *      rdma_init / rdma_scatter_read / rdma_bulk_read / rdma_destroy
 *
 * 2. Multi-instance RDMAContext class (used by rdma_memory_pool.py for
 *    multi-NIC interleave — each rank holds its own context on a
 *    different IB device):
 *      ctx = RDMAContext()
 *      ctx.init(ib_dev, local_ptr, local_size, remote_ptr, remote_size, max_wr)
 *      us  = ctx.bulk_read(total_bytes)
 *      us  = ctx.scatter_read(indices_ptr, top_k, item_size)
 *      ctx.destroy()
 *
 * Build: compiled together with rdma_scatter_read.c via Makefile.
 */

#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" {
#include "rdma_scatter_read.h"
}

namespace py = pybind11;

/* ------------------------------------------------------------------ */
/*  Legacy global-singleton API                                        */
/* ------------------------------------------------------------------ */

static rdma_context *g_ctx = nullptr;

static void py_rdma_init(const std::string &ib_dev_name,
                          uint64_t local_buf_ptr, size_t local_size,
                          uint64_t remote_buf_ptr, size_t remote_size,
                          int max_topk) {
    if (g_ctx) {
        rdma_destroy(g_ctx);
        free(g_ctx);
        g_ctx = nullptr;
    }
    g_ctx = static_cast<rdma_context *>(calloc(1, sizeof(rdma_context)));
    if (!g_ctx)
        throw std::runtime_error("rdma: failed to allocate context");

    int ret = rdma_init_loopback(g_ctx,
                                 ib_dev_name.c_str(),
                                 reinterpret_cast<void *>(local_buf_ptr), local_size,
                                 reinterpret_cast<void *>(remote_buf_ptr), remote_size,
                                 max_topk);
    if (ret != 0) {
        free(g_ctx);
        g_ctx = nullptr;
        throw std::runtime_error("rdma: rdma_init_loopback failed");
    }
}

static double py_rdma_scatter_read(uint64_t indices_ptr,
                                    int top_k,
                                    int item_size) {
    if (!g_ctx)
        throw std::runtime_error("rdma: context not initialised; call rdma_init first");

    double us = rdma_scatter_read(g_ctx,
                                   reinterpret_cast<const int64_t *>(indices_ptr),
                                   top_k,
                                   item_size);
    if (us < 0)
        throw std::runtime_error("rdma: rdma_scatter_read failed");
    return us;
}

static double py_rdma_bulk_read(size_t total_bytes) {
    if (!g_ctx)
        throw std::runtime_error("rdma: context not initialised; call rdma_init first");

    double us = rdma_bulk_read(g_ctx, total_bytes);
    if (us < 0)
        throw std::runtime_error("rdma: rdma_bulk_read failed");
    return us;
}

static void py_rdma_destroy() {
    if (g_ctx) {
        rdma_destroy(g_ctx);
        free(g_ctx);
        g_ctx = nullptr;
    }
}

/* ------------------------------------------------------------------ */
/*  Multi-instance RDMAContext class                                    */
/* ------------------------------------------------------------------ */

class RDMAContextHandle {
    rdma_context ctx_;
    bool initialized_ = false;

public:
    RDMAContextHandle() { memset(&ctx_, 0, sizeof(ctx_)); }

    void init(const std::string &ib_dev,
              uint64_t local_ptr, size_t local_size,
              uint64_t remote_ptr, size_t remote_size,
              int max_wr) {
        if (initialized_) {
            rdma_destroy(&ctx_);
            initialized_ = false;
        }
        int ret = rdma_init_loopback(&ctx_, ib_dev.c_str(),
                                     reinterpret_cast<void *>(local_ptr), local_size,
                                     reinterpret_cast<void *>(remote_ptr), remote_size,
                                     max_wr);
        if (ret != 0)
            throw std::runtime_error("RDMAContext::init failed on " + ib_dev);
        initialized_ = true;
    }

    double bulk_read(size_t total_bytes) {
        if (!initialized_)
            throw std::runtime_error("RDMAContext not initialized");
        double us = rdma_bulk_read(&ctx_, total_bytes);
        if (us < 0)
            throw std::runtime_error("RDMAContext::bulk_read failed");
        return us;
    }

    double scatter_read(uint64_t indices_ptr, int top_k, int item_size) {
        if (!initialized_)
            throw std::runtime_error("RDMAContext not initialized");
        double us = rdma_scatter_read(&ctx_,
                                       reinterpret_cast<const int64_t *>(indices_ptr),
                                       top_k, item_size);
        if (us < 0)
            throw std::runtime_error("RDMAContext::scatter_read failed");
        return us;
    }

    void destroy() {
        if (initialized_) {
            rdma_destroy(&ctx_);
            memset(&ctx_, 0, sizeof(ctx_));
            initialized_ = false;
        }
    }

    bool is_initialized() const { return initialized_; }

    ~RDMAContextHandle() { destroy(); }
};

/* ------------------------------------------------------------------ */
/*  Module definition                                                   */
/* ------------------------------------------------------------------ */

PYBIND11_MODULE(rdma_scatter_ext, m) {
    m.doc() = "pybind11 wrapper for RDMA loopback scatter read benchmark";

    // Legacy global-singleton API
    m.def("rdma_init", &py_rdma_init,
          py::arg("ib_dev_name"),
          py::arg("local_buf_ptr"), py::arg("local_size"),
          py::arg("remote_buf_ptr"), py::arg("remote_size"),
          py::arg("max_topk"),
          "Initialise RDMA loopback context. Pointers are raw uint64 addresses.");

    m.def("rdma_scatter_read", &py_rdma_scatter_read,
          py::arg("indices_ptr"), py::arg("top_k"), py::arg("item_size"),
          "Perform top_k discrete RDMA READs. Returns elapsed microseconds.");

    m.def("rdma_bulk_read", &py_rdma_bulk_read,
          py::arg("total_bytes"),
          "Perform a bulk contiguous RDMA READ. Returns elapsed microseconds.");

    m.def("rdma_destroy", &py_rdma_destroy,
          "Destroy the RDMA context and free resources.");

    // Multi-instance API for per-rank NIC interleave
    py::class_<RDMAContextHandle>(m, "RDMAContext",
        "Per-instance RDMA loopback context. Each instance binds to one IB "
        "device, enabling multi-NIC interleave where each TP rank uses a "
        "different NIC.")
        .def(py::init<>())
        .def("init", &RDMAContextHandle::init,
             py::arg("ib_dev"),
             py::arg("local_ptr"), py::arg("local_size"),
             py::arg("remote_ptr"), py::arg("remote_size"),
             py::arg("max_wr"),
             "Initialize loopback RC QP on the given IB device.")
        .def("bulk_read", &RDMAContextHandle::bulk_read,
             py::arg("total_bytes"),
             py::call_guard<py::gil_scoped_release>(),
             "Bulk contiguous RDMA READ. Returns elapsed microseconds.")
        .def("scatter_read", &RDMAContextHandle::scatter_read,
             py::arg("indices_ptr"), py::arg("top_k"), py::arg("item_size"),
             py::call_guard<py::gil_scoped_release>(),
             "Per-token discrete RDMA READs. Returns elapsed microseconds.")
        .def("destroy", &RDMAContextHandle::destroy,
             "Tear down RDMA resources.")
        .def("is_initialized", &RDMAContextHandle::is_initialized);
}
