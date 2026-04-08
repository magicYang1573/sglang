/*
 * pybind11 wrapper for the RDMA scatter read C library.
 *
 * Exposes three functions to Python:
 *   - rdma_init(ib_dev, local_buf_ptr, local_size, remote_buf_ptr, remote_size, max_topk)
 *   - rdma_scatter_read(ctx_handle, indices_ptr, top_k, item_size) -> elapsed_us
 *   - rdma_destroy(ctx_handle)
 *
 * Build: compiled together with rdma_scatter_read.c via Makefile.
 */

#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

extern "C" {
#include "rdma_scatter_read.h"
}

namespace py = pybind11;

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

PYBIND11_MODULE(rdma_scatter_ext, m) {
    m.doc() = "pybind11 wrapper for RDMA loopback scatter read benchmark";

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
}
