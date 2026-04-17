# Makefile for building the RDMA scatter read pybind11 extension.
#
# HiSparse RDMA host pool loads this extension at server startup only (no JIT).
# Build with the same interpreter you use for ``python -m sglang.launch_server``:
#
#   cd test/cxl_utils && make rdma PYTHON=/path/to/venv/bin/python
#
# Usage:
#   make rdma          # Build rdma_scatter_ext$(EXT_SUFFIX)
#   make clean         # Remove build artefacts
#
# Prerequisites:
#   - libibverbs-dev installed (provides infiniband/verbs.h)
#   - pybind11 installed (pip install pybind11)
#   - Python 3 development headers

PYTHON        ?= python3
PYBIND11_INC  := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXT    := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYTHON_INC    := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")

CXX           ?= g++
CC            ?= gcc
CFLAGS        := -O3 -Wall -fPIC
CXXFLAGS      := -O3 -Wall -fPIC -std=c++17

TARGET        := rdma_scatter_ext$(PYTHON_EXT)

RDMA_C_OBJ    := rdma_scatter_read.o
RDMA_PY_OBJ   := rdma_scatter_read_py.o

.PHONY: rdma clean

rdma: $(TARGET)

$(TARGET): $(RDMA_C_OBJ) $(RDMA_PY_OBJ)
	$(CXX) -shared -o $@ $^ -libverbs

$(RDMA_C_OBJ): rdma_scatter_read.c rdma_scatter_read.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(RDMA_PY_OBJ): rdma_scatter_read_py.cpp rdma_scatter_read.h
	$(CXX) $(CXXFLAGS) $(PYBIND11_INC) -I$(PYTHON_INC) -c -o $@ $<

clean:
	rm -f *.o $(TARGET)
