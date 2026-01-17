# Base Makefile template for building a single-file SYCL translation.
#
# Usage:
#   make -f sycl_makefile_template.mk \
#        SYCL_COMPILER=icpx \
#        SYCL_FLAGS='-fsycl -O3 -std=c++17' \
#        SRC=Main_01-16-2026_t12_34_56_Trial1.cpp \
#        OUT=xsbench_sycl
#
# Notes:
# - This is intentionally generic so it can be reused for all generated SYCL code.
# - The pipeline/notebook should pass SRC/OUT per experiment.

SYCL_COMPILER ?= <SYCL_COMPILER>
SYCL_FLAGS    ?= <SYCL_CXXFLAGS>
SYCL_LDFLAGS  ?= <SYCL_LDFLAGS>

SRC ?= <SYCL_SOURCE_CPP>
OUT ?= <SYCL_OUTPUT_EXE>

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	$(SYCL_COMPILER) $(SYCL_FLAGS) $(SRC) -o $(OUT) $(SYCL_LDFLAGS)

clean:
	rm -f $(OUT)
