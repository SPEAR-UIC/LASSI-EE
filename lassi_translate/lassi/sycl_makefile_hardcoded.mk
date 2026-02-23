# Hardcoded Makefile template for building a single-file SYCL translation.
#
# This file is used by the pipeline to avoid passing compiler flags via
# `make VAR=...` on the command line (because the pipeline's make wrapper
# splits on whitespace, and GNU Make will treat tokens like `-std=c++17` as
# make options).
#
# The pipeline substitutes:
#   <SYCL_COMPILER> -> absolute path to SYCL compiler (e.g., /home/vchirio/adaptivecpp/bin/acpp)
#   <SYCL_FLAGS>    -> compiler flags string (may contain spaces)
#   <SYCL_LDFLAGS>  -> linker flags string (may contain spaces)
#   <SYCL_SRC>      -> absolute path to the generated .cpp file
#   <SYCL_OUT>      -> output executable name

SYCL_COMPILER := <SYCL_COMPILER>
SYCL_FLAGS    := <SYCL_FLAGS>
SYCL_LDFLAGS  := <SYCL_LDFLAGS>

SRC := <SYCL_SRC>
OUT := <SYCL_OUT>

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	$(SYCL_COMPILER) $(SYCL_FLAGS) $(SRC) -o $(OUT) $(SYCL_LDFLAGS)

clean:
	rm -f $(OUT)
