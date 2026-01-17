# Hardcoded Makefile template for building a single-file SYCL translation.
#
# This file is used by the pipeline to avoid passing compiler flags via
# `make VAR=...` on the command line (because the pipeline's make wrapper
# splits on whitespace, and GNU Make will treat tokens like `-std=c++17` as
# make options).
#
# The pipeline substitutes:
#   /home/vchirio/adaptivecpp/bin/acpp -> absolute path to SYCL compiler (e.g., /home/vchirio/adaptivecpp/bin/acpp)
#   -O3 -std=c++17    -> compiler flags string (may contain spaces)
#     -> linker flags string (may contain spaces)
#   /home/vchirio/Desktop/LASSI-EE/source_codes/Miniapps/XSBench/cuda/Main.cpp      -> absolute path to the generated .cpp file
#   Main      -> output executable name

SYCL_COMPILER := /home/vchirio/adaptivecpp/bin/acpp
SYCL_FLAGS    := -O3 -std=c++17
SYCL_LDFLAGS  := 

SRC := /home/vchirio/Desktop/LASSI-EE/source_codes/Miniapps/XSBench/cuda/Main.cpp
OUT := Main

.PHONY: all clean

all: $(OUT)

$(OUT):
	$(SYCL_COMPILER) $(SYCL_FLAGS) $(SRC) -o $(OUT) $(SYCL_LDFLAGS)

clean:
	rm -f $(OUT)
