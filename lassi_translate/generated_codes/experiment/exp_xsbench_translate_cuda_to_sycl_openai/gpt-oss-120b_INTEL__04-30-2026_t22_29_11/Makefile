# Hardcoded Makefile template for building a single-file SYCL translation.
#
# This file is used by the pipeline to avoid passing compiler flags via
# `make VAR=...` on the command line (because the pipeline's make wrapper
# splits on whitespace, and GNU Make will treat tokens like `-std=c++17` as
# make options).
#
# The pipeline substitutes:
#   icpx -> absolute path to SYCL compiler (e.g., /home/vchirio/adaptivecpp/bin/acpp)
#   -O2 -std=c++17 -fsycl    -> compiler flags string (may contain spaces)
#     -> linker flags string (may contain spaces)
#   /home/vchirio/LASSI-EE/lassi_translate/generated_codes/experiment/exp_xsbench_translate_cuda_to_sycl_openai/gpt-oss-120b_INTEL__04-30-2026_t22_29_11/Main_attempt9.cpp      -> absolute path to the generated .cpp file
#   Main      -> output executable name

SYCL_COMPILER := icpx
SYCL_FLAGS    := -O2 -std=c++17 -fsycl
SYCL_LDFLAGS  := 

SRC := /home/vchirio/LASSI-EE/lassi_translate/generated_codes/experiment/exp_xsbench_translate_cuda_to_sycl_openai/gpt-oss-120b_INTEL__04-30-2026_t22_29_11/Main_attempt9.cpp
OUT := Main

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	$(SYCL_COMPILER) $(SYCL_FLAGS) $(SRC) -o $(OUT) $(SYCL_LDFLAGS)

clean:
	rm -f $(OUT)
