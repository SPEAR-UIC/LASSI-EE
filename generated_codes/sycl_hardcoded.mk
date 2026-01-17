# Hardcoded Makefile for building a single-file SYCL translation with AdaptiveCpp.
#
# This Makefile is intentionally minimal and avoids passing compiler flags via
# `make VAR=...` from the command line (which is what caused the
# `make: invalid option -- '='` errors when flags like `-std=c++17` were not
# quoted).

SYCL_COMPILER := /home/vchirio/adaptivecpp/bin/acpp
CXXFLAGS      := -O3 -std=c++17
LDFLAGS       :=

SRC := Main.cpp
OUT := Main

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	$(SYCL_COMPILER) $(CXXFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

clean:
	rm -f $(OUT)
