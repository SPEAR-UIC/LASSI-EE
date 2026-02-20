#!/bin/bash
# LASSI XSBench CUDA→SYCL Demo - Interactive Compute Node Runner
#
# Run this script ONCE you are already on an Aurora compute node obtained via:
#   qsub -I -l select=1,walltime=01:00:00,filesystems=flare -A EE-ECP -q debug
#
# Usage (from the interactive shell on the compute node):
#   cd /home/vchirio/LASSI-EE/lassi_translate
#   bash run_aurora_demo_interactive.sh

set -eo pipefail

echo "========================================================================"
echo "  LASSI  |  XSBench CUDA→SYCL Autonomous Translation Demo"
echo "========================================================================"
echo "Hostname:   $(hostname)"
echo "Login node: ${PBS_O_HOST:-unknown}"
echo "Start:      $(date)"
echo "========================================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f ~/inference_auth_token.py ]; then
    echo "ERROR: ALCF token helper not found at ~/inference_auth_token.py"
    echo "  Download : wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py"
    echo "  Auth     : conda run -n globus_env python3 ~/inference_auth_token.py authenticate"
    exit 1
fi

if [ ! -f source_codes/Miniapps/XSBench/cuda/Main.cu ]; then
    echo "ERROR: XSBench CUDA source not found."
    echo "  Expected: $SCRIPT_DIR/source_codes/Miniapps/XSBench/cuda/Main.cu"
    exit 1
fi

echo "  ✓ ALCF token helper found"
echo "  ✓ XSBench CUDA source found"
echo ""

# ── LLM configuration ────────────────────────────────────────────────────────
LLM_MODEL=${LLM_MODEL:-"gpt-oss-120b"}
ALCF_CLUSTER=${ALCF_CLUSTER:-"metis"}
ALCF_CONDA_ENV=${ALCF_CONDA_ENV:-"globus_env"}

export ALCF_CLUSTER
export ALCF_CONDA_ENV
export ALCF_MODEL="$LLM_MODEL"

echo "LLM configuration:"
echo "  Backend:   ALCF Inference Endpoints"
echo "  Model:     $ALCF_MODEL"
echo "  Cluster:   $ALCF_CLUSTER"
echo "  Conda env: $ALCF_CONDA_ENV"
echo ""

# ── Load Python environment ──────────────────────────────────────────────────
echo "Loading modules..."
module load frameworks
echo ""

# Resolve the globus_env Python by absolute path so that conda activate never
# touches PATH — conda activate inside a bash subshell deactivates whatever the
# frameworks module set up, stripping the oneAPI bin dir (sycl-ls, icpx, …)
# from the environment that Python inherits.
# We use "conda run -n globus_env which python3" only to LOCATE the interpreter;
# the pipeline itself is invoked directly so PATH is fully preserved.
PYTHON3=$(conda run -n globus_env which python3 2>/dev/null || true)
if [ ! -x "$PYTHON3" ]; then
    echo "ERROR: could not locate Python in globus_env conda environment."
    echo "  Check: conda env list | grep globus_env"
    exit 1
fi
echo "Python (globus_env): $($PYTHON3 --version)"
echo ""

# ── SYCL toolchain (icpx from default oneAPI env on Aurora) ──────────────────
export SYCL_COMPILER="icpx"
export SYCL_CXXFLAGS="-O2 -std=c++17 -fsycl"
export SYCL_LDFLAGS=""

ICPX_PATH=$(which icpx 2>/dev/null || echo "NOT FOUND")
if [ "$ICPX_PATH" = "NOT FOUND" ]; then
    echo "ERROR: icpx not found. The oneAPI env should provide it by default on Aurora."
    exit 1
fi

echo "SYCL toolchain:"
echo "  Compiler : icpx  ($ICPX_PATH)"
echo "  Version  : $(icpx --version 2>&1 | head -1)"
echo "  Flags    : $SYCL_CXXFLAGS"
echo ""

# ── Run pipeline ─────────────────────────────────────────────────────────────
echo "========================================================================"
echo "  Starting LASSI Translation Pipeline"
echo "========================================================================"
echo ""

"$PYTHON3" -u run_lassi_translate_v3.py \
    --model        "$LLM_MODEL" \
    --no-verifyprompt

PIPELINE_RC=$?

echo ""
echo "========================================================================"
echo "  Summary"
echo "========================================================================"
echo "End: $(date)"
echo ""

LATEST=$(ls -td generated_codes/experiment/exp_xsbench_* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "Experiment directory: $LATEST"
    echo ""
    if [ $PIPELINE_RC -eq 0 ]; then
        echo "Status: SUCCESS ✓"
        echo ""
        echo "--- Results CSV ---"
        cat "$LATEST/results.csv" 2>/dev/null || true
        echo ""
        echo "--- SYCL execution output ---"
        cat "$LATEST/"run_trial*_stdout.log 2>/dev/null | tail -30 || true
    else
        echo "Status: FAILED ✗"
        echo ""
        echo "Key log files:"
        ls "$LATEST/"*.log "$LATEST/"*.txt 2>/dev/null | head -10 || true
    fi
fi

echo "========================================================================"
exit $PIPELINE_RC
