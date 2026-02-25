#!/bin/bash
# LASSI XSBench CUDA→SYCL Demo - Interactive Compute Node Runner
#
# Run this script ONCE you are already on an Aurora compute node obtained via:
#   qsub -I -l select=1,walltime=01:00:00,filesystems=flare -A EE-ECP -q debug
#
# Usage (from the interactive shell on the compute node):
#   cd /home/vchirio/LASSI-EE/lassi_translate
#   bash run_aurora_demo_interactive.sh [--debug] [--no-self-prompting]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# ── LLM / cluster configuration ──────────────────────────────────────────────
LLM_MODEL=${LLM_MODEL:-"openai/gpt-oss-120b"}
ALCF_CLUSTER=${ALCF_CLUSTER:-"sophia"}
ALCF_CONDA_ENV=${ALCF_CONDA_ENV:-"globus_env"}

export ALCF_CLUSTER
export ALCF_CONDA_ENV
export ALCF_MODEL="$LLM_MODEL"

# ── Load Python environment ──────────────────────────────────────────────────
module load frameworks 2>/dev/null || true

# Resolve the globus_env Python by absolute path so that conda activate never
# touches PATH — conda activate inside a bash subshell deactivates whatever the
# frameworks module set up, stripping the oneAPI bin dir (sycl-ls, icpx, …)
# from the environment that Python inherits.
PYTHON3=$(conda run -n globus_env which python3 2>/dev/null || true)
if [ ! -x "$PYTHON3" ]; then
    echo "ERROR: could not locate Python in globus_env conda environment."
    echo "  Check: conda env list | grep globus_env"
    exit 1
fi

# ── SYCL toolchain ────────────────────────────────────────────────────────────
export SYCL_COMPILER="icpx"
export SYCL_CXXFLAGS="-O2 -std=c++17 -fsycl"
export SYCL_LDFLAGS=""

if ! which icpx &>/dev/null; then
    echo "ERROR: icpx not found. The oneAPI env should provide it by default on Aurora."
    exit 1
fi

# ── Run pipeline ─────────────────────────────────────────────────────────────
"$PYTHON3" -u run_lassi_translate_v3.py \
    --model        "$LLM_MODEL" \
    --no-verifyprompt \
    "$@"

PIPELINE_RC=$?

# ── Summary ──────────────────────────────────────────────────────────────────
LATEST=$(ls -td generated_codes/experiment/exp_xsbench_* 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo ""
    echo "========================================================================"
    if [ $PIPELINE_RC -eq 0 ]; then
        printf "\033[1;32m  ✅  Translation Successful\033[0m\n"
        echo "========================================================================"

        # Runtime from CSV
        RUNTIME=$(tail -1 "$LATEST/results.csv" 2>/dev/null | awk -F',' '{print $NF}')
        [ -n "$RUNTIME" ] && echo "  Pipeline runtime : ${RUNTIME} min"

        echo ""
        echo "  SYCL Execution Output"
        echo "  ─────────────────────"

        # Show stdout with key lines highlighted
        STDOUT_LOG=$(ls "$LATEST/"run_trial*_stdout.log 2>/dev/null | head -1)
        if [ -n "$STDOUT_LOG" ]; then
            while IFS= read -r line; do
                if echo "$line" | grep -q "Verification checksum"; then
                    printf "  \033[1;32m%s\033[0m\n" "$line"
                elif echo "$line" | grep -q "Runtime:\|Lookups/s:\|Lookups:"; then
                    printf "  \033[1m%s\033[0m\n" "$line"
                else
                    printf "  %s\n" "$line"
                fi
            done < "$STDOUT_LOG"
        fi
    else
        printf "\033[1;31m  ❌  Translation Failed\033[0m\n"
        echo "========================================================================"
        echo "  Logs: $LATEST"
    fi
    echo "========================================================================"
fi

exit $PIPELINE_RC
