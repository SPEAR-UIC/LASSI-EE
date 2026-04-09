#!/bin/bash
# LASSI XSBench CUDA→SYCL Demo - Polaris Compute Node Runner
#
# Run this script ONCE you are already on a Polaris compute node obtained via:
#   qsub -I -l select=1,walltime=01:00:00,filesystems=home:eagle -A EE-ECP -q debug
#
# Usage (from the interactive shell on the compute node):
#   cd /eagle/EE-ECP/lassi_translate
#   bash run_polaris_demo.sh [--debug] [--no-self-prompting]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f ~/inference_auth_token.py ]; then
    echo "ERROR: ALCF token helper not found at ~/inference_auth_token.py"
    echo "  Download : wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py -O ~/inference_auth_token.py"
    echo "  Auth     : source $ALCF_VENV/bin/activate && python ~/inference_auth_token.py authenticate"
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
ALCF_VENV=${ALCF_VENV:-"/eagle/EE-ECP/LASSI-EE/venvs/inference"}

export ALCF_CLUSTER
export ALCF_MODEL="$LLM_MODEL"

# ── Load Python environment ──────────────────────────────────────────────────
# PBS job environments do not initialize the module system automatically.
# Source lmod's init script so that `module` commands work in non-interactive
# shells (compute nodes, qsub-submitted jobs).
if ! command -v module &>/dev/null; then
    source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
fi

module load cuda 2>/dev/null || true
module load oneapi/release 2>/dev/null || true

# Source conda directly rather than via `module load conda` so that conda
# activation is independent of lmod being fully initialized. The venv is then
# activated by absolute path — conda activate inside a bash subshell would
# deactivate whatever the oneapi module set up, stripping the oneAPI bin dir
# (sycl-ls, icpx, …) from the environment that Python inherits.
CONDA_INIT=$(ls -d /soft/applications/conda/*/mconda3/etc/profile.d/conda.sh 2>/dev/null | sort | tail -1)
if [ -n "$CONDA_INIT" ]; then
    # shellcheck source=/dev/null
    source "$CONDA_INIT"
    conda activate base 2>/dev/null || true
fi

PYTHON3="$ALCF_VENV/bin/python3"
if [ ! -x "$PYTHON3" ]; then
    echo "ERROR: could not locate Python in inference venv: $ALCF_VENV"
    echo "  Create it with:"
    echo "    module use /soft/modulefiles; module load conda; conda activate base"
    echo "    python -m venv $ALCF_VENV --system-site-packages"
    echo "    source $ALCF_VENV/bin/activate && pip install globus_sdk openai"
    exit 1
fi

# ── SYCL toolchain ────────────────────────────────────────────────────────────
export SYCL_COMPILER="icpx"
export SYCL_CXXFLAGS="-O2 -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80"
export SYCL_LDFLAGS=""

if ! which icpx &>/dev/null; then
    echo "ERROR: icpx not found. Load the oneAPI module with: module load oneapi/release"
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
