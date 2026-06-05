#!/bin/bash
# polaris_worker.sh — One-shot compile + run on a Polaris compute node
# (NVIDIA A100 / sm_80).
#
# Usage: bash polaris_worker.sh <task> <src> <outdir> [run_args...]
#   task : translate | cuda_energy
#
# Optional env overrides:
#   SYCL_COMPILER, SYCL_FLAGS, SYCL_LDFLAGS
#   CUDA_COMPILER, CUDA_FLAGS, CUDA_LDFLAGS
#
# Exit codes: 0=both OK, 1=compile failed, 2=execution failed.

set -o pipefail
# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

TASK="$1"
SRC="$2"
OUTDIR="$3"
shift 3
RUN_ARGS=("$@")

mkdir -p "$OUTDIR"
LOG="${OUTDIR}/build_stdout.log"
: > "$LOG"
: > "${OUTDIR}/build_stderr.log"

{
    echo "=== polaris_worker.sh ==="
    echo "Task    : $TASK"
    echo "Date    : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Hostname: $(hostname)"
    echo "Source  : $SRC"
    echo "Outdir  : $OUTDIR"
    echo "Run args: ${RUN_ARGS[*]:-<none>}"
    echo ""
} >> "$LOG"

_setup_lmod_oneapi "$LOG"
_resolve_icpx       "$LOG"
_setup_cuda         "$LOG"
_resolve_cuda_path  "$LOG"
_resolve_nvidia_smi

{
    echo "=== Tools ==="
    echo "  icpx      : $(command -v icpx 2>/dev/null || echo NOT FOUND)"
    echo "  nvcc      : $(command -v nvcc 2>/dev/null || echo NOT FOUND)"
    echo "  nvidia-smi: ${NVIDIA_SMI:-NOT FOUND}"
    echo ""
} >> "$LOG"

# ── Per-task settings ─────────────────────────────────────────────────────────
_CUDA_FLAG=""
[ -n "$CUDA_PATH" ] && _CUDA_FLAG="--cuda-path=${CUDA_PATH}"

case "$TASK" in
    cuda_energy)
        COMPILER="${CUDA_COMPILER:-nvcc}"
        FLAGS="${CUDA_FLAGS:--O3 -arch=sm_80 -std=c++17 -allow-unsupported-compiler -D_Float128=__float128}"
        LDFLAGS="${CUDA_LDFLAGS:-}"
        MONITOR=nvidia_smi
        DEVICE_SELECTOR=""
        ;;
    translate)
        COMPILER="${SYCL_COMPILER:-icpx}"
        _SYCL_DEFAULT="-O3 -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
-Xsycl-target-backend --cuda-gpu-arch=sm_80${_CUDA_FLAG:+ $_CUDA_FLAG}"
        FLAGS="${SYCL_FLAGS:-$_SYCL_DEFAULT}"
        LDFLAGS="${SYCL_LDFLAGS:-}"
        MONITOR=none
        DEVICE_SELECTOR="cuda:gpu"
        ;;
    *)
        echo "ERROR: unknown task '$TASK' (expected: translate|cuda_energy)" \
            | tee -a "$LOG" > "${OUTDIR}/build_stderr.log"
        echo -n "1" > "${OUTDIR}/build_rc.txt"
        exit 1
        ;;
esac

_compile_and_run "$COMPILER" "$FLAGS" "$LDFLAGS" "$SRC" "$OUTDIR" \
                 "$MONITOR" "$DEVICE_SELECTOR" 0 \
                 "${RUN_ARGS[@]}"

BUILD_RC=$(cat "${OUTDIR}/build_rc.txt")
RUN_RC=$(cat "${OUTDIR}/run_rc.txt" 2>/dev/null || echo -1)

[ "$BUILD_RC" -ne 0 ] && exit 1
[ "$RUN_RC"   -ne 0 ] && exit 2
exit 0
