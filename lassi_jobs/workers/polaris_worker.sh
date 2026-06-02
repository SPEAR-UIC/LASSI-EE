#!/bin/bash
# polaris_worker.sh — Compile and run CUDA or SYCL code on a Polaris compute node
# (NVIDIA A100 / sm_80).
#
# Called by run_lassi_jobs.py via _submit_job():
#   bash polaris_worker.sh <task> <src> <outdir> [run_args...]
#
# Arguments:
#   $1        : Task: translate | cuda_energy | sycl_energy
#   $2        : Absolute path to the source file (on Eagle)
#   $3        : Absolute path to the output directory (on Eagle)
#   $4, $5, …: Arguments to pass to the compiled executable
#
# Environment variables (all optional):
#   SYCL_COMPILER   Override SYCL compiler (default: icpx)
#   SYCL_FLAGS      Override SYCL compilation flags
#   SYCL_LDFLAGS    Extra SYCL linker flags
#   CUDA_COMPILER   Override CUDA compiler (default: nvcc)
#   CUDA_FLAGS      Override CUDA compilation flags
#   CUDA_LDFLAGS    Extra CUDA linker flags
#
# Outputs written to <outdir>:
#   build_rc.txt, build_stdout.log, build_stderr.log
#   run_rc.txt, run_stdout.log, run_stderr.log
#   power_samples.log, energy_results.txt  (energy tasks only)
#
# Exit codes:
#   0  — compilation AND execution both succeeded
#   1  — compilation failed
#   2  — compilation succeeded but execution failed

# NOTE: intentionally no 'set -u'; Lmod init scripts access unset variables.
set -o pipefail

# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

TASK="$1"
SRC="$2"
OUTDIR="$3"
shift 3
RUN_ARGS=("$@")

mkdir -p "$OUTDIR"
: > "${OUTDIR}/build_stdout.log"
: > "${OUTDIR}/build_stderr.log"

{
    echo "=== polaris_worker.sh ==="
    echo "Task      : $TASK"
    echo "Date      : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Hostname  : $(hostname)"
    echo "Source    : $SRC"
    echo "Output dir: $OUTDIR"
    echo "Run args  : ${RUN_ARGS[*]:-<none>}"
    echo "Initial PATH: $PATH"
    echo ""
} >> "${OUTDIR}/build_stdout.log"

# ── oneAPI / icpx setup ────────────────────────────────────────────────────────
echo "=== oneAPI environment setup ===" >> "${OUTDIR}/build_stdout.log"

_LMOD_LOADED=false
for _lmod_init in \
    /etc/profile.d/z00_lmod.sh \
    /etc/profile.d/lmod.sh \
    /usr/share/lmod/lmod/init/bash \
    /usr/local/lmod/lmod/init/bash \
    /opt/cray/pe/lmod/lmod/init/bash
do
    [ -f "$_lmod_init" ] || continue
    echo "  [L1] sourcing Lmod init: $_lmod_init" >> "${OUTDIR}/build_stdout.log"
    # shellcheck source=/dev/null
    source "$_lmod_init" >> "${OUTDIR}/build_stdout.log" 2>&1 || true
    module load oneapi/release >> "${OUTDIR}/build_stdout.log" 2>&1 || true
    _LMOD_LOADED=true
    break
done
if [ "$_LMOD_LOADED" = false ]; then
    echo "  [L1] no Lmod init found — skipping" >> "${OUTDIR}/build_stdout.log"
fi

if ! command -v icpx &>/dev/null; then
    for _sv in /soft/compilers/oneapi/release/*/setvars.sh; do
        [ -f "$_sv" ] || continue
        # shellcheck source=/dev/null
        source "$_sv" --force >> "${OUTDIR}/build_stdout.log" 2>&1 || true
        break
    done
fi

if ! command -v icpx &>/dev/null; then
    for _icpx_bin in \
        /soft/compilers/oneapi/release/*/compiler/*/linux/bin/icpx \
        /soft/compilers/oneapi/release/*/compiler/*/linux/bin-llvm/icpx
    do
        [ -x "$_icpx_bin" ] || continue
        export PATH="$(dirname "$_icpx_bin"):${PATH}"
        break
    done
fi

# ── CUDA / nvcc setup ─────────────────────────────────────────────────────────
echo "=== CUDA environment setup ===" >> "${OUTDIR}/build_stdout.log"

if command -v module &>/dev/null; then
    module load cudatoolkit-standalone >> "${OUTDIR}/build_stdout.log" 2>&1 \
        || module load cudatoolkit >> "${OUTDIR}/build_stdout.log" 2>&1 \
        || module load cuda >> "${OUTDIR}/build_stdout.log" 2>&1 \
        || true
fi

if ! command -v nvcc &>/dev/null; then
    for _nvcc_bin in \
        /usr/local/cuda/bin/nvcc \
        /soft/compilers/cudatoolkit/cuda_*/bin/nvcc \
        /soft/compilers/cudatoolkit/*/bin/nvcc \
        /opt/cuda/bin/nvcc
    do
        [ -x "$_nvcc_bin" ] || continue
        export PATH="$(dirname "$_nvcc_bin"):${PATH}"
        break
    done
fi

# ── CUDA path discovery (for icpx --cuda-path) ────────────────────────────────
echo "=== CUDA path discovery ===" >> "${OUTDIR}/build_stdout.log"

CUDA_PATH="${CUDA_PATH:-${CUDATOOLKIT_HOME:-${CUDA_ROOT:-}}}"
if [ -z "$CUDA_PATH" ] || [ ! -f "${CUDA_PATH}/nvvm/libdevice/libdevice.10.bc" ]; then
    CUDA_PATH=""
    for _cuda_dir in \
        /usr/local/cuda \
        /soft/compilers/cudatoolkit/cuda_* \
        /soft/compilers/cudatoolkit/* \
        /opt/cuda
    do
        [ -f "${_cuda_dir}/nvvm/libdevice/libdevice.10.bc" ] || continue
        CUDA_PATH="$_cuda_dir"
        break
    done
fi

if [ -n "$CUDA_PATH" ]; then
    echo "  CUDA_PATH : $CUDA_PATH" >> "${OUTDIR}/build_stdout.log"
else
    echo "  CUDA_PATH : NOT FOUND" >> "${OUTDIR}/build_stdout.log"
fi

# ── nvidia-smi discovery ──────────────────────────────────────────────────────
NVIDIA_SMI="$(command -v nvidia-smi 2>/dev/null)"
if [ -z "$NVIDIA_SMI" ]; then
    for _smi_path in \
        /usr/bin/nvidia-smi \
        /usr/local/cuda/bin/nvidia-smi \
        /usr/local/bin/nvidia-smi
    do
        [ -x "$_smi_path" ] && { NVIDIA_SMI="$_smi_path"; break; }
    done
fi

{
    echo ""
    echo "=== Tool discovery ==="
    echo "  icpx      : $(command -v icpx 2>/dev/null || echo NOT FOUND)"
    echo "  nvcc      : $(command -v nvcc 2>/dev/null || echo NOT FOUND)"
    echo "  nvidia-smi: ${NVIDIA_SMI:-NOT FOUND}"
    echo "  Final PATH: $PATH"
    echo ""
} >> "${OUTDIR}/build_stdout.log"

# ── Task dispatch: set compiler and flags ──────────────────────────────────────
_CUDA_FLAG=""
[ -n "$CUDA_PATH" ] && _CUDA_FLAG="--cuda-path=${CUDA_PATH}"

case "$TASK" in
    cuda_energy)
        COMPILER="${CUDA_COMPILER:-nvcc}"
        FLAGS="${CUDA_FLAGS:--O3 -arch=sm_80 -std=c++17}"
        LDFLAGS="${CUDA_LDFLAGS:-}"
        MONITOR=nvidia_smi
        DEVICE_SELECTOR=""
        ;;
    translate|sycl_energy)
        COMPILER="${SYCL_COMPILER:-icpx}"
        _SYCL_DEFAULT="-O3 -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
-Xsycl-target-backend --cuda-gpu-arch=sm_80${_CUDA_FLAG:+ $_CUDA_FLAG}"
        FLAGS="${SYCL_FLAGS:-$_SYCL_DEFAULT}"
        LDFLAGS="${SYCL_LDFLAGS:-}"
        MONITOR=$([ "$TASK" = sycl_energy ] && echo nvidia_smi || echo none)
        DEVICE_SELECTOR="cuda:gpu"
        ;;
    *)
        {
            echo "ERROR: Unknown task '$TASK'."
            echo "  Valid tasks: translate, cuda_energy, sycl_energy"
        } | tee -a "${OUTDIR}/build_stdout.log" > "${OUTDIR}/build_stderr.log"
        echo -n "1" > "${OUTDIR}/build_rc.txt"
        exit 1
        ;;
esac

OUTEXE="${OUTDIR}/Main"

{
    echo "=== Compiling ($TASK) ==="
    echo "  Compiler : $COMPILER  ($(command -v "$COMPILER" 2>/dev/null || echo NOT FOUND))"
    echo "  Flags    : $FLAGS"
    echo "  LDFLAGS  : $LDFLAGS"
    echo ""
} >> "${OUTDIR}/build_stdout.log"

if ! command -v "$COMPILER" &>/dev/null; then
    {
        echo "ERROR: Compiler '$COMPILER' not found after all setup layers."
        echo "  See build_stdout.log for diagnosis."
    } | tee -a "${OUTDIR}/build_stdout.log" > "${OUTDIR}/build_stderr.log"
    echo -n "1" > "${OUTDIR}/build_rc.txt"
    exit 1
fi

"$COMPILER" --version >> "${OUTDIR}/build_stdout.log" 2>&1 || true

read -ra FLAGS_ARR   <<< "$FLAGS"
read -ra LDFLAGS_ARR <<< "$LDFLAGS"

"$COMPILER" "${FLAGS_ARR[@]}" "$SRC" -o "$OUTEXE" "${LDFLAGS_ARR[@]}" \
    >> "${OUTDIR}/build_stdout.log" \
    2> "${OUTDIR}/build_stderr.log"

BUILD_RC=$?
echo -n "$BUILD_RC" > "${OUTDIR}/build_rc.txt"

if [ "$BUILD_RC" -ne 0 ]; then
    echo "=== Compilation FAILED (exit code $BUILD_RC) ===" >> "${OUTDIR}/build_stdout.log"
    exit 1
fi

echo "=== Compilation OK ===" >> "${OUTDIR}/build_stdout.log"

# ── Run ────────────────────────────────────────────────────────────────────────
chmod +x "$OUTEXE"

[ -n "$DEVICE_SELECTOR" ] && export ONEAPI_DEVICE_SELECTOR="$DEVICE_SELECTOR"

MONITOR_PID=""
if [ "$MONITOR" = nvidia_smi ] && [ -n "$NVIDIA_SMI" ]; then
    : > "${OUTDIR}/power_samples.log"
    "$NVIDIA_SMI" dmon -s p -d 1 -i 0 >> "${OUTDIR}/power_samples.log" 2>&1 &
    MONITOR_PID=$!
    echo "  Power monitor PID: $MONITOR_PID" >> "${OUTDIR}/build_stdout.log"
fi

START_TS_MS=$(date +%s%3N)

"$OUTEXE" "${RUN_ARGS[@]}" \
    > "${OUTDIR}/run_stdout.log" \
    2> "${OUTDIR}/run_stderr.log"

RUN_RC=$?
END_TS_MS=$(date +%s%3N)

if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null
fi

echo -n "$RUN_RC" > "${OUTDIR}/run_rc.txt"

if [ "$MONITOR" = nvidia_smi ]; then
    _compute_energy "$START_TS_MS" "$END_TS_MS" \
        "${OUTDIR}/power_samples.log" "${OUTDIR}/energy_results.txt"
    {
        echo ""
        echo "=== Energy results ==="
        cat "${OUTDIR}/energy_results.txt" 2>/dev/null || echo "  (energy_results.txt not written)"
        echo ""
    } >> "${OUTDIR}/run_stdout.log"
fi

if [ "$RUN_RC" -ne 0 ]; then
    echo "" >> "${OUTDIR}/run_stdout.log"
    echo "=== Execution FAILED (exit code $RUN_RC) ===" >> "${OUTDIR}/run_stdout.log"
    exit 2
fi

echo "" >> "${OUTDIR}/run_stdout.log"
echo "=== Execution OK ===" >> "${OUTDIR}/run_stdout.log"
exit 0
