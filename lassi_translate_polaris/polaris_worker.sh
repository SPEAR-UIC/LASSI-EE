#!/bin/bash
# polaris_worker.sh — Compile and run a generated SYCL .cpp on a Polaris compute node.
#
# Called by run_lassi_polaris_v1.py via polaris.submit():
#   bash polaris_worker.sh <src.cpp> <outdir> [run_args...]
#
# Arguments:
#   $1        : Absolute path to the generated SYCL .cpp file (on Eagle)
#   $2        : Absolute path to the output directory (on Eagle)
#   $3, $4, …: Arguments to pass to the compiled executable (e.g., -m event)
#
# Environment variables (all optional — defaults target Polaris A100 / sm_80):
#   SYCL_COMPILER   Override compiler          (default: icpx, found via PATH)
#   SYCL_FLAGS      Override compilation flags (see DEFAULT_FLAGS below)
#   SYCL_LDFLAGS    Extra linker flags         (default: empty)
#
# Outputs written to <outdir>:
#   build_rc.txt       — compiler exit code (integer string)
#   build_stdout.log   — compilation stdout  (includes echo diagnostics)
#   build_stderr.log   — compilation stderr  (compiler errors/warnings)
#   run_rc.txt         — executable exit code (only written if build succeeded)
#   run_stdout.log     — executable stdout    (XSBench progress + checksum line)
#   run_stderr.log     — executable stderr
#   pbs_stdout.log  \  — written by PBS (via polaris.submit stdout_path/stderr_path)
#   pbs_stderr.log  /
#
# Exit codes:
#   0  — compilation AND execution both succeeded
#   1  — compilation failed  (run_rc.txt not written)
#   2  — compilation succeeded but execution failed

# NOTE: we intentionally do NOT use 'set -u' here.  Lmod init scripts and
# oneAPI setvars.sh access unset variables; sourcing them under 'set -u'
# silently kills the entire script before any output files are written.
set -o pipefail

SRC="$1"
OUTDIR="$2"
shift 2
RUN_ARGS=("$@")

# ── Create output directory and stub log files ─────────────────────────────────
# The stub files ensure Eagle can always find them even if the script exits early.
mkdir -p "$OUTDIR"
: > "${OUTDIR}/build_stdout.log"
: > "${OUTDIR}/build_stderr.log"

# ── Write initial log header (before any module loading that might fail) ───────
{
    echo "=== polaris_worker.sh ==="
    echo "Date      : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Hostname  : $(hostname)"
    echo "Source    : $SRC"
    echo "Output dir: $OUTDIR"
    echo "Run args  : ${RUN_ARGS[*]:-<none>}"
    echo "Initial PATH: $PATH"
    echo ""
} >> "${OUTDIR}/build_stdout.log"

# ── oneAPI / icpx setup ────────────────────────────────────────────────────────
# PBS batch jobs on Polaris start with a bare PATH=/bin:/usr/bin.
# The Lmod module system is NOT initialized automatically in non-login shells.
#
# We try three layers in sequence, stopping as soon as 'icpx' is found on PATH.
#
# Layer 1 — initialize Lmod from known init paths, then load oneapi/release.
# Layer 2 — source setvars.sh from any installed oneAPI version under /soft.
# Layer 3 — glob-search for the icpx binary across all version directories and
#            prepend its parent directory to PATH (handles version mismatch).

echo "=== oneAPI environment setup ===" >> "${OUTDIR}/build_stdout.log"

# Layer 1 — Lmod init + module load
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
if ! "$_LMOD_LOADED"; then
    echo "  [L1] no Lmod init found — skipping" >> "${OUTDIR}/build_stdout.log"
fi

# Layer 2 — source setvars.sh from the first found oneAPI release installation
if ! command -v icpx &>/dev/null; then
    _SV_FOUND=false
    for _sv in /soft/compilers/oneapi/release/*/setvars.sh; do
        [ -f "$_sv" ] || continue
        echo "  [L2] sourcing: $_sv" >> "${OUTDIR}/build_stdout.log"
        # shellcheck source=/dev/null
        source "$_sv" --force >> "${OUTDIR}/build_stdout.log" 2>&1 || true
        _SV_FOUND=true
        break
    done
    if ! "$_SV_FOUND"; then
        echo "  [L2] no setvars.sh found under /soft/compilers/oneapi/release/" \
            >> "${OUTDIR}/build_stdout.log"
    fi
fi

# Layer 3 — direct glob search for the icpx binary (handles any version string)
if ! command -v icpx &>/dev/null; then
    _ICPX_FOUND=false
    for _icpx_bin in \
        /soft/compilers/oneapi/release/*/compiler/*/linux/bin/icpx \
        /soft/compilers/oneapi/release/*/compiler/*/linux/bin-llvm/icpx
    do
        [ -x "$_icpx_bin" ] || continue
        _icpx_dir="$(dirname "$_icpx_bin")"
        echo "  [L3] found icpx at: $_icpx_bin — prepending to PATH" \
            >> "${OUTDIR}/build_stdout.log"
        export PATH="${_icpx_dir}:${PATH}"
        _ICPX_FOUND=true
        break
    done
    if ! "$_ICPX_FOUND"; then
        echo "  [L3] glob search found no icpx candidates" >> "${OUTDIR}/build_stdout.log"
        {
            echo "  /soft/compilers/oneapi/ contents:"
            ls /soft/compilers/oneapi/ 2>&1 || echo "    (not accessible)"
            echo ""
            echo "  All icpx glob hits:"
            ls /soft/compilers/oneapi/release/*/compiler/*/linux/bin/icpx \
               /soft/compilers/oneapi/release/*/compiler/*/linux/bin-llvm/icpx \
               2>/dev/null || echo "    (none)"
        } >> "${OUTDIR}/build_stdout.log"
    fi
fi

COMPILER="${SYCL_COMPILER:-icpx}"

# ── CUDA path discovery ────────────────────────────────────────────────────────
# icpx -fsycl-targets=nvptx64-nvidia-cuda needs CUDA's libdevice.
# Try (in order): env vars set by modules, then common Polaris filesystem paths.

echo "=== CUDA path discovery ===" >> "${OUTDIR}/build_stdout.log"

# Try loading the cudatoolkit module if Lmod is available (sets CUDA_PATH / CUDATOOLKIT_HOME)
if command -v module &>/dev/null; then
    module load cudatoolkit-standalone 2>/dev/null \
        || module load cudatoolkit 2>/dev/null \
        || module load cuda 2>/dev/null \
        || true
fi

CUDA_PATH="${CUDA_PATH:-${CUDATOOLKIT_HOME:-${CUDA_ROOT:-}}}"

# Glob-search common Polaris paths if still unset or libdevice not present there
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
    _CUDA_FLAG="--cuda-path=${CUDA_PATH}"
else
    echo "  CUDA_PATH : NOT FOUND" >> "${OUTDIR}/build_stdout.log"
    {
        echo "  /soft/compilers/cudatoolkit/ contents:"
        ls /soft/compilers/cudatoolkit/ 2>&1 || echo "    (not accessible)"
    } >> "${OUTDIR}/build_stdout.log"
    _CUDA_FLAG=""
fi

DEFAULT_FLAGS="-O3 -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80${_CUDA_FLAG:+ $_CUDA_FLAG}"
FLAGS="${SYCL_FLAGS:-$DEFAULT_FLAGS}"
LDFLAGS="${SYCL_LDFLAGS:-}"

OUTEXE="${OUTDIR}/Main"

{
    echo ""
    echo "  Compiler  : $COMPILER  ($(command -v "$COMPILER" 2>/dev/null || echo 'NOT FOUND'))"
    echo "  Flags     : $FLAGS"
    echo "  LDFLAGS   : $LDFLAGS"
    echo "  Final PATH: $PATH"
    echo ""
} >> "${OUTDIR}/build_stdout.log"

# ── Verify compiler is available ──────────────────────────────────────────────
if ! command -v "$COMPILER" &>/dev/null; then
    {
        echo "ERROR: Compiler '$COMPILER' not found after all setup layers."
        echo "  See build_stdout.log for diagnosis."
    } | tee -a "${OUTDIR}/build_stdout.log" > "${OUTDIR}/build_stderr.log"
    echo -n "1" > "${OUTDIR}/build_rc.txt"
    exit 1
fi

# Confirm version for diagnostics
"$COMPILER" --version >> "${OUTDIR}/build_stdout.log" 2>&1 || true

# ── Compile ────────────────────────────────────────────────────────────────────
echo "" >> "${OUTDIR}/build_stdout.log"
echo "=== Compiling ===" >> "${OUTDIR}/build_stdout.log"

# Split flags string into an array so flags with spaces are handled safely.
read -ra FLAGS_ARR   <<< "$FLAGS"
read -ra LDFLAGS_ARR <<< "$LDFLAGS"

"$COMPILER" "${FLAGS_ARR[@]}" "$SRC" -o "$OUTEXE" "${LDFLAGS_ARR[@]}" \
    >> "${OUTDIR}/build_stdout.log" \
    2> "${OUTDIR}/build_stderr.log"

BUILD_RC=$?
echo -n "$BUILD_RC" > "${OUTDIR}/build_rc.txt"

if [ "$BUILD_RC" -ne 0 ]; then
    echo "" >> "${OUTDIR}/build_stdout.log"
    echo "=== Compilation FAILED (exit code $BUILD_RC) ===" >> "${OUTDIR}/build_stdout.log"
    exit 1
fi

echo "=== Compilation OK ===" >> "${OUTDIR}/build_stdout.log"

# ── Run ────────────────────────────────────────────────────────────────────────
chmod +x "$OUTEXE"

# Required for SYCL to discover the CUDA backend on Polaris A100 GPUs.
export ONEAPI_DEVICE_SELECTOR=cuda:gpu

# Run the compiled executable with the provided arguments.
# Stdout captures XSBench progress and the critical "Verification checksum:" line.
"$OUTEXE" "${RUN_ARGS[@]}" \
    > "${OUTDIR}/run_stdout.log" \
    2> "${OUTDIR}/run_stderr.log"

RUN_RC=$?
echo -n "$RUN_RC" > "${OUTDIR}/run_rc.txt"

if [ "$RUN_RC" -ne 0 ]; then
    echo "" >> "${OUTDIR}/run_stdout.log"
    echo "=== Execution FAILED (exit code $RUN_RC) ===" >> "${OUTDIR}/run_stdout.log"
    exit 2
fi

echo "" >> "${OUTDIR}/run_stdout.log"
echo "=== Execution OK ===" >> "${OUTDIR}/run_stdout.log"
exit 0
