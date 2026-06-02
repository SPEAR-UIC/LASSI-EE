#!/bin/bash
# aurora_worker.sh — Compile and run SYCL code on an Aurora compute node
# (Intel Data Center GPU Max / Ponte Vecchio, icpx + Level Zero).
#
# Called by run_lassi_jobs.py via _submit_job():
#   bash aurora_worker.sh <task> <src> <outdir> [run_args...]
#
# Arguments:
#   $1        : Task: translate | sycl_energy
#   $2        : Absolute path to the SYCL .cpp file (on Flare)
#   $3        : Absolute path to the output directory (on Flare)
#   $4, $5, …: Arguments to pass to the compiled executable
#
# Environment variables (all optional):
#   SYCL_COMPILER   Override compiler (default: icpx)
#   SYCL_FLAGS      Override compilation flags (see DEFAULT_FLAGS below)
#   SYCL_LDFLAGS    Extra linker flags
#
# NOTE: compilation uses eval to correctly pass -Xs "-device pvc" (embedded quotes).
#
# Outputs written to <outdir>:
#   build_rc.txt, build_stdout.log, build_stderr.log
#   run_rc.txt, run_stdout.log, run_stderr.log
#   power_samples.log, energy_results.txt  (sycl_energy task only)
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
    echo "=== aurora_worker.sh ==="
    echo "Task      : $TASK"
    echo "Date      : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Hostname  : $(hostname)"
    echo "Source    : $SRC"
    echo "Output dir: $OUTDIR"
    echo "Run args  : ${RUN_ARGS[*]:-<none>}"
    echo "Initial PATH: $PATH"
    echo ""
} >> "${OUTDIR}/build_stdout.log"

# ── oneAPI / icpx setup (Aurora) ──────────────────────────────────────────────
echo "=== oneAPI environment setup (Aurora) ===" >> "${OUTDIR}/build_stdout.log"

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
    module load oneapi/release >> "${OUTDIR}/build_stdout.log" 2>&1 \
        || module load oneapi >> "${OUTDIR}/build_stdout.log" 2>&1 \
        || true
    _LMOD_LOADED=true
    break
done
if [ "$_LMOD_LOADED" = false ]; then
    echo "  [L1] no Lmod init found — skipping" >> "${OUTDIR}/build_stdout.log"
fi

if ! command -v icpx &>/dev/null; then
    for _sv in \
        /soft/compilers/oneapi/release/*/setvars.sh \
        /opt/intel/oneapi/setvars.sh \
        /usr/local/intel/oneapi/setvars.sh
    do
        [ -f "$_sv" ] || continue
        echo "  [L2] sourcing: $_sv" >> "${OUTDIR}/build_stdout.log"
        # shellcheck source=/dev/null
        source "$_sv" --force >> "${OUTDIR}/build_stdout.log" 2>&1 || true
        break
    done
fi

if ! command -v icpx &>/dev/null; then
    for _icpx_bin in \
        /soft/compilers/oneapi/release/*/compiler/*/linux/bin/icpx \
        /soft/compilers/oneapi/release/*/compiler/*/linux/bin-llvm/icpx \
        /opt/intel/oneapi/compiler/*/linux/bin/icpx
    do
        [ -x "$_icpx_bin" ] || continue
        export PATH="$(dirname "$_icpx_bin"):${PATH}"
        break
    done
fi

COMPILER="${SYCL_COMPILER:-icpx}"

# ── Intel GPU / xpu-smi discovery ─────────────────────────────────────────────
echo "=== Intel GPU / xpu-smi discovery ===" >> "${OUTDIR}/build_stdout.log"

if command -v module &>/dev/null; then
    module load intel-gpu-tools 2>/dev/null || true
    module load level-zero 2>/dev/null || true
fi

XPU_SMI="$(command -v xpu-smi 2>/dev/null)"
if [ -z "$XPU_SMI" ]; then
    for _xpu_path in \
        /usr/bin/xpu-smi \
        /usr/local/bin/xpu-smi \
        /opt/intel/xpumanager/bin/xpu-smi
    do
        [ -x "$_xpu_path" ] && { XPU_SMI="$_xpu_path"; break; }
    done
fi

{
    echo "  icpx   : $(command -v icpx 2>/dev/null || echo NOT FOUND)"
    echo "  xpu-smi: ${XPU_SMI:-NOT FOUND}"
    echo "  Final PATH: $PATH"
    if [ -n "$XPU_SMI" ]; then
        "$XPU_SMI" discovery 2>&1 | head -20 || true
    fi
    echo ""
} >> "${OUTDIR}/build_stdout.log"

# ── Task dispatch: set flags and monitor ──────────────────────────────────────
# -fsycl-targets=spir64_gen + -Xs "-device pvc" targets Intel PVC AOT.
_SYCL_DEFAULT="-O3 -std=c++17 -fsycl -fsycl-targets=spir64_gen -Xs \"-device pvc\""

case "$TASK" in
    translate|sycl_energy)
        FLAGS="${SYCL_FLAGS:-$_SYCL_DEFAULT}"
        LDFLAGS="${SYCL_LDFLAGS:-}"
        MONITOR=$([ "$TASK" = sycl_energy ] && echo xpu_smi || echo none)
        ;;
    *)
        {
            echo "ERROR: Unknown task '$TASK'."
            echo "  Valid tasks: translate, sycl_energy"
        } | tee -a "${OUTDIR}/build_stdout.log" > "${OUTDIR}/build_stderr.log"
        echo -n "1" > "${OUTDIR}/build_rc.txt"
        exit 1
        ;;
esac

OUTEXE="${OUTDIR}/Main"

{
    echo ""
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

# NOTE: eval is required to correctly expand the embedded quotes in -Xs "-device pvc".
eval "$COMPILER $FLAGS \"$SRC\" -o \"$OUTEXE\" $LDFLAGS" \
    >> "${OUTDIR}/build_stdout.log" \
    2> "${OUTDIR}/build_stderr.log"

BUILD_RC=$?
echo -n "$BUILD_RC" > "${OUTDIR}/build_rc.txt"

if [ "$BUILD_RC" -ne 0 ]; then
    echo "=== Compilation FAILED (exit code $BUILD_RC) ===" >> "${OUTDIR}/build_stdout.log"
    exit 1
fi

echo "=== Compilation OK ===" >> "${OUTDIR}/build_stdout.log"

# ── Run with Level Zero backend ────────────────────────────────────────────────
chmod +x "$OUTEXE"

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

MONITOR_PID=""
if [ "$MONITOR" = xpu_smi ] && [ -n "$XPU_SMI" ]; then
    : > "${OUTDIR}/power_samples.log"
    # Poll xpu-smi at 1-second intervals; emit "gpu_idx power_W" lines to match
    # the nvidia-smi dmon format consumed by _compute_energy.
    (
        while true; do
            pwr=$("$XPU_SMI" stats -d 0 2>/dev/null \
                  | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tile = data[0] if isinstance(data, list) else data
    pwr_keys = [k for k in tile if 'power' in k.lower() or 'Power' in k]
    val = tile[pwr_keys[0]] if pwr_keys else -1
    print(float(val))
except Exception:
    print(-1)
" 2>/dev/null || echo "-1")
            echo "0 $pwr" >> "${OUTDIR}/power_samples.log"
            sleep 1
        done
    ) &
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

if [ "$MONITOR" = xpu_smi ]; then
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
