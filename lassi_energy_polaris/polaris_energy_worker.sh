#!/bin/bash
# polaris_energy_worker.sh — Compile a CUDA .cu file with nvcc on a Polaris compute
# node and run it with GPU power monitoring to produce energy consumption metrics.
#
# Called by run_lassi_energy_polaris_v1.py via polaris.submit():
#   bash polaris_energy_worker.sh <src.cu> <outdir> [run_args...]
#
# Arguments:
#   $1        : Absolute path to the CUDA .cu file (on Eagle)
#   $2        : Absolute path to the output directory (on Eagle)
#   $3, $4, …: Arguments to pass to the compiled executable (e.g., -m event)
#
# Environment variables (all optional — defaults target Polaris A100 / sm_80):
#   CUDA_COMPILER   Override compiler          (default: nvcc, found via PATH)
#   CUDA_FLAGS      Override compilation flags (default: -O3 -arch=sm_80 -std=c++17)
#   CUDA_LDFLAGS    Extra linker flags         (default: empty)
#
# Outputs written to <outdir>:
#   build_rc.txt       — compiler exit code (integer string)
#   build_stdout.log   — compilation stdout  (includes echo diagnostics)
#   build_stderr.log   — compilation stderr  (compiler errors/warnings)
#   run_rc.txt         — executable exit code (only written if build succeeded)
#   run_stdout.log     — executable stdout    (XSBench progress + checksum line)
#   run_stderr.log     — executable stderr
#   power_samples.log  — raw nvidia-smi dmon output captured during execution
#   energy_results.txt — JSON: {energy_joules, avg_power_w, duration_s, num_samples}
#   pbs_stdout.log  \  — written by PBS (via polaris.submit stdout_path/stderr_path)
#   pbs_stderr.log  /
#
# Exit codes:
#   0  — compilation AND execution both succeeded
#   1  — compilation failed  (run_rc.txt not written)
#   2  — compilation succeeded but execution failed

# NOTE: intentionally no 'set -u'; Lmod init scripts access unset variables and
# sourcing them under 'set -u' kills the script before output files are written.
set -o pipefail

SRC="$1"
OUTDIR="$2"
shift 2
RUN_ARGS=("$@")

# ── Create output directory and stub log files ─────────────────────────────────
# Stub files ensure Eagle can always find them even if the script exits early.
mkdir -p "$OUTDIR"
: > "${OUTDIR}/build_stdout.log"
: > "${OUTDIR}/build_stderr.log"
: > "${OUTDIR}/power_samples.log"

# ── Write initial log header ───────────────────────────────────────────────────
{
    echo "=== polaris_energy_worker.sh ==="
    echo "Date      : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Hostname  : $(hostname)"
    echo "Source    : $SRC"
    echo "Output dir: $OUTDIR"
    echo "Run args  : ${RUN_ARGS[*]:-<none>}"
    echo "Initial PATH: $PATH"
    echo ""
} >> "${OUTDIR}/build_stdout.log"

# ── CUDA / nvcc setup ─────────────────────────────────────────────────────────
# PBS batch jobs on Polaris start with a bare PATH=/bin:/usr/bin.
# We try two layers to get nvcc on PATH before falling through to a compiler error.
#
# Layer 1 — initialize Lmod from known init paths, then load cudatoolkit module.
# Layer 2 — glob-search for nvcc binary across common Polaris filesystem paths.

echo "=== CUDA environment setup ===" >> "${OUTDIR}/build_stdout.log"

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
    module load cudatoolkit-standalone >> "${OUTDIR}/build_stdout.log" 2>&1 \
        || module load cudatoolkit >> "${OUTDIR}/build_stdout.log" 2>&1 \
        || module load cuda >> "${OUTDIR}/build_stdout.log" 2>&1 \
        || true
    _LMOD_LOADED=true
    break
done
if ! "$_LMOD_LOADED"; then
    echo "  [L1] no Lmod init found — skipping" >> "${OUTDIR}/build_stdout.log"
fi

# Layer 2 — glob-search for the nvcc binary (handles any version string)
if ! command -v nvcc &>/dev/null; then
    _NVCC_FOUND=false
    for _nvcc_bin in \
        /usr/local/cuda/bin/nvcc \
        /soft/compilers/cudatoolkit/cuda_*/bin/nvcc \
        /soft/compilers/cudatoolkit/*/bin/nvcc \
        /opt/cuda/bin/nvcc
    do
        [ -x "$_nvcc_bin" ] || continue
        _nvcc_dir="$(dirname "$_nvcc_bin")"
        echo "  [L2] found nvcc at: $_nvcc_bin — prepending to PATH" \
            >> "${OUTDIR}/build_stdout.log"
        export PATH="${_nvcc_dir}:${PATH}"
        _NVCC_FOUND=true
        break
    done
    if ! "$_NVCC_FOUND"; then
        echo "  [L2] glob search found no nvcc candidates" >> "${OUTDIR}/build_stdout.log"
        {
            echo "  /soft/compilers/cudatoolkit/ contents:"
            ls /soft/compilers/cudatoolkit/ 2>&1 || echo "    (not accessible)"
            echo ""
            echo "  All nvcc glob hits:"
            ls /soft/compilers/cudatoolkit/cuda_*/bin/nvcc \
               /soft/compilers/cudatoolkit/*/bin/nvcc \
               /usr/local/cuda/bin/nvcc \
               2>/dev/null || echo "    (none)"
        } >> "${OUTDIR}/build_stdout.log"
    fi
fi

COMPILER="${CUDA_COMPILER:-nvcc}"
DEFAULT_FLAGS="-O3 -arch=sm_80 -std=c++17"
FLAGS="${CUDA_FLAGS:-$DEFAULT_FLAGS}"
LDFLAGS="${CUDA_LDFLAGS:-}"

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

# Log compiler version for reproducibility
"$COMPILER" --version >> "${OUTDIR}/build_stdout.log" 2>&1 || true

# ── Compile ────────────────────────────────────────────────────────────────────
echo "" >> "${OUTDIR}/build_stdout.log"
echo "=== Compiling ===" >> "${OUTDIR}/build_stdout.log"

# Split flags strings into arrays so flags with spaces are handled safely.
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

# ── Locate nvidia-smi for power monitoring ────────────────────────────────────
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
    echo "=== GPU / nvidia-smi discovery ==="
    echo "  nvidia-smi: ${NVIDIA_SMI:-NOT FOUND}"
    if [ -n "$NVIDIA_SMI" ]; then
        "$NVIDIA_SMI" -L 2>&1 || true
        echo ""
        "$NVIDIA_SMI" --query-gpu=gpu_name,power.limit,memory.total \
            --format=csv,noheader 2>&1 | head -4 || true
    fi
    echo ""
} >> "${OUTDIR}/build_stdout.log"

# ── Run with GPU power monitoring ─────────────────────────────────────────────
chmod +x "$OUTEXE"

# Start nvidia-smi dmon in background if available.
# -s p  : power columns only
# -d 1  : 1-second sampling interval
# -i 0  : GPU 0 (primary A100 on Polaris compute node)
MONITOR_PID=""
if [ -n "$NVIDIA_SMI" ]; then
    "$NVIDIA_SMI" dmon -s p -d 1 -i 0 >> "${OUTDIR}/power_samples.log" 2>&1 &
    MONITOR_PID=$!
    echo "  Power monitor PID: $MONITOR_PID" >> "${OUTDIR}/build_stdout.log"
fi

# Record execution window timestamps (millisecond precision).
START_TS_MS=$(date +%s%3N)

"$OUTEXE" "${RUN_ARGS[@]}" \
    > "${OUTDIR}/run_stdout.log" \
    2> "${OUTDIR}/run_stderr.log"

RUN_RC=$?
END_TS_MS=$(date +%s%3N)

# Terminate the power monitor.
if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null
fi

echo -n "$RUN_RC" > "${OUTDIR}/run_rc.txt"

# ── Compute energy from power samples ─────────────────────────────────────────
# Parse the nvidia-smi dmon log and integrate power × time.
# Writes energy_results.txt as JSON so the orchestrator can read it via Eagle API.
python3 - "$START_TS_MS" "$END_TS_MS" \
         "${OUTDIR}/power_samples.log" \
         "${OUTDIR}/energy_results.txt" << 'PYEOF'
import sys, json

start_ms  = int(sys.argv[1])
end_ms    = int(sys.argv[2])
power_log = sys.argv[3]
out_file  = sys.argv[4]

duration_s = max((end_ms - start_ms) / 1000.0, 0.0)
samples = []

# nvidia-smi dmon -s p output format (two header comment lines, then data rows):
#   # gpu   pwr  gtemp  mtemp    sm   mem   enc   dec   jpg   ofa
#   # Idx     W      C      C     %     %     %     %     %     %
#       0   145     47      0    82    30     0     0     0     0
#
# With -s p (power only), columns are: gpu_idx, pwr_W
# Power values of 0 or negative mean the driver couldn't read the sensor — skip them.
try:
    with open(power_log) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pwr = float(parts[1])
                    if pwr > 0:
                        samples.append(pwr)
                except (ValueError, IndexError):
                    pass
except Exception:
    samples = []

if samples and duration_s > 0:
    avg_power_w   = sum(samples) / len(samples)
    # Trapezoidal integration: assume samples are spaced evenly over the execution window.
    dt            = duration_s / len(samples)
    energy_joules = sum(samples) * dt
else:
    avg_power_w   = -1.0
    energy_joules = -1.0

result = {
    "energy_joules": round(energy_joules, 3),
    "avg_power_w":   round(avg_power_w, 3),
    "duration_s":    round(duration_s, 3),
    "num_samples":   len(samples),
}

with open(out_file, "w") as fh:
    json.dump(result, fh)
PYEOF

{
    echo ""
    echo "=== Energy results ==="
    cat "${OUTDIR}/energy_results.txt" 2>/dev/null || echo "  (energy_results.txt not written)"
    echo ""
} >> "${OUTDIR}/run_stdout.log"

if [ "$RUN_RC" -ne 0 ]; then
    echo "" >> "${OUTDIR}/run_stdout.log"
    echo "=== Execution FAILED (exit code $RUN_RC) ===" >> "${OUTDIR}/run_stdout.log"
    exit 2
fi

echo "=== Execution OK ===" >> "${OUTDIR}/run_stdout.log"
exit 0
