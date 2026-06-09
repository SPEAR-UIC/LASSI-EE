#!/bin/bash
# _common.sh — Shared helpers sourced by all LASSI-EE worker scripts.
# Do not execute directly. Intentionally no 'set -u'; Lmod init scripts
# routinely access unset variables.

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup (Lmod / oneAPI / CUDA / Intel GPU tooling)
# ─────────────────────────────────────────────────────────────────────────────

# _setup_lmod_oneapi <log_file>
# Source the first Lmod init we find, then load the oneAPI module.
_setup_lmod_oneapi() {
    local _log="$1"
    local _loaded=false
    for _init in \
        /etc/profile.d/z00_lmod.sh \
        /etc/profile.d/lmod.sh \
        /usr/share/lmod/lmod/init/bash \
        /usr/local/lmod/lmod/init/bash \
        /opt/cray/pe/lmod/lmod/init/bash
    do
        [ -f "$_init" ] || continue
        echo "  [Lmod] sourcing $_init" >> "$_log"
        # shellcheck source=/dev/null
        source "$_init" >> "$_log" 2>&1 || true
        # Polaris compute nodes need /soft/modulefiles on MODULEPATH (it's
        # not there by default in non-interactive child shells, even when
        # the parent shell `module use`'d it earlier).
        module use /soft/modulefiles >> "$_log" 2>&1 || true
        module load oneapi/release >> "$_log" 2>&1 \
            || module load oneapi   >> "$_log" 2>&1 \
            || true
        _loaded=true
        break
    done
    [ "$_loaded" = false ] && echo "  [Lmod] no init found — skipping" >> "$_log"
}

# _resolve_icpx <log_file>
# Try setvars.sh and direct binary discovery if icpx is not on PATH yet.
_resolve_icpx() {
    local _log="$1"
    command -v icpx &>/dev/null && return 0
    for _sv in \
        /soft/compilers/oneapi/release/*/setvars.sh \
        /opt/intel/oneapi/setvars.sh \
        /usr/local/intel/oneapi/setvars.sh
    do
        [ -f "$_sv" ] || continue
        # shellcheck source=/dev/null
        source "$_sv" --force >> "$_log" 2>&1 || true
        break
    done
    command -v icpx &>/dev/null && return 0
    for _bin in \
        /soft/compilers/oneapi/release/*/compiler/*/linux/bin/icpx \
        /soft/compilers/oneapi/release/*/compiler/*/linux/bin-llvm/icpx \
        /opt/intel/oneapi/compiler/*/linux/bin/icpx
    do
        [ -x "$_bin" ] || continue
        export PATH="$(dirname "$_bin"):${PATH}"
        return 0
    done
}

# _setup_cuda <log_file>
# Load CUDA via module + discover nvcc via fallback paths.  Also forces
# gcc-native/12 because nvcc 12.2 can't parse gcc-14's C++ stdlib headers
# (the auto-swap that happens when conda is loaded brings in gcc-14).
_setup_cuda() {
    local _log="$1"
    if command -v module &>/dev/null; then
        module use /soft/modulefiles >> "$_log" 2>&1 || true
        module load cudatoolkit-standalone >> "$_log" 2>&1 \
            || module load cudatoolkit     >> "$_log" 2>&1 \
            || module load cuda            >> "$_log" 2>&1 \
            || true
        # Downgrade gcc-native to a version nvcc 12.2 can parse.
        module load gcc-native/12 >> "$_log" 2>&1 || true
    fi
    command -v nvcc &>/dev/null && return 0
    for _bin in \
        /usr/local/cuda/bin/nvcc \
        /soft/compilers/cudatoolkit/cuda_*/bin/nvcc \
        /soft/compilers/cudatoolkit/*/bin/nvcc \
        /opt/cuda/bin/nvcc
    do
        [ -x "$_bin" ] || continue
        export PATH="$(dirname "$_bin"):${PATH}"
        return 0
    done
}

# _resolve_cuda_path <log_file>
# Locate a CUDA toolkit directory containing libdevice.10.bc (needed by
# icpx --cuda-path=...). Exports CUDA_PATH on success.
_resolve_cuda_path() {
    local _log="$1"
    CUDA_PATH="${CUDA_PATH:-${CUDATOOLKIT_HOME:-${CUDA_ROOT:-}}}"
    if [ -z "$CUDA_PATH" ] || [ ! -f "${CUDA_PATH}/nvvm/libdevice/libdevice.10.bc" ]; then
        CUDA_PATH=""
        for _dir in \
            /usr/local/cuda \
            /soft/compilers/cudatoolkit/cuda_* \
            /soft/compilers/cudatoolkit/* \
            /opt/cuda
        do
            [ -f "${_dir}/nvvm/libdevice/libdevice.10.bc" ] || continue
            CUDA_PATH="$_dir"
            break
        done
    fi
    export CUDA_PATH
    echo "  CUDA_PATH: ${CUDA_PATH:-NOT FOUND}" >> "$_log"
}

# _resolve_nvidia_smi
# Sets the global NVIDIA_SMI variable.
_resolve_nvidia_smi() {
    NVIDIA_SMI="$(command -v nvidia-smi 2>/dev/null)"
    [ -n "$NVIDIA_SMI" ] && return 0
    for _bin in /usr/bin/nvidia-smi /usr/local/cuda/bin/nvidia-smi /usr/local/bin/nvidia-smi; do
        [ -x "$_bin" ] && { NVIDIA_SMI="$_bin"; return 0; }
    done
}

# _resolve_xpu_smi <log_file>
# Aurora-only. Loads intel-gpu-tools / level-zero modules and discovers xpu-smi.
# Sets the global XPU_SMI variable.
_resolve_xpu_smi() {
    local _log="$1"
    if command -v module &>/dev/null; then
        module load intel-gpu-tools 2>/dev/null || true
        module load level-zero      2>/dev/null || true
    fi
    XPU_SMI="$(command -v xpu-smi 2>/dev/null)"
    [ -n "$XPU_SMI" ] && return 0
    for _bin in /usr/bin/xpu-smi /usr/local/bin/xpu-smi /opt/intel/xpumanager/bin/xpu-smi; do
        [ -x "$_bin" ] && { XPU_SMI="$_bin"; return 0; }
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Power monitors (set MONITOR_PID; consumed by _stop_power_monitor)
# ─────────────────────────────────────────────────────────────────────────────

# _start_nvidia_power_monitor <outdir>
# Requires $NVIDIA_SMI to be set. No-op (MONITOR_PID="") if it isn't.
_start_nvidia_power_monitor() {
    local _outdir="$1"
    MONITOR_PID=""
    [ -z "$NVIDIA_SMI" ] && return 0
    : > "${_outdir}/power_samples.log"
    "$NVIDIA_SMI" dmon -s p -d 1 -i 0 >> "${_outdir}/power_samples.log" 2>&1 &
    MONITOR_PID=$!
}

_stop_power_monitor() {
    [ -z "$MONITOR_PID" ] && return 0
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null
    MONITOR_PID=""
}

# ─────────────────────────────────────────────────────────────────────────────
# Energy / result-ready file writers
# ─────────────────────────────────────────────────────────────────────────────

# _compute_energy <start_ms> <end_ms> <power_samples.log> <energy_results.txt>
_compute_energy() {
    local _start_ms="$1" _end_ms="$2" _power_log="$3" _out="$4"
    python3 - "$_start_ms" "$_end_ms" "$_power_log" "$_out" << 'ENERGY_PYEOF'
import sys, json

start_ms  = int(sys.argv[1])
end_ms    = int(sys.argv[2])
power_log = sys.argv[3]
out_file  = sys.argv[4]

duration_s = max((end_ms - start_ms) / 1000.0, 0.0)
samples = []

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
    dt            = duration_s / len(samples)
    energy_joules = sum(samples) * dt
else:
    avg_power_w   = -1.0
    energy_joules = -1.0

json.dump({
    "energy_joules": round(energy_joules, 3),
    "avg_power_w":   round(avg_power_w, 3),
    "duration_s":    round(duration_s, 3),
    "num_samples":   len(samples),
}, open(out_file, "w"))
ENERGY_PYEOF
}

# _write_result_ready <build_rc> <run_rc> <outdir> <ipc_dir>
# Atomic write via tmp + mv so the Python poller never sees a partial file.
_write_result_ready() {
    local _build_rc="$1" _run_rc="$2" _outdir="$3" _ipc_dir="$4"
    python3 - "$_build_rc" "$_run_rc" "$_outdir" \
              "${_ipc_dir}/result_ready.json.tmp" << 'RESULT_PYEOF'
import json, sys
json.dump(
    {"build_rc": int(sys.argv[1]), "run_rc": int(sys.argv[2]), "outdir": sys.argv[3]},
    open(sys.argv[4], "w"),
)
RESULT_PYEOF
    mv "${_ipc_dir}/result_ready.json.tmp" "${_ipc_dir}/result_ready.json"
}

# ─────────────────────────────────────────────────────────────────────────────
# _compile_and_run — the core compile + run + energy sequence.
#
# Usage:
#   _compile_and_run <compiler> <flags> <ldflags> <src> <outdir> \
#                    <monitor> <device_selector> <use_eval> \
#                    [run_args...]
#
#   <monitor>          : none | nvidia_smi
#   <device_selector>  : value for ONEAPI_DEVICE_SELECTOR (empty for nvcc)
#   <use_eval>         : 0 | 1   — Aurora's `-Xs "-device pvc"` needs eval
#                                  to preserve the embedded quotes
#
# Writes into <outdir>:
#   build_rc.txt, build_stdout.log, build_stderr.log
#   run_rc.txt,   run_stdout.log,   run_stderr.log
#   power_samples.log, energy_results.txt (energy monitors only)
#
# Always returns 0; exit codes live in the *_rc.txt files.
# ─────────────────────────────────────────────────────────────────────────────
_compile_and_run() {
    local _compiler="$1" _flags="$2" _ldflags="$3" _src="$4" _outdir="$5"
    local _monitor="$6" _device_selector="$7" _use_eval="$8"
    shift 8
    local _run_args=("$@")

    local _outexe="${_outdir}/Main"
    local _build_stdout="${_outdir}/build_stdout.log"
    local _build_stderr="${_outdir}/build_stderr.log"

    {
        echo "=== Compiling ==="
        echo "  Compiler: $_compiler  ($(command -v "$_compiler" 2>/dev/null || echo NOT FOUND))"
        echo "  Flags   : $_flags"
        echo "  LDFLAGS : $_ldflags"
        echo ""
    } >> "$_build_stdout"

    if ! command -v "$_compiler" &>/dev/null; then
        echo "ERROR: $_compiler not found." \
            | tee -a "$_build_stdout" > "$_build_stderr"
        echo -n "1" > "${_outdir}/build_rc.txt"
        return 0
    fi

    "$_compiler" --version >> "$_build_stdout" 2>&1 || true

    local _build_rc
    if [ "$_use_eval" = "1" ]; then
        eval "$_compiler $_flags \"$_src\" -o \"$_outexe\" $_ldflags" \
            >> "$_build_stdout" 2> "$_build_stderr"
        _build_rc=$?
    else
        local _flags_arr _ldflags_arr
        read -ra _flags_arr   <<< "$_flags"
        read -ra _ldflags_arr <<< "$_ldflags"
        "$_compiler" "${_flags_arr[@]}" "$_src" -o "$_outexe" "${_ldflags_arr[@]}" \
            >> "$_build_stdout" 2> "$_build_stderr"
        _build_rc=$?
    fi
    echo -n "$_build_rc" > "${_outdir}/build_rc.txt"

    if [ "$_build_rc" -ne 0 ]; then
        echo "=== Compilation FAILED (exit $_build_rc) ===" >> "$_build_stdout"
        return 0
    fi
    echo "=== Compilation OK ===" >> "$_build_stdout"

    # ── Run ────────────────────────────────────────────────────────────────
    chmod +x "$_outexe"
    [ -n "$_device_selector" ] && export ONEAPI_DEVICE_SELECTOR="$_device_selector"

    case "$_monitor" in
        nvidia_smi) _start_nvidia_power_monitor "$_outdir" ;;
        *)          MONITOR_PID="" ;;
    esac

    local _start_ms
    _start_ms=$(date +%s%3N)
    "$_outexe" "${_run_args[@]}" \
        > "${_outdir}/run_stdout.log" \
        2> "${_outdir}/run_stderr.log"
    local _run_rc=$?
    local _end_ms
    _end_ms=$(date +%s%3N)

    _stop_power_monitor
    echo -n "$_run_rc" > "${_outdir}/run_rc.txt"

    # Always write duration; energy/power fields are -1 when no monitor ran
    # (the Python parser handles the missing power_samples.log gracefully).
    _compute_energy "$_start_ms" "$_end_ms" \
        "${_outdir}/power_samples.log" "${_outdir}/energy_results.txt" 2>/dev/null || true
    if [ "$_monitor" != "none" ]; then
        {
            echo ""
            echo "=== Energy results ==="
            cat "${_outdir}/energy_results.txt" 2>/dev/null \
                || echo "  (energy_results.txt not written)"
        } >> "${_outdir}/run_stdout.log"
    fi

    {
        echo ""
        if [ "$_run_rc" -ne 0 ]; then
            echo "=== Execution FAILED (exit $_run_rc) ==="
        else
            echo "=== Execution OK ==="
        fi
    } >> "${_outdir}/run_stdout.log"
}
