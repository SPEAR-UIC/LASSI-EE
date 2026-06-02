#!/bin/bash
# polaris_persistent_worker.sh — Long-running PBS daemon for LASSI-EE on Polaris.
#
# Runs continuously on a Polaris compute node (NVIDIA A100 / sm_80), processing
# compile-and-run requests delivered via signal files on Eagle filesystem.
# Handles cuda_energy, sycl_energy, and translate task types so a single PBS job
# covers the entire self-correction pipeline.
#
# Called by run_lassi_jobs.py via _launch_persistent_job():
#   bash polaris_persistent_worker.sh <ipc_dir>
#
# IPC protocol — all files in <ipc_dir> on Eagle:
#   job_started        written once at startup; content = hostname
#   trigger.json       Python writes to request work (JSON)
#   result_ready.json  worker writes when work is complete (JSON)
#   exit               Python writes to shut the worker down
#
# trigger.json schema:
#   {
#     "task":         "translate | sycl_energy | cuda_energy",
#     "src":          "/eagle/.../source.ext",
#     "outdir":       "/eagle/.../attempt_N/",
#     "run_args":     ["-m", "event"],
#     "compiler_env": {"SYCL_FLAGS": "...", "CUDA_FLAGS": "..."}
#   }
#
# result_ready.json schema:
#   {"build_rc": 0, "run_rc": 0, "outdir": "/eagle/.../attempt_N/"}
#
# Per-attempt outputs written to outdir (same layout as existing *_worker.sh):
#   build_rc.txt, build_stdout.log, build_stderr.log
#   run_rc.txt, run_stdout.log, run_stderr.log
#   power_samples.log, energy_results.txt  (energy tasks only)
#
# PBS stdout/stderr:
#   <ipc_dir>/pbs_stdout.log, <ipc_dir>/pbs_stderr.log
#
# Exit codes:
#   0  received exit signal (normal shutdown)
#   1  fatal startup error

# NOTE: intentionally no 'set -u'; Lmod init scripts access unset variables.
set -o pipefail

# shellcheck source=_common.sh
source "$(dirname "$0")/_common.sh"

IPC_DIR="$1"
if [ -z "$IPC_DIR" ]; then
    echo "ERROR: IPC_DIR argument missing" >&2
    exit 1
fi

mkdir -p "$IPC_DIR"

WORKER_LOG="${IPC_DIR}/worker.log"
: > "$WORKER_LOG"

_log() { echo "[$(date -u '+%H:%M:%S')] $*" >> "$WORKER_LOG"; }

_log "=== polaris_persistent_worker.sh ==="
_log "Date     : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
_log "Hostname : $(hostname)"
_log "IPC dir  : $IPC_DIR"
_log "Initial PATH: $PATH"

# ── oneAPI / icpx setup ────────────────────────────────────────────────────────
_log "=== oneAPI environment setup ==="

_LMOD_LOADED=false
for _lmod_init in \
    /etc/profile.d/z00_lmod.sh \
    /etc/profile.d/lmod.sh \
    /usr/share/lmod/lmod/init/bash \
    /usr/local/lmod/lmod/init/bash \
    /opt/cray/pe/lmod/lmod/init/bash
do
    [ -f "$_lmod_init" ] || continue
    _log "  [L1] sourcing Lmod init: $_lmod_init"
    # shellcheck source=/dev/null
    source "$_lmod_init" >> "$WORKER_LOG" 2>&1 || true
    module load oneapi/release >> "$WORKER_LOG" 2>&1 || true
    _LMOD_LOADED=true
    break
done
if [ "$_LMOD_LOADED" = false ]; then
    _log "  [L1] no Lmod init found — skipping"
fi

if ! command -v icpx &>/dev/null; then
    for _sv in /soft/compilers/oneapi/release/*/setvars.sh; do
        [ -f "$_sv" ] || continue
        # shellcheck source=/dev/null
        source "$_sv" --force >> "$WORKER_LOG" 2>&1 || true
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
_log "=== CUDA environment setup ==="

if command -v module &>/dev/null; then
    module load cudatoolkit-standalone >> "$WORKER_LOG" 2>&1 \
        || module load cudatoolkit >> "$WORKER_LOG" 2>&1 \
        || module load cuda >> "$WORKER_LOG" 2>&1 \
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
_log "  CUDA_PATH: ${CUDA_PATH:-NOT FOUND}"

# ── nvidia-smi discovery ──────────────────────────────────────────────────────
NVIDIA_SMI="$(command -v nvidia-smi 2>/dev/null)"
if [ -z "$NVIDIA_SMI" ]; then
    for _smi in /usr/bin/nvidia-smi /usr/local/cuda/bin/nvidia-smi /usr/local/bin/nvidia-smi; do
        [ -x "$_smi" ] && { NVIDIA_SMI="$_smi"; break; }
    done
fi

_log "  icpx     : $(command -v icpx 2>/dev/null || echo NOT FOUND)"
_log "  nvcc     : $(command -v nvcc 2>/dev/null || echo NOT FOUND)"
_log "  nvidia-smi: ${NVIDIA_SMI:-NOT FOUND}"
_log "  Final PATH: $PATH"

# ── Signal readiness ───────────────────────────────────────────────────────────
hostname > "${IPC_DIR}/job_started"
_log "=== Worker ready — entering event loop ==="

# ── Event loop ─────────────────────────────────────────────────────────────────
while true; do

    # ── Exit signal ───────────────────────────────────────────────────────────
    if [ -f "${IPC_DIR}/exit" ]; then
        _log "Received exit signal — shutting down."
        exit 0
    fi

    # ── Trigger check ─────────────────────────────────────────────────────────
    if [ ! -f "${IPC_DIR}/trigger.json" ]; then
        sleep 2
        continue
    fi

    _log "--- New trigger received ---"

    # ── Parse trigger.json ────────────────────────────────────────────────────
    # Extract variables into the shell environment via eval.
    if ! eval "$(python3 - "${IPC_DIR}/trigger.json" << 'PARSE_PYEOF'
import json, sys, shlex
with open(sys.argv[1]) as f:
    d = json.load(f)
print(f"TASK={shlex.quote(d.get('task', 'translate'))}")
print(f"SRC={shlex.quote(d.get('src', ''))}")
print(f"OUTDIR={shlex.quote(d.get('outdir', ''))}")
print(f"RUN_ARGS_JSON={shlex.quote(json.dumps(d.get('run_args', [])))}")
cenv = d.get("compiler_env", {})
for k in ("SYCL_COMPILER", "SYCL_FLAGS", "SYCL_LDFLAGS",
          "CUDA_COMPILER", "CUDA_FLAGS", "CUDA_LDFLAGS"):
    if k in cenv:
        print(f"{k}={shlex.quote(cenv[k])}")
PARSE_PYEOF
    )"; then
        _log "ERROR: Failed to parse trigger.json — skipping."
        rm -f "${IPC_DIR}/trigger.json"
        sleep 2
        continue
    fi

    # Reconstruct RUN_ARGS as a bash array from its JSON form.
    readarray -t RUN_ARGS < <(python3 -c "
import json, sys
for a in json.loads(sys.argv[1]):
    print(a)
" "$RUN_ARGS_JSON")

    _log "  TASK     : $TASK"
    _log "  SRC      : $SRC"
    _log "  OUTDIR   : $OUTDIR"
    _log "  RUN_ARGS : ${RUN_ARGS[*]:-<none>}"

    # ── Per-attempt initialisation ────────────────────────────────────────────
    BUILD_RC=-1
    RUN_RC=-1

    mkdir -p "$OUTDIR"
    : > "${OUTDIR}/build_stdout.log"
    : > "${OUTDIR}/build_stderr.log"

    {
        echo "=== polaris_persistent_worker.sh ==="
        echo "Date     : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "Hostname : $(hostname)"
        echo "Task     : $TASK"
        echo "Source   : $SRC"
        echo "Outdir   : $OUTDIR"
        echo "Run args : ${RUN_ARGS[*]:-<none>}"
        echo ""
    } >> "${OUTDIR}/build_stdout.log"

    # ── Task dispatch ──────────────────────────────────────────────────────────
    case "$TASK" in

        # ── cuda_energy: nvcc compile + nvidia-smi power monitoring ──────────
        cuda_energy)
            COMPILER="${CUDA_COMPILER:-nvcc}"
            _CUDA_DEFAULT="-O3 -arch=sm_80 -std=c++17"
            FLAGS="${CUDA_FLAGS:-$_CUDA_DEFAULT}"
            LDFLAGS="${CUDA_LDFLAGS:-}"
            OUTEXE="${OUTDIR}/Main"

            {
                echo "=== CUDA environment ==="
                echo "  Compiler: $COMPILER  ($(command -v "$COMPILER" 2>/dev/null || echo NOT FOUND))"
                echo "  Flags   : $FLAGS"
                echo ""
            } >> "${OUTDIR}/build_stdout.log"

            if ! command -v "$COMPILER" &>/dev/null; then
                echo "ERROR: $COMPILER not found." \
                    | tee -a "${OUTDIR}/build_stdout.log" > "${OUTDIR}/build_stderr.log"
                BUILD_RC=1
            else
                "$COMPILER" --version >> "${OUTDIR}/build_stdout.log" 2>&1 || true
                echo "=== Compiling ===" >> "${OUTDIR}/build_stdout.log"
                read -ra FLAGS_ARR   <<< "$FLAGS"
                read -ra LDFLAGS_ARR <<< "$LDFLAGS"
                "$COMPILER" "${FLAGS_ARR[@]}" "$SRC" -o "$OUTEXE" "${LDFLAGS_ARR[@]}" \
                    >> "${OUTDIR}/build_stdout.log" \
                    2> "${OUTDIR}/build_stderr.log"
                BUILD_RC=$?
            fi

            echo -n "$BUILD_RC" > "${OUTDIR}/build_rc.txt"

            if [ "$BUILD_RC" -eq 0 ]; then
                echo "=== Compilation OK ===" >> "${OUTDIR}/build_stdout.log"
                : > "${OUTDIR}/power_samples.log"
                chmod +x "$OUTEXE"

                MONITOR_PID=""
                if [ -n "$NVIDIA_SMI" ]; then
                    "$NVIDIA_SMI" dmon -s p -d 1 -i 0 \
                        >> "${OUTDIR}/power_samples.log" 2>&1 &
                    MONITOR_PID=$!
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
                _compute_energy "$START_TS_MS" "$END_TS_MS" \
                    "${OUTDIR}/power_samples.log" "${OUTDIR}/energy_results.txt" || true
                {
                    echo ""
                    echo "=== Energy results ==="
                    cat "${OUTDIR}/energy_results.txt" 2>/dev/null \
                        || echo "  (energy_results.txt not written)"
                } >> "${OUTDIR}/run_stdout.log"
                if [ "$RUN_RC" -ne 0 ]; then
                    echo "=== Execution FAILED (exit $RUN_RC) ===" \
                        >> "${OUTDIR}/run_stdout.log"
                else
                    echo "=== Execution OK ===" >> "${OUTDIR}/run_stdout.log"
                fi
            else
                echo "=== Compilation FAILED (exit $BUILD_RC) ===" \
                    >> "${OUTDIR}/build_stdout.log"
            fi
            ;;

        # ── sycl_energy: icpx compile + nvidia-smi power monitoring ──────────
        sycl_energy)
            COMPILER="${SYCL_COMPILER:-icpx}"
            _CUDA_FLAG=""
            [ -n "$CUDA_PATH" ] && _CUDA_FLAG="--cuda-path=${CUDA_PATH}"
            _SYCL_DEFAULT="-O3 -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
-Xsycl-target-backend --cuda-gpu-arch=sm_80${_CUDA_FLAG:+ $_CUDA_FLAG}"
            FLAGS="${SYCL_FLAGS:-$_SYCL_DEFAULT}"
            LDFLAGS="${SYCL_LDFLAGS:-}"
            OUTEXE="${OUTDIR}/Main"

            {
                echo "=== SYCL/oneAPI environment ==="
                echo "  Compiler: $COMPILER  ($(command -v "$COMPILER" 2>/dev/null || echo NOT FOUND))"
                echo "  Flags   : $FLAGS"
                echo ""
            } >> "${OUTDIR}/build_stdout.log"

            if ! command -v "$COMPILER" &>/dev/null; then
                echo "ERROR: $COMPILER not found." \
                    | tee -a "${OUTDIR}/build_stdout.log" > "${OUTDIR}/build_stderr.log"
                BUILD_RC=1
            else
                "$COMPILER" --version >> "${OUTDIR}/build_stdout.log" 2>&1 || true
                echo "=== Compiling ===" >> "${OUTDIR}/build_stdout.log"
                read -ra FLAGS_ARR   <<< "$FLAGS"
                read -ra LDFLAGS_ARR <<< "$LDFLAGS"
                "$COMPILER" "${FLAGS_ARR[@]}" "$SRC" -o "$OUTEXE" "${LDFLAGS_ARR[@]}" \
                    >> "${OUTDIR}/build_stdout.log" \
                    2> "${OUTDIR}/build_stderr.log"
                BUILD_RC=$?
            fi

            echo -n "$BUILD_RC" > "${OUTDIR}/build_rc.txt"

            if [ "$BUILD_RC" -eq 0 ]; then
                echo "=== Compilation OK ===" >> "${OUTDIR}/build_stdout.log"
                : > "${OUTDIR}/power_samples.log"
                chmod +x "$OUTEXE"
                export ONEAPI_DEVICE_SELECTOR=cuda:gpu

                MONITOR_PID=""
                if [ -n "$NVIDIA_SMI" ]; then
                    "$NVIDIA_SMI" dmon -s p -d 1 -i 0 \
                        >> "${OUTDIR}/power_samples.log" 2>&1 &
                    MONITOR_PID=$!
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
                _compute_energy "$START_TS_MS" "$END_TS_MS" \
                    "${OUTDIR}/power_samples.log" "${OUTDIR}/energy_results.txt" || true
                {
                    echo ""
                    echo "=== Energy results ==="
                    cat "${OUTDIR}/energy_results.txt" 2>/dev/null \
                        || echo "  (energy_results.txt not written)"
                } >> "${OUTDIR}/run_stdout.log"
                if [ "$RUN_RC" -ne 0 ]; then
                    echo "=== Execution FAILED (exit $RUN_RC) ===" \
                        >> "${OUTDIR}/run_stdout.log"
                else
                    echo "=== Execution OK ===" >> "${OUTDIR}/run_stdout.log"
                fi
            else
                echo "=== Compilation FAILED (exit $BUILD_RC) ===" \
                    >> "${OUTDIR}/build_stdout.log"
            fi
            ;;

        # ── translate: icpx compile + run (no energy monitoring) ─────────────
        translate)
            COMPILER="${SYCL_COMPILER:-icpx}"
            _CUDA_FLAG=""
            [ -n "$CUDA_PATH" ] && _CUDA_FLAG="--cuda-path=${CUDA_PATH}"
            _SYCL_DEFAULT="-O3 -std=c++17 -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
-Xsycl-target-backend --cuda-gpu-arch=sm_80${_CUDA_FLAG:+ $_CUDA_FLAG}"
            FLAGS="${SYCL_FLAGS:-$_SYCL_DEFAULT}"
            LDFLAGS="${SYCL_LDFLAGS:-}"
            OUTEXE="${OUTDIR}/Main"

            {
                echo "=== SYCL/oneAPI environment ==="
                echo "  Compiler: $COMPILER  ($(command -v "$COMPILER" 2>/dev/null || echo NOT FOUND))"
                echo "  Flags   : $FLAGS"
                echo ""
            } >> "${OUTDIR}/build_stdout.log"

            if ! command -v "$COMPILER" &>/dev/null; then
                echo "ERROR: $COMPILER not found." \
                    | tee -a "${OUTDIR}/build_stdout.log" > "${OUTDIR}/build_stderr.log"
                BUILD_RC=1
            else
                "$COMPILER" --version >> "${OUTDIR}/build_stdout.log" 2>&1 || true
                echo "=== Compiling ===" >> "${OUTDIR}/build_stdout.log"
                read -ra FLAGS_ARR   <<< "$FLAGS"
                read -ra LDFLAGS_ARR <<< "$LDFLAGS"
                "$COMPILER" "${FLAGS_ARR[@]}" "$SRC" -o "$OUTEXE" "${LDFLAGS_ARR[@]}" \
                    >> "${OUTDIR}/build_stdout.log" \
                    2> "${OUTDIR}/build_stderr.log"
                BUILD_RC=$?
            fi

            echo -n "$BUILD_RC" > "${OUTDIR}/build_rc.txt"

            if [ "$BUILD_RC" -eq 0 ]; then
                echo "=== Compilation OK ===" >> "${OUTDIR}/build_stdout.log"
                chmod +x "$OUTEXE"
                export ONEAPI_DEVICE_SELECTOR=cuda:gpu
                "$OUTEXE" "${RUN_ARGS[@]}" \
                    > "${OUTDIR}/run_stdout.log" \
                    2> "${OUTDIR}/run_stderr.log"
                RUN_RC=$?
                echo -n "$RUN_RC" > "${OUTDIR}/run_rc.txt"
                if [ "$RUN_RC" -ne 0 ]; then
                    echo "" >> "${OUTDIR}/run_stdout.log"
                    echo "=== Execution FAILED (exit $RUN_RC) ===" \
                        >> "${OUTDIR}/run_stdout.log"
                else
                    echo "" >> "${OUTDIR}/run_stdout.log"
                    echo "=== Execution OK ===" >> "${OUTDIR}/run_stdout.log"
                fi
            else
                echo "=== Compilation FAILED (exit $BUILD_RC) ===" \
                    >> "${OUTDIR}/build_stdout.log"
            fi
            ;;

        # ── Unknown task ──────────────────────────────────────────────────────
        *)
            _log "ERROR: Unknown task '$TASK'"
            BUILD_RC=1
            echo "ERROR: Unknown task '$TASK'" > "${OUTDIR}/build_stderr.log"
            echo -n "$BUILD_RC" > "${OUTDIR}/build_rc.txt"
            ;;
    esac

    _log "  BUILD_RC=$BUILD_RC  RUN_RC=$RUN_RC"

    # ── Write result_ready.json atomically ────────────────────────────────────
    _write_result_ready "$BUILD_RC" "$RUN_RC" "$OUTDIR" "$IPC_DIR"
    _log "  result_ready.json written."

    # ── Remove trigger to signal ready for next work ──────────────────────────
    rm -f "${IPC_DIR}/trigger.json"

    sleep 2
done
