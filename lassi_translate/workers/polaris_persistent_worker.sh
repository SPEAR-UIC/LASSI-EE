#!/bin/bash
# polaris_persistent_worker.sh — Long-running PBS daemon on Polaris (NVIDIA A100).
#
# Usage: bash polaris_persistent_worker.sh <ipc_dir>
#
# IPC protocol (all files in <ipc_dir> on Eagle):
#   job_started        written once at startup; content = hostname
#   trigger.json       Python writes to request work
#   result_ready.json  worker writes when work is complete
#   exit               Python writes to shut the worker down
#
# trigger.json schema:
#   {"task": "translate|cuda_energy|sycl_energy",
#    "src": "...", "outdir": "...",
#    "run_args": [...], "compiler_env": {...}}

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
_log "Hostname: $(hostname)"
_log "IPC dir : $IPC_DIR"

# One-time environment setup (oneapi/icpx + cuda/nvcc).
_setup_lmod_oneapi "$WORKER_LOG"
_resolve_icpx       "$WORKER_LOG"
_setup_cuda         "$WORKER_LOG"
_resolve_cuda_path  "$WORKER_LOG"
_resolve_nvidia_smi

_log "  icpx      : $(command -v icpx 2>/dev/null || echo NOT FOUND)"
_log "  nvcc      : $(command -v nvcc 2>/dev/null || echo NOT FOUND)"
_log "  nvidia-smi: ${NVIDIA_SMI:-NOT FOUND}"

# Signal readiness — Python polls for this file.
hostname > "${IPC_DIR}/job_started"
_log "=== Worker ready — entering event loop ==="

while true; do
    if [ -f "${IPC_DIR}/exit" ]; then
        _log "Received exit signal — shutting down."
        exit 0
    fi
    if [ ! -f "${IPC_DIR}/trigger.json" ]; then
        sleep 2
        continue
    fi

    _log "--- New trigger ---"

    # Clear stale compiler-env vars so a trigger that omits one truly defaults.
    unset SYCL_COMPILER SYCL_FLAGS SYCL_LDFLAGS \
          CUDA_COMPILER CUDA_FLAGS CUDA_LDFLAGS

    # Parse trigger.json: emit shell assignments via shlex.quote.
    if ! eval "$(python3 - "${IPC_DIR}/trigger.json" << 'PARSE_PYEOF'
import json, sys, shlex
with open(sys.argv[1]) as f:
    d = json.load(f)
print(f"TASK={shlex.quote(d.get('task', 'translate'))}")
print(f"SRC={shlex.quote(d.get('src', ''))}")
print(f"OUTDIR={shlex.quote(d.get('outdir', ''))}")
print(f"RUN_ARGS_JSON={shlex.quote(json.dumps(d.get('run_args', [])))}")
for k, v in d.get("compiler_env", {}).items():
    if k in ("SYCL_COMPILER","SYCL_FLAGS","SYCL_LDFLAGS",
             "CUDA_COMPILER","CUDA_FLAGS","CUDA_LDFLAGS"):
        print(f"{k}={shlex.quote(v)}")
PARSE_PYEOF
    )"; then
        _log "ERROR: failed to parse trigger.json — skipping"
        rm -f "${IPC_DIR}/trigger.json"
        sleep 2
        continue
    fi

    readarray -t RUN_ARGS < <(python3 -c '
import json, sys
for a in json.loads(sys.argv[1]):
    print(a)
' "$RUN_ARGS_JSON")

    _log "  TASK=$TASK  SRC=$SRC  OUTDIR=$OUTDIR  RUN_ARGS=${RUN_ARGS[*]:-<none>}"

    mkdir -p "$OUTDIR"
    : > "${OUTDIR}/build_stdout.log"
    : > "${OUTDIR}/build_stderr.log"
    {
        echo "=== polaris_persistent_worker.sh ==="
        echo "Hostname: $(hostname)"
        echo "Task    : $TASK"
        echo "Source  : $SRC"
        echo "Outdir  : $OUTDIR"
        echo "Run args: ${RUN_ARGS[*]:-<none>}"
        echo ""
    } >> "${OUTDIR}/build_stdout.log"

    # ── Per-task settings ─────────────────────────────────────────────────
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
            _log "ERROR: unknown task '$TASK'"
            echo "ERROR: unknown task '$TASK'" > "${OUTDIR}/build_stderr.log"
            echo -n "1" > "${OUTDIR}/build_rc.txt"
            _write_result_ready "1" "-1" "$OUTDIR" "$IPC_DIR"
            rm -f "${IPC_DIR}/trigger.json"
            sleep 2
            continue
            ;;
    esac

    _compile_and_run "$COMPILER" "$FLAGS" "$LDFLAGS" "$SRC" "$OUTDIR" \
                     "$MONITOR" "$DEVICE_SELECTOR" 0 \
                     "${RUN_ARGS[@]}"

    BUILD_RC=$(cat "${OUTDIR}/build_rc.txt" 2>/dev/null || echo -1)
    RUN_RC=$(cat   "${OUTDIR}/run_rc.txt"   2>/dev/null || echo -1)
    _log "  BUILD_RC=$BUILD_RC  RUN_RC=$RUN_RC"

    _write_result_ready "$BUILD_RC" "$RUN_RC" "$OUTDIR" "$IPC_DIR"
    _log "  result_ready.json written"

    rm -f "${IPC_DIR}/trigger.json"
    sleep 2
done
