#!/bin/bash
# aurora_persistent_worker.sh — Long-running PBS daemon on Aurora (Intel PVC).
#
# Usage: bash aurora_persistent_worker.sh <ipc_dir>
# Same IPC protocol as polaris_persistent_worker.sh; handles translate and
# sycl_energy task types.

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

_log "=== aurora_persistent_worker.sh ==="
_log "Hostname: $(hostname)"
_log "IPC dir : $IPC_DIR"

_setup_lmod_oneapi "$WORKER_LOG"
_resolve_icpx       "$WORKER_LOG"
_resolve_xpu_smi    "$WORKER_LOG"

_log "  icpx   : $(command -v icpx 2>/dev/null || echo NOT FOUND)"
_log "  xpu-smi: ${XPU_SMI:-NOT FOUND}"

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

    unset SYCL_COMPILER SYCL_FLAGS SYCL_LDFLAGS

    if ! eval "$(python3 - "${IPC_DIR}/trigger.json" << 'PARSE_PYEOF'
import json, sys, shlex
with open(sys.argv[1]) as f:
    d = json.load(f)
print(f"TASK={shlex.quote(d.get('task', 'translate'))}")
print(f"SRC={shlex.quote(d.get('src', ''))}")
print(f"OUTDIR={shlex.quote(d.get('outdir', ''))}")
print(f"RUN_ARGS_JSON={shlex.quote(json.dumps(d.get('run_args', [])))}")
for k, v in d.get("compiler_env", {}).items():
    if k in ("SYCL_COMPILER","SYCL_FLAGS","SYCL_LDFLAGS"):
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
        echo "=== aurora_persistent_worker.sh ==="
        echo "Hostname: $(hostname)"
        echo "Task    : $TASK"
        echo "Source  : $SRC"
        echo "Outdir  : $OUTDIR"
        echo "Run args: ${RUN_ARGS[*]:-<none>}"
        echo ""
    } >> "${OUTDIR}/build_stdout.log"

    case "$TASK" in
        translate|sycl_energy)
            COMPILER="${SYCL_COMPILER:-icpx}"
            _SYCL_DEFAULT='-O3 -std=c++17 -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc"'
            FLAGS="${SYCL_FLAGS:-$_SYCL_DEFAULT}"
            LDFLAGS="${SYCL_LDFLAGS:-}"
            MONITOR=$([ "$TASK" = sycl_energy ] && echo xpu_smi || echo none)
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

    # use_eval=1: preserve embedded quotes in -Xs "-device pvc"
    _compile_and_run "$COMPILER" "$FLAGS" "$LDFLAGS" "$SRC" "$OUTDIR" \
                     "$MONITOR" "level_zero:gpu" 1 \
                     "${RUN_ARGS[@]}"

    BUILD_RC=$(cat "${OUTDIR}/build_rc.txt" 2>/dev/null || echo -1)
    RUN_RC=$(cat   "${OUTDIR}/run_rc.txt"   2>/dev/null || echo -1)
    _log "  BUILD_RC=$BUILD_RC  RUN_RC=$RUN_RC"

    _write_result_ready "$BUILD_RC" "$RUN_RC" "$OUTDIR" "$IPC_DIR"
    _log "  result_ready.json written"

    rm -f "${IPC_DIR}/trigger.json"
    sleep 2
done
