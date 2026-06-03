#!/bin/bash
# aurora_worker.sh — One-shot compile + run on an Aurora compute node
# (Intel Data Center GPU Max / Ponte Vecchio, icpx + Level Zero).
#
# Usage: bash aurora_worker.sh <task> <src> <outdir> [run_args...]
#   task : translate | sycl_energy
#
# Optional env overrides: SYCL_COMPILER, SYCL_FLAGS, SYCL_LDFLAGS
#
# NOTE: compilation uses eval to preserve quoting around -Xs "-device pvc".
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
    echo "=== aurora_worker.sh ==="
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
_resolve_xpu_smi    "$LOG"

{
    echo "=== Tools ==="
    echo "  icpx   : $(command -v icpx 2>/dev/null || echo NOT FOUND)"
    echo "  xpu-smi: ${XPU_SMI:-NOT FOUND}"
    [ -n "$XPU_SMI" ] && "$XPU_SMI" discovery 2>&1 | head -20 || true
    echo ""
} >> "$LOG"

case "$TASK" in
    translate|sycl_energy)
        COMPILER="${SYCL_COMPILER:-icpx}"
        _SYCL_DEFAULT='-O3 -std=c++17 -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc"'
        FLAGS="${SYCL_FLAGS:-$_SYCL_DEFAULT}"
        LDFLAGS="${SYCL_LDFLAGS:-}"
        MONITOR=$([ "$TASK" = sycl_energy ] && echo xpu_smi || echo none)
        ;;
    *)
        echo "ERROR: unknown task '$TASK' (expected: translate|sycl_energy)" \
            | tee -a "$LOG" > "${OUTDIR}/build_stderr.log"
        echo -n "1" > "${OUTDIR}/build_rc.txt"
        exit 1
        ;;
esac

# use_eval=1: preserve embedded quotes in -Xs "-device pvc"
_compile_and_run "$COMPILER" "$FLAGS" "$LDFLAGS" "$SRC" "$OUTDIR" \
                 "$MONITOR" "level_zero:gpu" 1 \
                 "${RUN_ARGS[@]}"

BUILD_RC=$(cat "${OUTDIR}/build_rc.txt")
RUN_RC=$(cat "${OUTDIR}/run_rc.txt" 2>/dev/null || echo -1)

[ "$BUILD_RC" -ne 0 ] && exit 1
[ "$RUN_RC"   -ne 0 ] && exit 2
exit 0
