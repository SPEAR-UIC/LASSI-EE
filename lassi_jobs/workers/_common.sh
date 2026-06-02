#!/bin/bash
# _common.sh — Shared helpers sourced by all LASSI-EE worker scripts.
# Do not execute directly.

# Usage: _compute_energy <start_ms> <end_ms> <power_samples.log> <energy_results.txt>
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

# power_samples.log format: "<gpu_idx> <power_W>" (nvidia-smi dmon -s p or xpu-smi poll)
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

result = {
    "energy_joules": round(energy_joules, 3),
    "avg_power_w":   round(avg_power_w, 3),
    "duration_s":    round(duration_s, 3),
    "num_samples":   len(samples),
}

with open(out_file, "w") as fh:
    json.dump(result, fh)
ENERGY_PYEOF
}

# Usage: _write_result_ready <build_rc> <run_rc> <outdir> <ipc_dir>
# Writes result_ready.json atomically (tmp + mv).
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
