#!/usr/bin/env python3
"""lassi_translate preflight check across all modes.  Prints PASS/FAIL/SKIP
per check with a per-mode readiness verdict.  Tiers (fastest → slowest):
    0  environment + deps      (instant, offline)
    1  static project checks   (instant, offline)
    2  amsc connect + fs ops   (seconds, needs Globus auth)
    3  SSH + rsync round-trip  (seconds, needs Polaris ssh key/pwd)
    4  ALCF Inference auth     (seconds, needs Globus inference auth)
    5  compute.submit PBS job  (minutes, opt-in via --full)
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── ANSI colors ──────────────────────────────────────────────────────────────
G  = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[1m"; D = "\033[2m"; E = "\033[0m"
PASS = f"{G}PASS{E}"; FAIL = f"{R}FAIL{E}"; SKIP = f"{Y}SKIP{E}"

ROOT = Path(__file__).resolve().parent
_results: list[tuple[str, str, str, str]] = []   # (tier, name, status, message)
_per_mode_blockers: dict[str, list[str]] = {"remote": [], "on-resource": [], "submit": []}


class SkipTest(Exception):
    """Signal SKIP rather than FAIL from inside a check."""


def _log(tier, name, status, msg, blocks=None):
    _results.append((tier, name, status, msg))
    print(f"  [{tier}] {name}{' ' * max(0, 50 - len(name))} {status}  {D}{msg}{E}")
    if status == FAIL and blocks:
        for m in blocks: _per_mode_blockers[m].append(name)


def _header(label):
    print(f"\n{B}── {label} ──────────────────────────────────────────────────{E}")


def _try(tier, name, fn, blocks=None):
    """Run fn(); log PASS/SKIP/FAIL.  fn may return a short PASS message."""
    try: _log(tier, name, PASS, str(fn() or "ok")[:80])
    except SkipTest as exc: _log(tier, name, SKIP, str(exc))
    except Exception as exc: _log(tier, name, FAIL, f"{type(exc).__name__}: {exc}"[:200], blocks)


# ── Tier 0: environment ──────────────────────────────────────────────────────

def tier0_environment():
    _header("Tier 0 — Environment")

    def py_version():
        v = sys.version_info
        if v < (3, 11):
            raise RuntimeError(f"Python {v.major}.{v.minor} found; need >=3.11")
        return f"Python {v.major}.{v.minor}.{v.micro}"
    _try("env", "Python version", py_version, blocks=["remote", "on-resource", "submit"])

    def venv():
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("not running inside a venv (sys.prefix == sys.base_prefix)")
        return f"venv at {sys.prefix}"
    _try("env", "venv active", venv, blocks=["remote", "submit"])

    def cwd():
        if Path.cwd() != ROOT:
            raise RuntimeError(f"CWD {Path.cwd()} != ROOT {ROOT}; run from project root")
        return f"in {ROOT}"
    _try("env", "Working directory", cwd)

    for pkg in ("amsc_client", "globus_sdk"):
        def chk(p=pkg):
            mod = importlib.import_module(p)
            return f"{p} {getattr(mod, '__version__', '?')}"
        _try("env", f"import {pkg}", chk, blocks=["remote", "submit"])


# ── Tier 1: static project checks ────────────────────────────────────────────

def tier1_static():
    _header("Tier 1 — Static project checks")

    # All lassi/* modules importable
    for m in ("lassi.alcf", "lassi.alcf_inference", "lassi.format_colors", "lassi.hpc_jobs",
              "lassi.hpc_session", "lassi.llm_client", "lassi.local_runner", "lassi.local_session",
              "lassi.pipeline", "lassi.prompt_builder", "lassi.prompt_dictionary",
              "lassi.resources", "lassi.results", "lassi.source_loader"):
        def chk(name=m):
            importlib.import_module(name); return ""
        _try("mod", m, chk, blocks=["remote", "on-resource", "submit"])

    # CLI parser builds
    def cli():
        import run_lassi
        p = run_lassi._build_parser({})
        h = p.format_help()
        assert "--task" in h and "--submit" in h and "--on-resource" in h
        return f"{len(p._actions)} CLI args"
    _try("cli", "run_lassi parser builds", cli, blocks=["remote", "on-resource", "submit"])

    # lassi_config.py loads with required keys
    def cfg():
        import run_lassi
        cfg = run_lassi._load_config(ROOT / "lassi_config.py")
        required = {"resource", "task", "alcf_username", "argo_model", "inference_model"}
        missing = required - set(cfg)
        if missing: raise RuntimeError(f"missing keys: {sorted(missing)}")
        return f"resource={cfg['resource']}, user={cfg['alcf_username']}, task={cfg['task']}"
    _try("cfg", "lassi_config.py", cfg, blocks=["remote", "on-resource", "submit"])

    # source_codes symlink resolves
    def src_symlink():
        s = ROOT / "source_codes"
        if not s.exists(): raise RuntimeError("source_codes/ missing (broken symlink?)")
        return f"resolves to {s.resolve()}"
    _try("src", "source_codes symlink", src_symlink,
         blocks=["remote", "on-resource", "submit"])

    # source_apps.json parseable + XSBench entry
    def apps():
        from lassi.source_loader import find_apps_json
        j = json.loads(find_apps_json(ROOT).read_text())
        if "Miniapps" not in j or "XSBench" not in j["Miniapps"]:
            raise RuntimeError("XSBench entry missing")
        return f"{sum(len(v) for v in j.values())} apps total"
    _try("src", "source_apps.json", apps, blocks=["remote", "on-resource", "submit"])

    # XSBench actually loads end-to-end (the real test the pipeline uses)
    def load_xsbench():
        from lassi.source_loader import load_cuda_source
        code, run_args, main, ver = load_cuda_source(ROOT, "Miniapps", "XSBench")
        if len(code) < 1000: raise RuntimeError(f"only {len(code)} chars loaded")
        return f"{len(code.splitlines())} lines, main={main.name}, verify={ver}"
    _try("src", "load_cuda_source(Miniapps/XSBench)", load_xsbench,
         blocks=["remote", "on-resource", "submit"])

    # All workers/*.sh pass bash -n
    for sh in sorted((ROOT / "workers").glob("*.sh")):
        def chk(p=sh):
            r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip()[:150])
            return f"{p.stat().st_size}B"
        _try("sh", f"workers/{sh.name} syntax", chk,
             blocks=["remote", "on-resource", "submit"])

    # prompt_dictionary loads with the 9 used keys
    def pd():
        from lassi.prompt_dictionary import PromptDictionary
        d = PromptDictionary()
        # system
        for k in ("CUDA_to_SYCL", "CUDA_ee"):
            assert d.get_system_prompt(k) != "Key not found", f"sys.{k}"
        # codetranslate
        assert d.get_codetranslate_prompt("CUDA_to_SYCL") != "Key not found"
        # energy refactor
        assert d.get_code_ee_refactoring_prompt("CUDA_baseline_nocontext") != "Key not found"
        # error correction
        for k in ("CUDA_error", "SYCL_error"):
            assert d.get_code_error_correction_prompt(k) != "Key not found", f"err.{k}"
        # comparison
        assert d.get_code_comparison("CUDA_cpp") != "Key not found"
        # codeknowledge (context docs)
        for k in ("cuda_optimization_summary", "sycl_programming"):
            txt = d.get_codeknowledge(k)
            assert txt and txt != "Key not found", f"ctx.{k}"
        return "all 9 required prompts + 2 context docs"
    _try("pd", "prompt_dictionary integrity", pd,
         blocks=["remote", "on-resource", "submit"])

    # prompt_builder TASK_KEYS resolve through PromptDictionary
    def pb():
        from lassi.prompt_builder import _TASK_KEYS, build_stage1_prompt
        from lassi.prompt_dictionary import PromptDictionary
        pd = PromptDictionary()
        for combo, keys in _TASK_KEYS.items():
            sp, up = build_stage1_prompt(prompts=pd, task=combo[0], resource=combo[1],
                                          input_code="// x", guide_summary="", source_description="")
            assert sp and up and "Key not found" not in sp + up, f"{combo}"
        return f"{len(_TASK_KEYS)} (task,resource) combos OK"
    _try("pb", "prompt_builder TASK_KEYS resolve", pb,
         blocks=["remote", "on-resource", "submit"])

    # context markdowns present
    def ctx():
        cdir = ROOT / "lassi" / "context"
        wanted = ["CUDA_C_Optimization_Guide_Summary.md", "SYCL_Programming_Guide.md"]
        missing = [w for w in wanted if not (cdir / w).exists()]
        if missing: raise RuntimeError(f"missing: {missing}")
        return f"{len(wanted)}/{len(wanted)} context files"
    _try("ctx", "lassi/context/ files", ctx,
         blocks=["remote", "on-resource", "submit"])


# ── Tier 2: amsc connect + fs round-trip ────────────────────────────────────

_amsc_state: dict = {}

def tier2_amsc(skip: bool):
    _header("Tier 2 — amsc connectivity + filesystem round-trip")
    if skip:
        _log("amsc", "all checks", SKIP, "--no-amsc set")
        return
    try:
        from lassi.alcf        import amsc_connect
        from lassi.hpc_session import HPCSession
        from lassi.resources   import RESOURCES
    except Exception as exc:
        _log("amsc", "import lassi.alcf", FAIL, str(exc), ["remote", "submit"])
        return

    cfg = RESOURCES["polaris"]
    try:
        compute, fs = amsc_connect(cfg.amsc_resource, cfg.amsc_fs, ROOT, quiet=True)
        _amsc_state["compute"] = compute
        _amsc_state["session"] = HPCSession(cfg.host, os.environ.get("USER", "vchirio"), fs)
        _log("amsc", "amsc_connect (Polaris+Eagle)", PASS, "Globus auth OK")
    except Exception as exc:
        _log("amsc", "amsc_connect (Polaris+Eagle)", FAIL,
             f"{type(exc).__name__}: {exc}", ["remote", "submit"])
        return

    s = _amsc_state["session"]
    base = f"{cfg.default_dir}/_debug_{int(time.time())}"
    payload = b"debug-payload\n"

    def mkdir():        s.mkdir(base); s.mkdir(base);            return "idempotent OK"
    def upload():
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(payload); tmp = f.name
        try:
            if not s.rsync(tmp, f"{base}/hello.txt"):
                raise RuntimeError("rsync upload returned False")
        finally: Path(tmp).unlink(missing_ok=True)
        return f"{len(payload)}B uploaded"
    def ls():
        names = [e.get("name") for e in s.ls(base)]
        if "hello.txt" not in names: raise RuntimeError(f"hello.txt missing; got {names}")
        return f"{len(names)} entries"
    def exists():
        assert s.file_exists(f"{base}/hello.txt"), "True expected"
        assert not s.file_exists(f"{base}/nope"), "False expected"
        return "True for existing, False for missing"
    def size():
        n = s.file_size(f"{base}/hello.txt")
        if n != len(payload): raise RuntimeError(f"size={n}, expected {len(payload)}")
        return f"{n}B"
    def read_file():
        if s.read_file(f"{base}/hello.txt").strip() != payload.decode().strip():
            raise RuntimeError("content mismatch")
        return f"{len(payload)}B read"
    def read_bytes():
        chunk = s.read_bytes(f"{base}/hello.txt", offset=0, size=5)
        if "debug" not in chunk: raise RuntimeError(f"got {chunk!r}")
        return f"got {chunk!r}"
    def rm():
        s.rm(f"{base}/hello.txt"); s.rm(f"{base}/hello.txt")   # idempotent
        if s.file_exists(f"{base}/hello.txt"): raise RuntimeError("file still there")
        return "deleted + idempotent re-rm"
    def cleanup():     s.rm(base); return "removed"

    for name, fn in [("mkdir + idempotent",     mkdir),
                     ("rsync upload",            upload),
                     ("ls",                       ls),
                     ("file_exists",              exists),
                     ("file_size (via ls)",       size),
                     ("read_file (fs.head)",      read_file),
                     ("read_bytes (fs.view)",     read_bytes),
                     ("rm + idempotent",          rm),
                     ("cleanup",                  cleanup)]:
        _try("fs", name, fn, blocks=["remote", "submit"])


# ── Tier 3: SSH ControlMaster + rsync round-trip ────────────────────────────

def tier3_ssh(skip: bool):
    _header("Tier 3 — SSH transport (ControlMaster + rsync pull)")
    if skip:
        _log("ssh", "all checks", SKIP, "--no-amsc skips SSH too (needs session)")
        return
    s = _amsc_state.get("session")
    if s is None:
        _log("ssh", "all checks", SKIP, "amsc session not created (tier 2 failed)")
        return

    def ping():
        r = subprocess.run(["ssh", "-o", f"ControlPath={s._ssh_ctl}",
                            "-o", "ControlMaster=auto", "-o", "ControlPersist=300",
                            "-o", "StrictHostKeyChecking=accept-new",
                            f"{s.user}@{s.host}", "hostname"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0: raise RuntimeError(r.stderr.strip()[:150])
        return f"hostname → {r.stdout.strip()}"

    def roundtrip():
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("ssh-roundtrip\n"); local_up = f.name
        remote = f"/home/{s.user}/_lassi_debug_ssh_{int(time.time())}.txt"
        try:
            if not s.rsync(local_up, remote): raise RuntimeError("upload failed")
            with tempfile.TemporaryDirectory() as d:
                if not s.rsync_pull(remote, Path(d) / "back.txt", is_dir=False):
                    raise RuntimeError("pull failed")
                got = (Path(d) / "back.txt").read_text()
                if "ssh-roundtrip" not in got: raise RuntimeError(f"content mismatch: {got!r}")
        finally:
            Path(local_up).unlink(missing_ok=True)
            try:
                subprocess.run(["ssh", "-o", f"ControlPath={s._ssh_ctl}",
                                f"{s.user}@{s.host}", "rm", "-f", remote],
                               capture_output=True, timeout=30)
            except Exception: pass
        return "upload → pull → content match"

    _try("ssh", "ssh hostname via ControlMaster", ping, blocks=["remote", "submit"])
    _try("ssh", "rsync upload + rsync_pull",      roundtrip, blocks=["remote", "submit"])


# ── Tier 4: ALCF Inference auth (compute-node-side scope) ────────────────────

def tier4_inference():
    _header("Tier 4 — ALCF Inference token (used by --on-resource / --submit)")
    helper = ROOT / "inference_auth_token.py"
    def auth_helper():
        if not helper.exists(): raise RuntimeError(f"{helper} missing")
        return f"{helper.stat().st_size}B"
    _try("inf", "inference_auth_token.py present", auth_helper,
         blocks=["on-resource", "submit"])

    def tokens_cached():
        from run_lassi import _local_tokens_dir
        td = _local_tokens_dir()
        tf = td / "tokens.json"
        if not tf.exists():
            raise SkipTest(f"no cached tokens at {tf} — first --submit run will trigger browser auth")
        return f"{tf.stat().st_size}B at {td}"
    _try("inf", "tokens.json cached locally", tokens_cached)

    def access_token():
        try:
            from lassi.alcf_inference import get_access_token
        except Exception as exc:
            raise RuntimeError(f"import alcf_inference: {exc}")
        try:
            tok = get_access_token(ROOT)
        except Exception as exc:
            raise RuntimeError(f"{type(exc).__name__}: {str(exc)[:120]}")
        if not tok or len(tok) < 20:
            raise RuntimeError(f"got {tok!r}")
        return f"length={len(tok)}"
    _try("inf", "alcf_inference.get_access_token()", access_token,
         blocks=["on-resource", "submit"])


# ── Tier 5: real PBS submit (opt-in via --full) ─────────────────────────────

def tier5_submit(skip: bool):
    _header("Tier 5 — compute.submit + Job.refresh + cancel (opt-in via --full)")
    if skip:
        _log("pbs", "all checks", SKIP, "default (use --full to run, queues a real job)")
        return
    compute = _amsc_state.get("compute")
    if compute is None:
        _log("pbs", "all checks", SKIP, "amsc compute not connected (tier 2 failed)")
        return

    from lassi.resources import RESOURCES
    cfg = RESOURCES["polaris"]
    job_state = {"obj": None}

    def submit_echo():
        base = f"{cfg.default_dir}/_debug_pbs_{int(time.time())}"
        s = _amsc_state["session"]; s.mkdir(base)
        log = f"{base}/echo.log"
        j = compute.submit(
            executable="/bin/echo", arguments=["preflight-ok"],
            directory=base, name=f"lassi-debug-{int(time.time())}",
            queue="debug", account="EE-ECP", duration=300, nodes=1,
            stdout_path=log, stderr_path=log + ".err",
            filesystems=cfg.filesystems)
        job_state["obj"] = j
        return f"job_id={j.id.split('.')[0]}"
    _try("pbs", "compute.submit (5-min /bin/echo)", submit_echo, blocks=["remote", "submit"])

    def refresh():
        j = job_state["obj"]
        if j is None: raise SkipTest("submit failed")
        for _ in range(3):
            state = j.refresh()
            time.sleep(1)
        return f"final state={state}"
    _try("pbs", "Job.refresh on held reference", refresh, blocks=["remote", "submit"])

    def jobs_lookup_is_501():
        from amsc_client.core.exceptions import ApiError
        j = job_state["obj"]
        if j is None: raise SkipTest("submit failed")
        try:
            compute.jobs([j.id])
            return "UNEXPECTEDLY worked — ALCF may have implemented job-by-id lookup; review hpc_jobs.py"
        except ApiError as exc:
            if "501" in str(exc) or "not implemented" in str(exc).lower():
                return "501 as expected"
            raise
    _try("pbs", "compute.jobs([id]) returns 501", jobs_lookup_is_501)

    j = job_state["obj"]
    if j is not None:
        try: j.cancel()
        except Exception as exc: print(f"  (could not cancel debug job {j.id}: {exc})")


# ── Summary ──────────────────────────────────────────────────────────────────

def _summary(full_run, no_amsc):
    _header("Summary")
    counts = {s: sum(1 for _, _, st, _ in _results if st == s) for s in (PASS, FAIL, SKIP)}
    total = sum(counts.values())
    print(f"  {counts[PASS]}/{total} passed, {counts[FAIL]} failed, {counts[SKIP]} skipped\n")

    print(f"  {B}Mode readiness:{E}")
    for mode in ("remote", "on-resource", "submit"):
        blockers = _per_mode_blockers[mode]
        if not blockers:
            print(f"    {G}✓ {mode:<12}{E}  ready")
        else:
            more = f" (+{len(blockers)-3} more)" if len(blockers) > 3 else ""
            print(f"    {R}✗ {mode:<12}{E}  blocked: {', '.join(blockers[:3])}{more}")

    if not full_run and not no_amsc:
        print(f"\n  {D}note: --full also queues a /bin/echo PBS job to verify compute.submit.{E}")
    if no_amsc:
        print(f"\n  {D}note: --no-amsc skipped all ALCF connectivity tests.{E}")
    return counts[FAIL]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full",    action="store_true", help="also run tier 5 (queues a real PBS job)")
    ap.add_argument("--no-amsc", action="store_true", help="skip all ALCF connectivity tests")
    args = ap.parse_args()

    print(f"{B}lassi_translate debug — {platform.node()} @ {time.strftime('%Y-%m-%d %H:%M:%S')}{E}")
    print(f"  Project root: {ROOT}")
    tier0_environment()
    tier1_static()
    tier2_amsc(args.no_amsc)
    tier3_ssh(args.no_amsc)
    tier4_inference()
    tier5_submit(skip=not args.full or args.no_amsc)
    return _summary(args.full, args.no_amsc)


if __name__ == "__main__":
    sys.exit(main())
