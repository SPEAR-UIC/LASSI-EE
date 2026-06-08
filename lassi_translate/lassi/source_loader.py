"""Find + load CUDA application source.  source_apps.json maps app → folder +
run args + verification; *.cuh/*.h headers are prepended so the LLM sees the full program."""
import json
from pathlib import Path

from lassi.hpc_session import HPCSession


def find_source_dir(script_dir, app_folder, category=""):
    """Find source dir for `app_folder` under source_codes/[category/]."""
    src = script_dir / "source_codes"
    for suffix in ([f"{category}/{app_folder}", app_folder] if category else [app_folder]):
        if (c := src / suffix).is_dir(): return c
    raise FileNotFoundError(f"Source dir not found: {app_folder!r} under {src}")


def find_apps_json(script_dir):
    if (p := script_dir / "source_apps.json").exists(): return p
    raise FileNotFoundError(f"source_apps.json not found in {script_dir}")


def _prepend_headers(code, src_dir, patterns, exclude=None):
    chunks = []
    for pat in patterns:
        for hdr in sorted(src_dir.glob(pat)):
            if hdr == exclude: continue
            chunks.append(f"// ===== HEADER: {hdr.name} =====\n"
                          f"{hdr.read_text(encoding='utf-8')}\n"
                          f"// ===== END HEADER: {hdr.name} =====\n")
    return ("\n".join(chunks) + "\n" + code) if chunks else code


def load_cuda_source(script_dir, app_category, app_name):
    """Return (combined_code, run_args, main_cu_path, verification_strategy)."""
    catalog = json.loads(find_apps_json(script_dir).read_text())
    cfg = catalog.get(app_category, {}).get(app_name)
    if cfg is None:
        raise KeyError(f"App {app_category}/{app_name} not found")
    src_dir = find_source_dir(script_dir, cfg["app_folder"], category=app_category)
    main = src_dir / "Main.cu"
    if not main.exists():
        cu = sorted(src_dir.glob("*.cu"))
        if not cu: raise FileNotFoundError(f"No .cu files in {src_dir}")
        main = cu[0]
    code = _prepend_headers(main.read_text(encoding="utf-8"), src_dir, ("*.cuh", "*.h"))
    return code, cfg.get("app_params", []), main, cfg.get("verification", "none")


def sync_file_to_remote(session: HPCSession, local_file, remote_dir, exp, filename):
    """rsync one file to the canonical experiment directory on the remote FS."""
    return session.rsync(str(local_file),
                         f"{remote_dir}/generated_codes/experiment/{exp}/{filename}")
