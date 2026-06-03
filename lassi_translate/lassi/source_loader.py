"""Source-code discovery and upload helpers."""
from __future__ import annotations

import json
from pathlib import Path

from lassi.hpc_session import HPCSession


def find_source_dir(script_dir: Path, app_folder: str, category: str = "") -> Path:
    """Locate an application source directory inside <script_dir>/source_codes/.

    Supports both flat (source_codes/XSBench/) and nested
    (source_codes/Miniapps/XSBench/) layouts.
    """
    source_codes = script_dir / "source_codes"
    suffixes = [f"{category}/{app_folder}", app_folder] if category else [app_folder]
    for suffix in suffixes:
        candidate = source_codes / suffix
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Source directory not found for app_folder={app_folder!r} "
        f"(category={category!r}).\n"
        f"Expected inside: {source_codes}\n"
        f"Copy or symlink the source_codes/ directory into {script_dir}."
    )


def find_apps_json(script_dir: Path) -> Path:
    candidate = script_dir / "source_apps.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"source_apps.json not found in {script_dir}.\n"
        f"Copy the file into {script_dir}."
    )


def _prepend_headers(code: str, src_dir: Path, patterns: tuple[str, ...],
                     exclude: Path | None = None) -> str:
    """Prepend matching header files to `code` with delimiting comments."""
    chunks: list[str] = []
    for pattern in patterns:
        for hdr in sorted(src_dir.glob(pattern)):
            if exclude is not None and hdr == exclude:
                continue
            chunks.append(
                f"// ===== HEADER: {hdr.name} =====\n"
                + hdr.read_text(encoding="utf-8")
                + f"\n// ===== END HEADER: {hdr.name} =====\n"
            )
    return ("\n".join(chunks) + "\n" + code) if chunks else code


def load_cuda_source(
    script_dir: Path, app_category: str, app_name: str,
) -> tuple[str, list[str], Path, str]:
    """Load CUDA source code and parameters for the given app.

    Returns (combined_source_code, run_args, main_cu_path, verification).
    Headers (*.cuh, *.h) are prepended so the full context is visible to the LLM.
    """
    apps_json = find_apps_json(script_dir)
    catalog   = json.loads(apps_json.read_text())

    cat_block = catalog.get(app_category)
    if cat_block is None:
        raise KeyError(f"App category {app_category!r} not found in source_apps.json")
    app_cfg = cat_block.get(app_name)
    if app_cfg is None:
        raise KeyError(f"App {app_name!r} not found in category {app_category!r}")

    app_folder   = app_cfg["app_folder"]
    app_params   = app_cfg.get("app_params", [])
    verification = app_cfg.get("verification", "none")
    src_dir      = find_source_dir(script_dir, app_folder, category=app_category)

    main_cu = src_dir / "Main.cu"
    if not main_cu.exists():
        cu_files = sorted(src_dir.glob("*.cu"))
        if not cu_files:
            raise FileNotFoundError(f"No .cu files found in {src_dir}")
        main_cu = cu_files[0]

    code = _prepend_headers(
        main_cu.read_text(encoding="utf-8"),
        src_dir, ("*.cuh", "*.h"),
    )
    return code, app_params, main_cu, verification


def load_sycl_source(sycl_path: Path) -> str:
    """Load SYCL source plus sibling headers from an explicit file path."""
    if not sycl_path.exists():
        raise FileNotFoundError(f"SYCL source file not found: {sycl_path}")
    return _prepend_headers(
        sycl_path.read_text(encoding="utf-8"),
        sycl_path.parent, ("*.hpp", "*.h"),
        exclude=sycl_path,
    )


def sync_file_to_remote(
    session: HPCSession,
    local_file: Path,
    remote_dir: str,
    exp_name: str,
    filename: str,
) -> bool:
    """rsync one generated source file to its experiment directory on the remote FS."""
    remote = f"{remote_dir}/generated_codes/experiment/{exp_name}/{filename}"
    return session.rsync(str(local_file), remote)
