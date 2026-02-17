#!/usr/bin/env python3
"""LASSI Translation Pipeline using ALCF Inference Endpoints.

This script runs the full LASSI code translation pipeline on Aurora compute nodes
using ALCF's inference endpoints, which are accessible from within the ALCF network.

Usage:
    # On Aurora compute node via PBS job
    python3 run_lassi_alcf.py

    # Configure via environment variables:
    export SOURCE_FILE=path/to/cuda_code.cu
    export OUTPUT_DIR=exp_name
    export MODEL=gpt52
    python3 run_lassi_alcf.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add lassi to path - use ALCF version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lassi'))


def run_translation(source_file, output_dir, model="gpt52", max_iterations=5):
    """Run CUDA to SYCL translation with iterative refinement.

    Args:
        source_file: Path to CUDA source code
        output_dir: Output directory for generated code
        model: Model to use (will be mapped to ALCF model)
        max_iterations: Maximum refinement iterations

    Returns:
        bool: True if successful
    """

    print("="*70)
    print("LASSI TRANSLATION PIPELINE - ALCF Inference Endpoints")
    print("="*70)
    print(f"Source file: {source_file}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model}")
    print(f"Max iterations: {max_iterations}")
    print(f"Timestamp: {datetime.now()}")
    print("="*70)

    # Import ALCF LLM backend
    from llm_config_alcf import argo_llm

    # Read source code
    if not os.path.exists(source_file):
        print(f"ERROR: Source file not found: {source_file}")
        return False

    with open(source_file, 'r') as f:
        source_code = f.read()

    print(f"\nSource code length: {len(source_code)} characters")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save source code
    (output_path / "original.cu").write_text(source_code)

    # Translation iterations
    error_feedback = None

    for iteration in range(max_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1} / {max_iterations}")
        print(f"{'='*70}")

        # Build prompt
        system_prompt = "You are an expert in CUDA to SYCL code translation."

        if error_feedback:
            content_prompt = f"""
Previous SYCL translation attempt had the following issues:

{error_feedback}

Please fix these issues and provide a corrected SYCL translation.

Original CUDA code:
```cuda
{source_code}
```

Provide ONLY the corrected SYCL code, no explanations.
"""
        else:
            content_prompt = f"""
Translate the following CUDA code to SYCL.

CUDA code:
```cuda
{source_code}
```

Provide ONLY the SYCL code, no explanations.
"""

        # Call LLM
        print(f"[{iteration+1}] Calling ALCF Inference API...")
        try:
            response = argo_llm(
                model=model,
                system_prompt=system_prompt,
                content_prompt=content_prompt,
                temp=0.2,
                verifyprompt=False
            )

            print(f"[{iteration+1}] ✓ Response received ({len(response)} characters)")

        except Exception as e:
            print(f"[{iteration+1}] ✗ LLM call failed: {e}")
            return False

        # Save generated code
        generated_file = output_path / f"generated_iter{iteration+1}.cpp"
        generated_file.write_text(response)
        print(f"[{iteration+1}] Saved to: {generated_file}")

        # Try to compile (optional, if compiler available)
        compile_success = try_compile(generated_file, output_path, iteration + 1)

        if compile_success:
            print(f"\n{'='*70}")
            print(f"SUCCESS after {iteration + 1} iteration(s)!")
            print(f"Final code: {generated_file}")
            print(f"{'='*70}")

            # Save success metadata
            metadata = {
                "status": "success",
                "iterations": iteration + 1,
                "final_file": str(generated_file),
                "timestamp": datetime.now().isoformat()
            }
            (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

            return True

        # Get compilation errors for next iteration
        if iteration < max_iterations - 1:
            error_log = output_path / f"compile_iter{iteration+1}.log"
            if error_log.exists():
                error_feedback = error_log.read_text()
                print(f"[{iteration+1}] Compilation failed, will retry with error feedback")
            else:
                print(f"[{iteration+1}] No compilation attempted, cannot iterate")
                break

    # Max iterations reached without success
    print(f"\n{'='*70}")
    print(f"Max iterations ({max_iterations}) reached without success")
    print(f"{'='*70}")

    metadata = {
        "status": "failed",
        "iterations": max_iterations,
        "timestamp": datetime.now().isoformat()
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return False


def try_compile(source_file, output_dir, iteration):
    """Attempt to compile generated SYCL code.

    Args:
        source_file: Path to generated SYCL code
        output_dir: Output directory
        iteration: Current iteration number

    Returns:
        bool: True if compilation succeeded
    """
    import subprocess

    print(f"[{iteration}] Attempting compilation...")

    # Check if compiler is available
    try:
        result = subprocess.run(
            ["which", "icpx"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )

        if result.returncode != 0:
            print(f"[{iteration}] Compiler (icpx) not available, skipping compilation")
            return False

    except Exception:
        print(f"[{iteration}] Could not check for compiler, skipping compilation")
        return False

    # Try to compile
    log_file = output_dir / f"compile_iter{iteration}.log"
    exe_file = output_dir / f"app_iter{iteration}"

    cmd = [
        "icpx", "-fsycl",
        "-o", str(exe_file),
        str(source_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=60
        )

        # Save compilation log
        log_file.write_text(result.stdout)

        if result.returncode == 0:
            print(f"[{iteration}] ✓ Compilation successful")
            return True
        else:
            print(f"[{iteration}] ✗ Compilation failed (see {log_file})")
            return False

    except subprocess.TimeoutExpired:
        print(f"[{iteration}] ✗ Compilation timed out")
        return False
    except Exception as e:
        print(f"[{iteration}] ✗ Compilation error: {e}")
        return False


def main():
    """Main entry point."""

    # Configuration - can be overridden by environment variables
    source_file = os.environ.get("SOURCE_FILE", "example_cuda_code.cu")
    output_dir = os.environ.get("OUTPUT_DIR", f"generated_codes/exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    model = os.environ.get("MODEL", "gpt52")
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "5"))

    # Run translation
    success = run_translation(source_file, output_dir, model, max_iterations)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
