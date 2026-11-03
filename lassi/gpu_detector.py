"""
GPU Vendor Detection Utility for LASSI
Detects available NVIDIA and AMD GPUs.
Created: September 16, 2025
Matthew T. Dearing
"""

import subprocess
import os

def detect_gpu_vendors():
    """Detect all available GPU vendors on the system"""
    vendors = []
    
    # Check for NVIDIA GPUs
    try:
        result = subprocess.run(['nvidia-smi', '-L'], 
                              capture_output=True, text=True, check=True)
        if 'GPU' in result.stdout:
            nvidia_count = len([line for line in result.stdout.split('\n') if 'GPU' in line and ':' in line])
            vendors.append({'vendor': 'nvidia', 'count': nvidia_count, 'tool': 'nvidia-smi'})
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check for AMD GPUs  
    try:
        result = subprocess.run(['rocm-smi', '--showid'], 
                              capture_output=True, text=True, check=True)
        if 'GPU' in result.stdout:
            amd_count = len([line for line in result.stdout.split('\n') if 'GPU' in line and ':' in line])
            vendors.append({'vendor': 'amd', 'count': amd_count, 'tool': 'rocm-smi'})
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
        
    return vendors


def get_cuda_arch():
    """Return the first NVIDIA GPU's compute capability as an nvcc arch string (e.g. 'sm_80'), or None."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True)
        major, minor = result.stdout.strip().splitlines()[0].strip().split('.')
        return f"sm_{major}{minor}"
    except Exception:
        return None


def check_gpu_runtime(vendor):
    """
    Check that the GPU *compute* runtime can initialize, not just that the device is visible.

    `nvidia-smi -L` and `rocm-smi` can succeed while CUDA/HIP initialization is broken (a stale
    nvidia_uvm module, a driver/library mismatch, ...). Benchmark programs that don't check CUDA
    errors then run on garbage and still print PASS, so every energy number is meaningless.

    Returns (ok, message): ok is True or False, or None if the check could not be performed
    (e.g. the runtime library isn't on the loader path); None should be treated as a warning.
    """
    import ctypes
    try:
        if vendor == "nvidia":
            lib = ctypes.CDLL("libcuda.so.1")
            code = lib.cuInit(0)
            if code != 0:
                return False, (f"CUDA driver initialization failed (cuInit returned {code}). "
                               "nvidia-smi can still work in this state. Error 999 is often a stale nvidia_uvm "
                               "module or another process (e.g. an Ollama service) holding the GPU: stop it and "
                               "run `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`, or reboot.")
            count = ctypes.c_int(0)
            lib.cuDeviceGetCount(ctypes.byref(count))
            if count.value < 1:
                return False, "CUDA initialized but reports no devices."
            return True, f"CUDA runtime OK ({count.value} device(s))"
        if vendor == "amd":
            lib = ctypes.CDLL("libamdhip64.so")
            code = lib.hipInit(0)
            if code != 0:
                return False, f"HIP runtime initialization failed (hipInit returned {code})."
            return True, "HIP runtime OK"
    except OSError as e:
        return None, f"Could not load the {vendor} GPU runtime library to verify it ({e})."
    except Exception as e:
        return None, f"GPU runtime check could not be completed ({e})."
    return None, f"No runtime check available for vendor {vendor!r}."


def test_gpu_detection():
    """Test function to verify GPU detection works"""
    print("=== LASSI GPU Vendor Detection Test ===")
    
    vendors = detect_gpu_vendors()
    
    if not vendors:
        print("❌ No GPUs detected")
        return False
    
    for vendor_info in vendors:
        vendor = vendor_info['vendor']
        count = vendor_info['count']
        tool = vendor_info['tool']
        print(f"✅ {vendor.upper()}: {count} GPU(s) detected via {tool}")
    
    return True

if __name__ == "__main__":
    test_gpu_detection()