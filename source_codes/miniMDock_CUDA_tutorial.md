# miniMDock: Build & Run Guide (CUDA + NVIDIA GPU)

---

## 1. Environment Setup

Make sure your system has:

- **CUDA Toolkit ≥ 8.0** (Recommended: ≥ 11.0; tested on CUDA 12.6)
- **GCC compiler** (e.g., GCC 9+ or 13+)
- An NVIDIA GPU (e.g., A100)

Then, set the required environment variables:

```bash
export GPU_INCLUDE_PATH=/usr/local/cuda/include
export GPU_LIBRARY_PATH=/usr/local/cuda/lib64
```

---

## 2. Add A100 Support to Makefile

Open the `Makefile` and update the architecture list to include A100 (compute capability 8.0):

```make
TARGETS = 52 61 70 80
```

This ensures that the binary supports your GPU (`sm_80`).

---

## 3. Compile miniMDock (CUDA Version)

Clean previous builds and compile with:

```bash
make clean
make API=CUDA DEVICE=GPU CARD=NVIDIA NUMWI=64
```

This will produce a binary such as:

```bash
./bin/autodock_gpu_64wi
```

You may optionally rename it for clarity:

```bash
mv ./bin/autodock_gpu_64wi ./bin/autodock_cuda_nv_64wi
```

---

## 4. Run miniMDock

Use the following command to launch docking:

```bash
./bin/autodock_cuda_nv_64wi -lfile ./input/7cpa/7cpa_ligand.pdbqt -nrun 100
```

### To increase run time:

- Increase the number of runs (e.g., `-nrun 100`)
- Use more complex ligand files from the `input/` directory
- Run multiple ligands sequentially or in batch

Example:

```bash
./bin/autodock_cuda_nv_64wi -lfile ./input/complex_ligand.pdbqt -nrun 100
```

---

## �� Notes

- The ligand file must be in `.pdbqt` format.
- Output logs and runtime messages will be printed to stdout.
- Be sure to check CUDA compatibility if switching systems.

---

## �� Tested System

- **OS:** Ubuntu 24.04  
- **GPU:** NVIDIA A100 (CUDA Compute Capability 8.0)  
- **CUDA Version:** 12.6  
- **GCC Version:** 13.3  

---

## �� Project Structure & File Dependencies

This diagram illustrates the high-level source code structure and function call relationships within `miniMDock`:

![Project Structure](miniMDock_structure.png)

---

## �� License

miniMDock is available under GNU GPL v2 and/or LGPL v2.1.