# Sweep Report — `sweep_20260608_205153`

**Task:** CUDA → SYCL translation of XSBench (Miniapps, large/event-based)
**Target:** Polaris (NVIDIA A100, icpx CUDA backend)
**Self-correction cap:** 15 attempts (16th iteration reported when the loop exhausts)
**Sweep composition:** 30 runs total — 10 per model — 30 s politeness sleep between runs

---

## Per-model summary

| Model | Mode | Success rate | Avg attempts (successful) | Avg pipeline runtime (successful) | Avg translation runtime (successful) |
|---|---|---:|---:|---:|---:|
| **Argo `gpt55`**          | remote  | **10 / 10 = 100 %** | **1.4** | **10.04 min** | **4.80 s** |
| **Argo `gpt41`**          | remote  | **5 / 10 = 50 %**   | **3.6** | **10.88 min** | **5.08 s** |
| **ALCF Inference `openai/gpt-oss-120b`** | `--submit` PBS | **5 / 10 = 50 %** | **9.4** | **18.33 min** | **5.23 s** |

- *Success rate* = fraction of trials that produced a SYCL translation that compiled, executed on the GPU, and returned XSBench checksum `952131`.
- *Avg attempts (successful)* averages the LASSI `Self-corr + 1` count across successful runs only (1 = succeeded on first attempt with no self-correction).
- *Avg pipeline runtime* is wall-clock time per successful run (orchestrator boot → final write of `results.csv`), measured from the sweep manifest's `elapsed_s`.
- *Avg translation runtime* is the GPU execution time of the **final accepted SYCL translation** (`translation_runtime_s` in `results.csv`).

---

## Per-trial detail

### `gpt55` (remote, Argo)
| Run | Outcome | Attempts | Pipeline (min) | Translation runtime (s) |
|---:|:---:|---:|---:|---:|
|  1 | ✓ |  1 | 14.90 | 4.714 |
|  2 | ✓ |  2 | 11.15 | 4.751 |
|  3 | ✓ |  1 |  8.25 | 4.730 |
|  4 | ✓ |  2 | 11.19 | 4.604 |
|  5 | ✓ |  1 |  8.44 | 4.908 |
|  6 | ✓ |  2 | 10.88 | 4.905 |
|  7 | ✓ |  1 |  8.27 | 4.906 |
|  8 | ✓ |  1 |  8.26 | 4.776 |
|  9 | ✓ |  2 | 10.45 | 4.699 |
| 10 | ✓ |  1 |  8.62 | 4.963 |

### `gpt41` (remote, Argo)
| Run | Outcome | Attempts | Pipeline (min) | Translation runtime (s) |
|---:|:---:|---:|---:|---:|
| 11 | ✗ | 16 | — | — |
| 12 | ✓ |  3 | 11.92 | 4.610 |
| 13 | ✓ |  3 | 10.36 | 7.097 |
| 14 | ✗ | 10 | — | — |
| 15 | ✗ |  8 | — | — |
| 16 | ✓ |  4 | 11.63 | 4.349 |
| 17 | ✓ |  3 |  8.39 | 4.568 |
| 18 | ✓ |  5 | 12.10 | 4.798 |
| 19 | ✗ | 13 | — | — |
| 20 | ✗ | 16 | — | — |

### `openai/gpt-oss-120b` (`--submit`, ALCF Inference / sophia)
| Run | Outcome | Attempts | Pipeline (min) | Translation runtime (s) |
|---:|:---:|---:|---:|---:|
| 21 | ✓ | 14 | 23.41 | 7.665 |
| 22 | ✓ |  8 | 19.50 | 4.611 |
| 23 | ✓ | 12 | 17.18 | 4.743 |
| 24 | ✓ |  7 | 16.50 | 4.563 |
| 25 | ✗ | 16 | — | — |
| 26 | ✗ | 16 | — | — |
| 27 | ✓ |  6 | 15.05 | 4.562 |
| 28 | ✗ | 16 | — | — |
| 29 | ✗ | 16 | — | — |
| 30 | ✗ | 16 | — | — |

---

## Observations

- **`gpt55` dominates** — every trial succeeded, and it averaged 1.4 attempts: the model translates XSBench either correctly on the first shot or with one round of self-correction.
- **`gpt41` and `gpt-oss-120b` tie on success rate** at 5 / 10, but they get there very differently:
  - `gpt41` succeeds quickly when it succeeds (≈ 3.6 attempts, ~11 min) and fails relatively early (8 – 16 attempts).
  - `gpt-oss-120b` *grinds* to its successes (≈ 9.4 attempts, ~18 min) and all 5 failures exhausted the 15-attempt cap.
- **Final-translation GPU runtime is essentially identical across all three models** (4.8 – 5.2 s mean, on a single A100). The pipeline produces functionally equivalent code regardless of which model finds it; the differentiator is *how reliably* and *how quickly* a model arrives at a valid translation.

---

## Sources
- `sweeps/sweep_20260608_205153/manifest.json` (run → exp_dir mapping, per-run wall times)
- `generated_codes/experiment/<exp_dir>/results.csv` (success flag, attempts, translation runtime)
- `generated_codes/experiment/<exp_dir>/attempts.json` (per-attempt failure-mode log, used for failure-mode analysis upstream)
