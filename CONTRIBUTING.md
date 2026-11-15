# Contributing

LASSI-EE is research code accompanying a published paper (see [`CITATION.cff`](CITATION.cff)). The pipeline, prompts and benchmark configuration are tied to specific reported results, so changes that affect reproducibility will be weighed against that.

Bug reports and small fixes are welcome via GitHub issues and pull requests. For larger changes or new features, please open an issue first to discuss the approach before submitting a PR. This is a research project, so reviews are best-effort.

When reporting a problem, please include your GPU and driver (or ROCm) version, the LLM backend and model, and the relevant part of the notebook output. Never include API keys or credentials in an issue or PR.

New HeCBench kernels can be added without code changes; see [Adding a New Kernel](README.md#adding-a-new-kernel).
