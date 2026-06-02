config = {
    "resource":        "polaris",       # "polaris" (NVIDIA A100) | "aurora" (Intel PVC)
    "task":            "energy",        # "energy" | "translate" (CUDA→SYCL) | "both" (translate then energy)
    "input_framework": "cuda",          # "cuda" | "sycl" (sycl only valid for aurora energy task)
    "alcf_username":   "vchirio",       # your ALCF username
    "app_category":    "HeCBench",      # "Miniapps" | "HeCBench" — top-level key in source_apps.json
    "app_name":        "jacobi",       # app key within the category, e.g. "XSBench", "miniMDock", "XSBench_hip"
    "account":         "EE-ECP",        # PBS allocation account
    "queue":           "debug",         # PBS queue, e.g. "debug" | "prod" | "preemptable" | "small"
    "walltime":        3600,            # persistent worker walltime in seconds
    "model":           "claudeopus46",         # Argo model: "gpt4o" | "gpt41" | "gpt55" | "claudesonnet4" | "gpto1preview"
    "min_energy_gain": 5.0,             # minimum % energy reduction to declare success (energy tasks only)
    "use_judge":       True,            # True | False — run LLM-as-judge after each successful compile+run
}
