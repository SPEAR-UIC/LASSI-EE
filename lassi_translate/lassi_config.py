config = {
    "resource": "polaris",
    "task": "translate",
    "alcf_username": "vchirio",
    "app_category": "Miniapps",
    "app_name": "XSBench",
    "account": "EE-ECP",
    "queue": "debug",
    "walltime": 3600,
    "argo_model": "gpt55",
    "inference_model": "openai/gpt-oss-120b",
    "inference_cluster": "sophia",
    "min_energy_gain": 5.0,
    "use_judge": True,
    # Temperature ratchet: auto-bump the LLM call temperature when the pipeline
    # gets stuck on the same error type 3+ times.  False holds the call temp at
    # 0.2 for every attempt.  Override per run with --temp-ratchet / --no-temp-ratchet.
    "ratchet_temperature": True,
}
