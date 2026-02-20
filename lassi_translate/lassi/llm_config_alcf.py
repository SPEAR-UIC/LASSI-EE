import json
import os
import subprocess
import threading
import time


_MAX_RETRIES = 3
_RETRY_DELAYS = [30, 60]  # seconds between retries (exponential backoff)

# Printed once when running on a compute node (not per-call).
_ROUTING_LOGGED = False

# SSE parser written to a temp file alongside the JSON payload.
# Protocol:
#   T:<json-encoded delta>  — one line per token chunk, written immediately
#   R:<json response>       — one line at the end (standard non-streaming format)
# If no SSE data found (error body, non-streaming fallback), passes raw through
# as R:<raw> so caller's existing error-detection logic fires unchanged.
_SSE_PARSER_PY = """\
import sys, json
parts = []
raw_chunks = []
for raw in sys.stdin:
    raw_chunks.append(raw)
    line = raw.strip()
    if not line or not line.startswith('data: '):
        continue
    payload = line[6:]
    if payload == '[DONE]':
        break
    try:
        delta = json.loads(payload)['choices'][0]['delta'].get('content', '')
        if delta:
            parts.append(delta)
            print('T:' + json.dumps(delta), flush=True)
    except Exception:
        pass
if parts:
    print('R:' + json.dumps({'choices': [{'message': {'content': ''.join(parts)}}]}), flush=True)
else:
    sys.stdout.write('R:' + ''.join(raw_chunks).strip() + '\\n')
    sys.stdout.flush()
"""


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temp: float = 0.2,
    verifyprompt: bool = False,
    max_tokens: int = 32768,
    on_token=None,
    debug: bool = False,
) -> str:
    global _ROUTING_LOGGED

    cluster = os.environ.get("ALCF_CLUSTER", "metis")
    endpoints = {
        "sophia": "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1/chat/completions",
        "metis":  "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1/chat/completions",
    }
    if cluster not in endpoints:
        raise ValueError(f"Unknown cluster: {cluster!r}. Choose 'sophia' or 'metis'")
    url = endpoints[cluster]

    if verifyprompt:
        print(f"\n=====> PROMPT REVIEW <=====")
        print(f"CLUSTER: {cluster}  MODEL: {model}  TEMP: {round(temp, 1)}")
        print(f"\nSYSTEM:\n{system_prompt}")
        print(f"\nUSER:\n{user_prompt}")
        print("=====^^^^^^^^^^^^^^^^^=====\n")

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": round(temp, 1),
        "max_tokens": max_tokens,
        "stream": True,
    }
    payload = json.dumps(data)

    conda_env = os.environ.get("ALCF_CONDA_ENV", "globus_env")

    # Write JSON payload and SSE parser to temp files — avoids ARG_MAX limits
    # and shell escaping issues with large prompts or the multi-line parser.
    shell_script = "\n".join([
        "module load frameworks 2>/dev/null",
        f'access_token=$(conda run -n {conda_env} python3 $HOME/inference_auth_token.py get_access_token)',
        'if [ -z "$access_token" ]; then echo "ERROR: empty token from inference_auth_token.py" >&2; exit 1; fi',
        'LASSI_TMP=$(mktemp /tmp/lassi_payload_XXXXXX.json)',
        'LASSI_PARSER=$(mktemp /tmp/lassi_parser_XXXXXX.py)',
        "cat > \"$LASSI_TMP\" << 'LASSI_PAYLOAD_EOF'",
        payload,
        'LASSI_PAYLOAD_EOF',
        "cat > \"$LASSI_PARSER\" << 'LASSI_PARSER_EOF'",
        _SSE_PARSER_PY,
        'LASSI_PARSER_EOF',
        (
            f"curl -sS --connect-timeout 10 --max-time 600 --no-buffer"
            f" -X POST '{url}'"
            f" -H \"Authorization: Bearer $access_token\""
            f" -H 'Content-Type: application/json'"
            f" -d @\"$LASSI_TMP\""
            f" | python3 \"$LASSI_PARSER\""
        ),
        'rm -f "$LASSI_TMP" "$LASSI_PARSER"',
    ])

    hostname = subprocess.run(
        ["hostname"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ).stdout.strip()

    if hostname.startswith("x"):
        # On compute node — SSH to login node to access the internet.
        login_node = os.environ.get("PBS_O_HOST", "aurora-uan-0009.hostmgmt.cm.aurora.alcf.anl.gov")
        if debug and not _ROUTING_LOGGED:
            print(f"[Compute node detected] Routing API calls via login node: {login_node}")
        _ROUTING_LOGGED = True
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
               login_node, "bash", "-l"]
        timeout = 660
    else:
        cmd = ["bash", "-l"]
        timeout = 640

    last_error: Exception = RuntimeError("No attempt made")
    for attempt in range(_MAX_RETRIES):
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        # Write shell script to stdin in a background thread to prevent deadlock
        # if the SSH/bash process fills its stdout pipe before reading all stdin.
        def _write_stdin(p=proc, script=shell_script):
            try:
                p.stdin.write(script)
                p.stdin.close()
            except OSError:
                pass

        stdin_thread = threading.Thread(target=_write_stdin, daemon=True)
        stdin_thread.start()

        # Drain stderr in a background thread to prevent pipe-buffer deadlock.
        stderr_buf: list[str] = []

        def _drain_stderr(p=proc, buf=stderr_buf):
            try:
                buf.append(p.stderr.read(2000))
            except OSError:
                pass

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        # Kill the process if it exceeds the timeout.
        _timed_out = [False]

        def _kill(p=proc, flag=_timed_out):
            flag[0] = True
            try:
                p.kill()
            except OSError:
                pass

        timer = threading.Timer(timeout, _kill)
        timer.start()

        response_line: str | None = None
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line.startswith("T:"):
                    if on_token is not None:
                        try:
                            on_token(json.loads(line[2:]))
                        except Exception:
                            pass
                elif line.startswith("R:"):
                    response_line = line[2:]
            proc.wait()
        except Exception:
            pass
        finally:
            timer.cancel()
            stdin_thread.join(timeout=2)
            stderr_thread.join(timeout=2)

        if _timed_out[0]:
            proc.wait()
            last_error = RuntimeError(f"ALCF API call timed out after {timeout}s")
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[RETRY] Timeout on attempt {attempt + 1}/{_MAX_RETRIES}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise last_error

        if proc.returncode != 0:
            stderr_snippet = (stderr_buf[0] if stderr_buf else "").strip()[:500]
            last_error = RuntimeError(
                f"ALCF API call failed (rc={proc.returncode}):\n"
                f"  stderr: {stderr_snippet}\n"
                f"  response: {(response_line or '')[:200]}"
            )
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[RETRY] Non-zero exit on attempt {attempt + 1}/{_MAX_RETRIES}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise last_error

        stdout_stripped = (response_line or "").strip()
        if not stdout_stripped or (stdout_stripped.startswith("error code:") and not stdout_stripped.startswith("{")):
            last_error = RuntimeError(f"ALCF endpoint returned transient error: {stdout_stripped[:200]}")
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[RETRY] Transient error on attempt {attempt + 1}/{_MAX_RETRIES}: {stdout_stripped[:80]}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise last_error

        try:
            response = json.loads(stdout_stripped)
            if "choices" in response and response["choices"]:
                return response["choices"][0]["message"]["content"]
            last_error = RuntimeError(f"Unexpected response format: {stdout_stripped[:200]}")
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[RETRY] Unexpected format on attempt {attempt + 1}/{_MAX_RETRIES}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise last_error
        except json.JSONDecodeError as e:
            last_error = RuntimeError(f"Failed to parse JSON response: {e}\nResponse: {stdout_stripped[:500]}")
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"[RETRY] JSON parse error on attempt {attempt + 1}/{_MAX_RETRIES}. Retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise last_error

    raise last_error
