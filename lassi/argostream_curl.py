import subprocess

def argo_fetch_with_curl(data):
    url = "https://localhost:5000/argoapi/api/v1/resource/streamchat/"
    headers = [
        "-H", "Content-Type: application/json",
        "-H", "Host: apps-dev.inside.anl.gov"
    ]

    # Run curl as a subprocess, capturing output as it streams
    process = subprocess.Popen(
        ["curl", "-k", "-X", "POST", url, *headers, "-d", data],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line-buffered for real-time output
    )

    output = ""

    # Read output line by line to maintain real-time streaming
    for line in iter(process.stdout.readline, ''):
        print(line, end="", flush=True)  # Print each chunk live
        output += line  # Capture full response

    process.stdout.close()
    process.wait()  # Ensure process exits cleanly

    return output  # Store full response for later use
