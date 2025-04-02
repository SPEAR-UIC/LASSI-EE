import subprocess
import html
import re
from IPython.display import display, HTML

def argo_fetch_with_curl(data):
    url = "https://localhost:5000/argoapi/api/v1/resource/streamchat/"
    headers = [
        "-H", "Content-Type: application/json",
        "-H", "Host: apps-dev.inside.anl.gov"
    ]

    # Create an HTML display object for live streaming output
    html_display = display(HTML(""), display_id=True)

    # Run curl as a subprocess, capturing output as it streams
    process = subprocess.Popen(
        ["curl", "-k", "-X", "POST", url, *headers, "-d", data],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line-buffered for real-time output
    )

    response_text = ""

    # Read output line by line to maintain real-time streaming
    for line in iter(process.stdout.readline, ''):
        # Append new content and escape HTML characters for safety
        response_text += line
        escaped_text = html.escape(response_text)

        # (Optional) If you want to add specific styling to tags, e.g. <think> tags:
        styled_text = re.sub(
            r'&lt;think&gt;',
            '<span style="color: #d32f2f; font-weight: bold;">&lt;think&gt;</span>',
            escaped_text
        )
        styled_text = re.sub(
            r'&lt;/think&gt;',
            '<span style="color: #d32f2f; font-weight: bold;">&lt;/think&gt;</span>',
            styled_text
        )
        
        # Create typewriter effect with a blinking cursor
        display_text = f"{styled_text}<span style='border-right:2px solid;animation:blink 1s infinite'> </span>"
        
        # Update the display output with styled text
        # Use dedent to remove any leading whitespace from the HTML template
        # Put everything on one line to avoid extra whitespace
        html_content = (
            f"<pre style='background:#f8f8f8;padding:10px;border-radius:5px;border:1px solid #ddd;"
            f"white-space:pre-wrap;font-family:monospace;margin:0;'>{display_text}</pre>"
            f"<style>"
            f"@keyframes blink {{0%{{opacity:1}}50%{{opacity:0}}100%{{opacity:1}}}}"
            f"</style>"
        )
        
        # Update the display output with styled text
        html_display.update(HTML(html_content))
        
        # print(line, end="", flush=True)  # Print each chunk live
        #output += line  # Capture full response

    process.stdout.close()
    process.wait()  # Ensure process exits cleanly

    return response_text  # Store full response for later use
