#!/usr/bin/env python3
"""Argo API test script for PBS job submission.

This script tests the Argo API connection and saves detailed output
to a log file for verification after the job completes.
"""

import sys
import os
from datetime import datetime
import traceback

# Add lassi to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lassi'))

from llm_config import argo_llm


def log_message(msg, log_file=None):
    """Log message to both stdout and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    if log_file:
        log_file.write(formatted_msg + "\n")
        log_file.flush()


def test_argo_connection(log_file):
    """Test Argo API connection with detailed logging."""
    log_message("=" * 70, log_file)
    log_message("ARGO API CONNECTION TEST", log_file)
    log_message("=" * 70, log_file)

    # Log environment info
    log_message(f"Hostname: {os.environ.get('HOSTNAME', 'unknown')}", log_file)
    log_message(f"User: {os.environ.get('USER', 'unknown')}", log_file)
    log_message(f"Working directory: {os.getcwd()}", log_file)
    log_message(f"Python version: {sys.version}", log_file)

    # Check proxy configuration
    proxy_in_code = "https://proxy.alcf.anl.gov:3128"
    log_message(f"Proxy configured in code: {proxy_in_code}", log_file)

    log_message("=" * 70, log_file)

    # Test parameters
    model = "gpt52"
    system_prompt = "You are a helpful assistant."
    content_prompt = "Say 'Hello, Argo API is working from compute node!' and nothing else."

    log_message(f"Model: {model}", log_file)
    log_message(f"System prompt: {system_prompt}", log_file)
    log_message(f"Content prompt: {content_prompt}", log_file)
    log_message("=" * 70, log_file)

    try:
        log_message("Sending request to Argo API...", log_file)

        response = argo_llm(
            model=model,
            system_prompt=system_prompt,
            content_prompt=content_prompt,
            temp=0.2,
            verifyprompt=False
        )

        log_message("SUCCESS: Response received from Argo API", log_file)
        log_message("=" * 70, log_file)
        log_message("RESPONSE:", log_file)
        log_message("-" * 70, log_file)
        log_message(response, log_file)
        log_message("-" * 70, log_file)
        log_message("=" * 70, log_file)
        log_message("TEST COMPLETED SUCCESSFULLY", log_file)
        log_message("=" * 70, log_file)

        return True

    except Exception as e:
        log_message("FAILURE: Error occurred during API call", log_file)
        log_message("=" * 70, log_file)
        log_message(f"Error type: {type(e).__name__}", log_file)
        log_message(f"Error message: {str(e)}", log_file)
        log_message("=" * 70, log_file)
        log_message("Full traceback:", log_file)
        log_message(traceback.format_exc(), log_file)
        log_message("=" * 70, log_file)
        log_message("TROUBLESHOOTING STEPS:", log_file)
        log_message("1. Verify proxy is accessible: curl -x https://proxy.alcf.anl.gov:3128 https://apps-dev.inside.anl.gov/", log_file)
        log_message("2. Check network connectivity from compute node", log_file)
        log_message("3. Verify Argo API endpoint is up", log_file)
        log_message("4. Check if curl is available: which curl", log_file)
        log_message("=" * 70, log_file)

        return False


def main():
    """Main entry point."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"argo_test_{timestamp}.log"
    log_filepath = os.path.join(output_dir, log_filename)

    print(f"Log file: {log_filepath}")

    with open(log_filepath, 'w') as log_file:
        success = test_argo_connection(log_file)

        # Write final status
        log_message("=" * 70, log_file)
        if success:
            log_message("FINAL STATUS: SUCCESS ✓", log_file)
            exit_code = 0
        else:
            log_message("FINAL STATUS: FAILURE ✗", log_file)
            exit_code = 1
        log_message("=" * 70, log_file)

    print(f"\nTest complete. Check log file: {log_filepath}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
