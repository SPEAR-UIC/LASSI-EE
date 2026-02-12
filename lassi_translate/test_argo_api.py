#!/usr/bin/env python3
"""Minimal test script for Argo API connection.

This script sends a simple test request to the Argo API to verify connectivity
and basic functionality.
"""

import sys
import os

# Add lassi to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lassi'))

from llm_config import argo_llm


def test_argo_simple():
    """Test a simple Argo API call with minimal prompt."""
    print("Testing Argo API connection...")
    print("-" * 50)

    model = "gpt52"
    system_prompt = "You are a helpful assistant."
    content_prompt = "Say 'Hello, Argo API is working!' and nothing else."

    try:
        response = argo_llm(
            model=model,
            system_prompt=system_prompt,
            content_prompt=content_prompt,
            temp=0.2,
            verifyprompt=False
        )

        print("Response received:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        print("✓ Test successful!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_argo_simple()
    sys.exit(0 if success else 1)
