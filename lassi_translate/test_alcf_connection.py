#!/usr/bin/env python3
"""Simple test script to verify ALCF Inference Endpoints connectivity.

This script tests:
1. Token retrieval
2. API connectivity
3. Simple LLM inference request
"""

import sys
import os
from datetime import datetime

# Add lassi to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lassi'))


def test_alcf_connection():
    """Test ALCF API connection with a simple request."""
    print("="*70)
    print("ALCF INFERENCE ENDPOINTS CONNECTION TEST")
    print("="*70)
    print(f"Timestamp: {datetime.now()}")
    print(f"Hostname: {os.environ.get('HOSTNAME', 'unknown')}")
    print("="*70)

    # Test 1: Check token helper
    print("\n[TEST 1] Checking for token helper...")
    token_helper = os.path.expanduser("~/inference_auth_token.py")
    if os.path.exists(token_helper):
        print(f"✓ Token helper found: {token_helper}")
    else:
        print(f"✗ Token helper not found at: {token_helper}")
        print("\nTo set up authentication:")
        print("1. Download: wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py")
        print("2. Authenticate: python inference_auth_token.py authenticate")
        return False

    # Test 2: Get token
    print("\n[TEST 2] Retrieving access token...")
    try:
        from llm_config_alcf import get_alcf_token
        token = get_alcf_token()
        print(f"✓ Token retrieved successfully (length: {len(token)})")
        print(f"  Token preview: {token[:20]}...")
    except Exception as e:
        print(f"✗ Token retrieval failed: {e}")
        return False

    # Test 3: Simple API call
    print("\n[TEST 3] Testing ALCF API with simple request...")
    print("Model: meta-llama/Llama-3.3-70B-Instruct")
    print("Cluster: sophia")

    try:
        from llm_config_alcf import alcf_llm

        response = alcf_llm(
            model="meta-llama/Llama-3.3-70B-Instruct",
            system_prompt="You are a helpful assistant.",
            content_prompt="Say 'Hello from ALCF!' and nothing else.",
            temp=0.2,
            verifyprompt=False,
            cluster="sophia"
        )

        print("✓ API call successful!")
        print("\nResponse:")
        print("-"*70)
        print(response)
        print("-"*70)

    except Exception as e:
        print(f"✗ API call failed: {e}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        return False

    # Test 4: Test drop-in replacement
    print("\n[TEST 4] Testing drop-in argo_llm replacement...")
    try:
        from llm_config_alcf import argo_llm

        response = argo_llm(
            model="gpt52",  # Will be mapped to Llama model
            system_prompt="You are a helpful assistant.",
            content_prompt="Count to 3.",
            temp=0.2,
            verifyprompt=False
        )

        print("✓ Drop-in replacement works!")
        print("\nResponse:")
        print("-"*70)
        print(response)
        print("-"*70)

    except Exception as e:
        print(f"✗ Drop-in replacement failed: {e}")
        return False

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nALCF Inference Endpoints are working correctly!")
    print("You can now use llm_config_alcf.py in your workflows.")
    print("="*70)

    return True


if __name__ == "__main__":
    success = test_alcf_connection()
    sys.exit(0 if success else 1)
