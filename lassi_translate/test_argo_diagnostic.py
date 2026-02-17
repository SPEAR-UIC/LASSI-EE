#!/usr/bin/env python3
"""Comprehensive diagnostic test for Argo API connectivity.

Tests each component separately to identify the exact failure point.
"""

import sys
import os
import subprocess
from datetime import datetime
import traceback

# Add lassi to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lassi'))


def run_command(cmd, description, timeout=30):
    """Run a command and return success status and output."""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*70}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout
        )

        print(f"Exit Code: {proc.returncode}")
        if proc.stdout:
            print(f"STDOUT:\n{proc.stdout[:500]}")
        if proc.stderr:
            print(f"STDERR:\n{proc.stderr[:500]}")

        return proc.returncode == 0, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Command exceeded {timeout} seconds")
        return False, "", "Timeout"
    except Exception as e:
        print(f"ERROR: {e}")
        return False, "", str(e)


def main():
    """Run comprehensive diagnostics."""
    results = {}

    print("="*70)
    print("ARGO API DIAGNOSTIC TEST")
    print("="*70)
    print(f"Hostname: {os.environ.get('HOSTNAME', 'unknown')}")
    print(f"User: {os.environ.get('USER', 'unknown')}")
    print(f"Time: {datetime.now()}")
    print("="*70)

    # Test 1: Basic curl functionality
    success, stdout, stderr = run_command(
        ["curl", "--version"],
        "Verify curl is available"
    )
    results["curl_available"] = success

    # Test 2: DNS resolution for proxy
    success, stdout, stderr = run_command(
        ["nslookup", "proxy.alcf.anl.gov"],
        "DNS resolution for proxy",
        timeout=10
    )
    results["proxy_dns"] = success

    # Test 3: DNS resolution for Argo endpoint
    success, stdout, stderr = run_command(
        ["nslookup", "apps-dev.inside.anl.gov"],
        "DNS resolution for Argo endpoint",
        timeout=10
    )
    results["argo_dns"] = success

    # Test 4: Direct connection to proxy (if possible)
    success, stdout, stderr = run_command(
        ["curl", "-v", "-x", "http://proxy.alcf.anl.gov:3128",
         "--max-time", "10", "http://www.google.com"],
        "Test proxy with external site (google.com)",
        timeout=15
    )
    results["proxy_external"] = success

    # Test 5: Proxy connection to ANL site
    success, stdout, stderr = run_command(
        ["curl", "-v", "-x", "http://proxy.alcf.anl.gov:3128",
         "--max-time", "10", "-k", "https://www.anl.gov"],
        "Test proxy with ANL site",
        timeout=15
    )
    results["proxy_anl"] = success

    # Test 6: Try Argo endpoint without proxy (will likely fail, but diagnostic)
    success, stdout, stderr = run_command(
        ["curl", "-v", "--max-time", "5", "-k",
         "https://apps-dev.inside.anl.gov/"],
        "Direct connection to Argo (no proxy)",
        timeout=10
    )
    results["argo_direct"] = success

    # Test 7: Try Argo endpoint WITH proxy
    success, stdout, stderr = run_command(
        ["curl", "-v", "-x", "http://proxy.alcf.anl.gov:3128",
         "--max-time", "10", "-k", "https://apps-dev.inside.anl.gov/"],
        "Connection to Argo through proxy",
        timeout=15
    )
    results["argo_via_proxy"] = success

    # Test 8: Try the actual API endpoint
    if results.get("argo_via_proxy"):
        success, stdout, stderr = run_command(
            ["curl", "-v", "-x", "http://proxy.alcf.anl.gov:3128",
             "--max-time", "10", "-k",
             "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/streamchat/"],
            "Connection to Argo API endpoint through proxy",
            timeout=15
        )
        results["argo_api_via_proxy"] = success

    # Test 9: Check environment variables
    print("\n" + "="*70)
    print("ENVIRONMENT VARIABLES")
    print("="*70)
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy"]:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    print("\nDetailed Results:")
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if not results.get("curl_available"):
        print("- curl is not available. Install curl to proceed.")
    elif not results.get("proxy_dns"):
        print("- Cannot resolve proxy.alcf.anl.gov. Check DNS configuration.")
    elif not results.get("argo_dns"):
        print("- Cannot resolve apps-dev.inside.anl.gov. Check DNS configuration.")
    elif not results.get("proxy_external") and not results.get("proxy_anl"):
        print("- Proxy is not working for any site. Check proxy configuration.")
        print("- The proxy may require authentication or whitelist access.")
        print("- Try: export http_proxy=http://proxy.alcf.anl.gov:3128")
    elif not results.get("argo_via_proxy"):
        print("- Argo endpoint is not reachable through the proxy.")
        print("- Possible reasons:")
        print("  1. Proxy blocks internal ANL sites (apps-dev.inside.anl.gov)")
        print("  2. Argo endpoint requires direct connection (no proxy)")
        print("  3. Firewall rules prevent compute nodes from accessing Argo")
        print("\nRECOMMENDED ACTIONS:")
        print("  - Check if compute nodes have direct access to internal ANL network")
        print("  - Contact ALCF support about compute node access to apps-dev.inside.anl.gov")
        print("  - Ask if there's a different proxy or route for internal services")
    else:
        print("- All connectivity tests passed!")
        print("- The Argo API should be accessible.")

    print("="*70)

    # Return success if critical tests pass
    critical_passed = results.get("curl_available") and results.get("argo_via_proxy")
    return 0 if critical_passed else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print(traceback.format_exc())
        sys.exit(2)
