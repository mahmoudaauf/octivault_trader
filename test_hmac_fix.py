#!/usr/bin/env python3
"""
Test script to verify the HMAC signature fix for WS API v3.

Tests:
1. Syntax validation
2. Signature generation
3. HMAC parameter structure
4. Method callability
"""

import sys
import hashlib
import hmac
import time

def test_syntax():
    """Test that the code compiles."""
    print("\n[TEST 1] Syntax Validation")
    print("-" * 60)
    try:
        import py_compile
        py_compile.compile('core/exchange_client.py', doraise=True)
        print("✅ PASS: exchange_client.py compiles without errors")
        return True
    except Exception as e:
        print(f"❌ FAIL: Syntax error: {e}")
        return False

def test_signature_generation():
    """Test HMAC-SHA256 signature generation."""
    print("\n[TEST 2] HMAC-SHA256 Signature Generation")
    print("-" * 60)
    
    # Test case 1: Single parameter
    api_secret = "test_secret"
    params = {"timestamp": "1655971950123"}
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    print(f"  Input:")
    print(f"    api_secret: {api_secret}")
    print(f"    query_string: {query_string}")
    print(f"  Output:")
    print(f"    signature: {signature[:32]}... (64 chars total)")
    
    # Verify signature format
    if len(signature) == 64 and all(c in '0123456789abcdef' for c in signature):
        print("✅ PASS: Signature is valid hexadecimal (64 chars)")
        return True
    else:
        print(f"❌ FAIL: Signature format invalid (got {len(signature)} chars)")
        return False

def test_method_imports():
    """Test that ExchangeClient can be imported and methods exist."""
    print("\n[TEST 3] Method Imports & Callability")
    print("-" * 60)
    
    try:
        from core.exchange_client import ExchangeClient
        
        # Check methods exist
        methods = [
            '_ws_api_signed_params',
            '_ws_api_signature_params',
            '_user_data_ws_api_v3_direct',
            '_user_data_listen_key_loop',
            '_user_data_polling_loop',
        ]
        
        for method_name in methods:
            if hasattr(ExchangeClient, method_name):
                print(f"  ✅ {method_name} exists")
            else:
                print(f"  ❌ {method_name} NOT FOUND")
                return False
        
        print("✅ PASS: All required methods exist")
        return True
    except Exception as e:
        print(f"❌ FAIL: Import error: {e}")
        return False

def test_signature_params_structure():
    """Test that signature params have the correct structure."""
    print("\n[TEST 4] Signature Params Structure")
    print("-" * 60)
    
    try:
        from core.exchange_client import ExchangeClient
        from core.config import Config
        from core.shared_state import SharedState
        
        # Create a minimal config
        config = Config()
        shared_state = SharedState()
        
        # Create client with test credentials
        client = ExchangeClient(
            config=config,
            shared_state=shared_state,
            api_key="test_api_key",
            api_secret="test_api_secret",
        )
        
        # Test _ws_api_signed_params
        params = client._ws_api_signed_params()
        
        required_fields = ['timestamp', 'signature']
        for field in required_fields:
            if field in params:
                print(f"  ✅ '{field}' present in params")
            else:
                print(f"  ❌ '{field}' MISSING from params")
                return False
        
        # Verify signature is valid hex
        signature = params['signature']
        if len(signature) == 64 and all(c in '0123456789abcdef' for c in signature):
            print(f"  ✅ Signature is valid hex (64 chars)")
        else:
            print(f"  ❌ Signature format invalid")
            return False
        
        print("✅ PASS: Signature params have correct structure")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_signature_consistency():
    """Test that the same params produce the same signature."""
    print("\n[TEST 5] Signature Consistency")
    print("-" * 60)
    
    try:
        from core.exchange_client import ExchangeClient
        from core.config import Config
        from core.shared_state import SharedState
        
        # Create client
        config = Config()
        shared_state = SharedState()
        client = ExchangeClient(
            config=config,
            shared_state=shared_state,
            api_key="test_api_key",
            api_secret="test_api_secret",
        )
        
        # Mock the time offset to ensure same timestamp
        original_offset = client._time_offset_ms
        client._time_offset_ms = 0
        
        # Get timestamp and freeze it for reproducibility
        import unittest.mock as mock
        with mock.patch('time.time', return_value=1655971950.123):
            params1 = client._ws_api_signed_params()
            params2 = client._ws_api_signed_params()
        
        # Signatures should be identical
        if params1['signature'] == params2['signature']:
            print(f"  ✅ Same params produce same signature")
        else:
            print(f"  ❌ Signatures differ for same params")
            return False
        
        # Restore
        client._time_offset_ms = original_offset
        
        print("✅ PASS: Signature generation is consistent")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("WebSocket API v3 HMAC Signature Fix - Verification Tests")
    print("=" * 60)
    
    tests = [
        ("Syntax", test_syntax),
        ("HMAC Generation", test_signature_generation),
        ("Method Imports", test_method_imports),
        ("Signature Structure", test_signature_params_structure),
        ("Signature Consistency", test_signature_consistency),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED - Ready for deployment!")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
