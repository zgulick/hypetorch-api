#!/usr/bin/env python3
"""
Test script for new API v2 endpoints
Run this to verify the new endpoints work correctly
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v2")
API_KEY = os.environ.get("API_KEY", "dev-key-12345")  # Use dev key for testing

def test_endpoint(endpoint, method="GET", data=None, params=None):
    """Test a single endpoint"""
    headers = {"X-API-Key": API_KEY}
    
    try:
        if method == "GET":
            response = requests.get(f"{API_BASE_URL}{endpoint}", headers=headers, params=params)
        elif method == "POST":
            response = requests.post(f"{API_BASE_URL}{endpoint}", headers=headers, json=data)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {method} {endpoint} - Success")
            if "data" in data:
                if isinstance(data["data"], list):
                    print(f"   📊 Returned {len(data['data'])} items")
                elif isinstance(data["data"], dict):
                    print(f"   📊 Returned object with keys: {list(data['data'].keys())}")
            return True
        else:
            print(f"❌ {method} {endpoint} - Error {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('message', 'Unknown error')}")
            except:
                print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ {method} {endpoint} - Exception: {e}")
        return False

def main():
    """Test all new endpoints"""
    print("🧪 Testing HypeTorch API v2 New Endpoints")
    print("=" * 50)
    
    # Test new endpoints
    endpoints_to_test = [
        # Basic health check
        ("/health", "GET", None, None),
        
        # Trending entities
        ("/trending", "GET", None, {"metric": "hype_score", "limit": 5}),
        ("/trending", "GET", None, {"metric": "rodmn_score", "limit": 3, "category": "Sports"}),
        
        # Recent metrics
        ("/metrics/recent", "GET", None, {"period": "current", "limit": 5}),
        ("/metrics/recent", "GET", None, {"period": "week_2025_07_27", "limit": 3}),
        
        # Dashboard widgets
        ("/dashboard/widgets", "GET", None, None),
        
        # Time periods
        ("/time-periods", "GET", None, None),
        
        # Existing endpoints (should still work)
        ("/entities", "GET", None, {"include_metrics": True, "category": "Sports"}),
        ("/entities/search", "GET", None, {"q": "Caitlin", "limit": 3}),
    ]
    
    passed = 0
    total = len(endpoints_to_test)
    
    for endpoint, method, data, params in endpoints_to_test:
        if test_endpoint(endpoint, method, data, params):
            passed += 1
        print()  # Add spacing
    
    print("=" * 50)
    print(f"📈 Test Results: {passed}/{total} endpoints working")
    
    if passed == total:
        print("🎉 All endpoints are working correctly!")
    else:
        print("⚠️ Some endpoints need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)