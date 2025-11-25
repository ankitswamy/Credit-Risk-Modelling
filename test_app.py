#!/usr/bin/env python3
"""
Test script for Credit Risk Prediction System
This script tests the Flask application endpoints
"""

import requests
import json
import pandas as pd
import time
import sys

def test_health_endpoint(base_url):
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_prediction(base_url):
    """Test single prediction endpoint"""
    print("🔍 Testing single prediction endpoint...")
    
    # First get the CSRF token from the main page
    try:
        session = requests.Session()
        main_page = session.get(f"{base_url}/")
        if main_page.status_code != 200:
            print(f"❌ Could not access main page: {main_page.status_code}")
            return False
        
        # Extract CSRF token from the page
        csrf_token = None
        for line in main_page.text.split('\n'):
            if 'csrf_token' in line and 'value=' in line:
                # Extract token from HTML
                start = line.find('value="') + 7
                end = line.find('"', start)
                csrf_token = line[start:end]
                break
        
        if not csrf_token:
            print("❌ Could not find CSRF token")
            return False
        
        # Sample data for testing
        test_data = {
            'csrf_token': csrf_token,
            'pct_tl_open_L6M': 0.1,
            'pct_tl_closed_L6M': 0.2,
            'Tot_TL_closed_L12M': 1,
            'pct_tl_closed_L12M': 0.3,
            'Tot_Missed_Pmnt': 0,
            'CC_TL': 0,
            'Home_TL': 0,
            'PL_TL': 4,
            'Secured_TL': 1,
            'Unsecured_TL': 4,
            'Other_TL': 0,
            'Age_Oldest_TL': 72,
            'Age_Newest_TL': 18,
            'time_since_recent_payment': 549,
            'max_recent_level_of_deliq': 29,
            'num_deliq_6_12mts': 0,
            'num_times_60p_dpd': 0,
            'num_std_12mts': 11,
            'num_sub': 0,
            'num_sub_6mts': 0,
            'num_sub_12mts': 0,
            'num_dbt': 0,
            'num_dbt_12mts': 0,
            'num_lss': 0,
            'recent_level_of_deliq': 29,
            'CC_enq_L12m': 0,
            'PL_enq_L12m': 0,
            'time_since_recent_enq': 566,
            'enq_L3m': 0,
            'NETMONTHLYINCOME': 51000,
            'Time_With_Curr_Empr': 114,
            'CC_Flag': 0,
            'PL_Flag': 1,
            'pct_PL_enq_L6m_of_ever': 0.0,
            'pct_CC_enq_L6m_of_ever': 0.0,
            'HL_Flag': 1,
            'GL_Flag': 0,
            'EDUCATION': 2,
            'MARITALSTATUS_Married': 1,
            'MARITALSTATUS_Single': 0,
            'GENDER_F': 0,
            'GENDER_M': 1,
            'last_prod_enq2_AL': 0,
            'last_prod_enq2_CC': 0,
            'last_prod_enq2_ConsumerLoan': 1,
            'last_prod_enq2_HL': 0,
            'last_prod_enq2_PL': 0,
            'last_prod_enq2_others': 0,
            'first_prod_enq2_AL': 0,
            'first_prod_enq2_CC': 0,
            'first_prod_enq2_ConsumerLoan': 0,
            'first_prod_enq2_HL': 0,
            'first_prod_enq2_PL': 1,
            'first_prod_enq2_others': 0
        }
        
        response = session.post(f"{base_url}/predict", data=test_data)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Single prediction successful: {data['prediction']}")
                if data.get('probabilities'):
                    print(f"   Probabilities: {data['probabilities']}")
                return True
            else:
                print(f"❌ Single prediction failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
    except Exception as e:
        print(f"❌ Single prediction error: {e}")
        return False

def test_batch_prediction(base_url):
    """Test batch prediction endpoint"""
    print("🔍 Testing batch prediction endpoint...")
    
    try:
        # First get the CSRF token from the main page
        session = requests.Session()
        main_page = session.get(f"{base_url}/")
        if main_page.status_code != 200:
            print(f"❌ Could not access main page: {main_page.status_code}")
            return False
        
        # Extract CSRF token from the page
        csrf_token = None
        for line in main_page.text.split('\n'):
            if 'csrf_token' in line and 'value=' in line:
                # Extract token from HTML
                start = line.find('value="') + 7
                end = line.find('"', start)
                csrf_token = line[start:end]
                break
        
        if not csrf_token:
            print("❌ Could not find CSRF token")
            return False
        
        # Read sample CSV file
        with open('sample_batch_data.csv', 'r') as f:
            files = {'csv_file': f}
            data = {'csrf_token': csrf_token}
            response = session.post(f"{base_url}/predict_batch", files=files, data=data)
        
        if response.status_code == 200:
            print("✅ Batch prediction successful")
            print(f"   Response size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Credit Risk Prediction System - Test Suite")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Wait a moment for server to start
    print("⏳ Waiting for server to start...")
    print("💡 Make sure to run: source credit_risk_env/bin/activate && python app.py")
    time.sleep(2)
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_health_endpoint(base_url)),
        ("Single Prediction", lambda: test_single_prediction(base_url)),
        ("Batch Prediction", lambda: test_batch_prediction(base_url))
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The application is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the application.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
