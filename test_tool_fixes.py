#!/usr/bin/env python3
"""
Test script to verify the tool fixes are working.
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000/api/test"
TEST_IMAGE = "temp/test_uploads/normal6.jpg"

def test_tool(name, endpoint, payload):
    """Test a single tool."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Endpoint: {endpoint}")
    print("-" * 60)
    
    url = f"{BASE_URL}/{endpoint}"
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()
        
        if data.get("success"):
            result = data.get("result", {})
            metadata = data.get("metadata", {})
            
            # Check if there's an error in the result
            if "error" in result:
                print(f"⚠️  Tool returned error: {result['error']}")
                print(f"   Status: {metadata.get('analysis_status', 'unknown')}")
                return False
            else:
                print(f"✅ SUCCESS!")
                # Show relevant output based on tool type
                if "response" in result:
                    response_preview = result["response"][:200] + "..." if len(result.get("response", "")) > 200 else result.get("response", "")
                    print(f"   Response: {response_preview}")
                elif "predictions" in result:
                    print(f"   Predictions found: {len(result.get('predictions', []))}")
                elif "segmentation_image_path" in result:
                    print(f"   Segmentation saved: {result['segmentation_image_path']}")
                else:
                    print(f"   Result keys: {list(result.keys())}")
                return True
        else:
            print(f"❌ Request failed: {data.get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏱️  Timeout (>30s)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Test the fixed tools."""
    print("=" * 60)
    print("TESTING TOOL FIXES")
    print("=" * 60)
    print(f"Test image: {TEST_IMAGE}")
    
    # Define the problematic tools we fixed
    tests = [
        {
            "name": "CheXagent VQA",
            "endpoint": "chexagent",
            "payload": {
                "image_path": TEST_IMAGE,
                "question": "What abnormalities are visible?"
            }
        },
        {
            "name": "MedGemma VQA",
            "endpoint": "medgemma",
            "payload": {
                "image_path": TEST_IMAGE,
                "question": "What abnormalities are visible?"
            }
        },
        {
            "name": "Phrase Grounding",
            "endpoint": "phrase_grounding",
            "payload": {
                "image_path": TEST_IMAGE,
                "phrase": "enlarged heart"
            }
        },
        {
            "name": "MedSAM2 (still working)",
            "endpoint": "medsam2",
            "payload": {
                "image_path": TEST_IMAGE,
                "prompt_type": "box",
                "prompt_coords": [100, 100, 200, 200]
            }
        }
    ]
    
    results = []
    for test in tests:
        success = test_tool(test["name"], test["endpoint"], test["payload"])
        results.append((test["name"], success))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ FIXED" if success else "❌ STILL BROKEN"
        print(f"{status}: {name}")
    
    successful = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nTotal: {successful}/{total} working")
    
    if successful < total:
        print("\n⚠️  Some tools still have issues. You may need to:")
        print("  1. Restart the backend to load the fixes")
        print("  2. Check if the models are properly loaded")
        print("  3. Review the error messages above")
    else:
        print("\n🎉 All tools are working!")
    
    return successful == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

