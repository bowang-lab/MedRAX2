#!/usr/bin/env python3
"""
Test script to verify all tools have consistent interfaces and work properly.
"""

import json
import requests
from pathlib import Path
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000/api/test"
TEST_IMAGE = "temp/test_uploads/pneumonia3.jpg"

def test_tool(endpoint: str, payload: Dict[str, Any]) -> Dict:
    """Test a single tool endpoint."""
    url = f"{BASE_URL}/{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=30)
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "data": response.json() if response.status_code == 200 else response.text
        }
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Test all tools for consistency."""
    
    print("=" * 60)
    print("TESTING TOOL CONSISTENCY")
    print("=" * 60)
    
    # Define test cases for each tool
    test_cases = [
        {
            "name": "TorchXRayVision Classifier",
            "endpoint": "torchxrayvision",
            "payload": {"image_path": TEST_IMAGE}
        },
        {
            "name": "ArcPlus Classifier",
            "endpoint": "arcplus",
            "payload": {"image_path": TEST_IMAGE}
        },
        {
            "name": "Chest Segmentation",
            "endpoint": "chest_segmentation",
            "payload": {
                "image_path": TEST_IMAGE,
                "organs": ["Left Lung", "Right Lung"],
                "threshold": 0.3
            }
        },
        {
            "name": "MedSAM2 Segmentation",
            "endpoint": "medsam2",
            "payload": {
                "image_path": TEST_IMAGE,
                "prompt_type": "auto",
                "prompt_coords": []
            }
        },
        {
            "name": "CheXagent VQA",
            "endpoint": "chexagent",
            "payload": {
                "image_path": TEST_IMAGE,  # Now single path!
                "question": "What abnormalities are visible?"
            }
        },
        {
            "name": "MedGemma VQA",
            "endpoint": "medgemma",
            "payload": {
                "image_path": TEST_IMAGE,  # Now single path!
                "question": "What abnormalities are visible?"
            }
        },
        {
            "name": "Report Generator",
            "endpoint": "report_generator",
            "payload": {"image_path": TEST_IMAGE}
        },
        {
            "name": "Phrase Grounding",
            "endpoint": "phrase_grounding",
            "payload": {
                "image_path": TEST_IMAGE,
                "phrase": "opacity"
            }
        }
    ]
    
    # Test each tool
    results = []
    for test_case in test_cases:
        print(f"\nTesting {test_case['name']}...")
        print(f"  Endpoint: {test_case['endpoint']}")
        print(f"  Payload: {json.dumps(test_case['payload'], indent=2)}")
        
        result = test_tool(test_case['endpoint'], test_case['payload'])
        results.append({
            "tool": test_case['name'],
            "success": result.get("success", False),
            "error": result.get("error") if not result.get("success") else None
        })
        
        if result.get("success"):
            data = result.get("data", {})
            if data.get("success"):
                print(f"  ✅ SUCCESS")
                # Check metadata for image_path consistency
                metadata = data.get("metadata", {})
                if "image_path" in metadata:
                    print(f"  ✓ Uses 'image_path' (consistent)")
                elif "image_paths" in metadata:
                    print(f"  ⚠️ Uses 'image_paths' (needs update)")
                else:
                    print(f"  ⚠️ No image path in metadata")
            else:
                print(f"  ❌ FAILED: {data.get('error', 'Unknown error')}")
        else:
            print(f"  ❌ REQUEST FAILED: {result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"Total tools tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed tools:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['tool']}: {result['error']}")
    
    print("\n" + "=" * 60)
    print("INTERFACE CONSISTENCY CHECK")
    print("=" * 60)
    
    print("\nAll tools should now use:")
    print("  - image_path: str (single image path)")
    print("  - NOT image_paths: List[str]")
    print("\nThis makes the interface consistent across all tools.")
    
    return successful == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


