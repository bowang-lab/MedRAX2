#!/usr/bin/env python3
"""
Quick test script for MedSAM2 segmentation tool.
Tests different prompt types and shows results.
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000/api/test/medsam2"
TEST_IMAGE = "temp/test_uploads/pneumonia3.jpg"

def test_medsam2(prompt_type, coords, description):
    """Test MedSAM2 with given prompt."""
    payload = {
        "image_path": TEST_IMAGE,
        "prompt_type": prompt_type,
        "prompt_coords": coords
    }
    
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Prompt Type: {prompt_type}")
    print(f"Coordinates: {coords if coords else 'None (auto)'}")
    print("-" * 60)
    
    try:
        response = requests.post(BASE_URL, json=payload, timeout=30)
        data = response.json()
        
        if data.get("success"):
            result = data["result"]
            metadata = data.get("metadata", {})
            
            print("✅ SUCCESS!")
            print(f"\n📊 Results:")
            print(f"  • Best confidence: {result['best_mask_score']:.1%}")
            print(f"  • All scores: {[f'{s:.1%}' for s in result['confidence_scores']]}")
            print(f"  • Segmented areas: {result['mask_summary']['segmented_area_pixels']} pixels")
            
            # Calculate percentage of image covered
            img_shape = metadata.get("image_shape", [856, 1144, 3])
            total_pixels = img_shape[0] * img_shape[1]
            percentages = [f"{(area/total_pixels)*100:.1f}%" 
                          for area in result['mask_summary']['segmented_area_pixels']]
            print(f"  • Area coverage: {percentages}")
            
            print(f"\n📁 Output file:")
            # Show both relative and absolute path
            rel_path = result['segmentation_image_path']
            if rel_path.startswith('temp/'):
                abs_path = Path("web_platform/backend") / rel_path
                print(f"  • Relative: {rel_path}")
                print(f"  • Absolute: {abs_path.absolute()}")
            else:
                print(f"  • Path: {rel_path}")
            
            return True
        else:
            print(f"❌ FAILED: {data.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run comprehensive MedSAM2 tests."""
    print("=" * 60)
    print("MEDSAM2 TESTING SUITE")
    print("=" * 60)
    print(f"Testing with image: {TEST_IMAGE}")
    print("Image size: 1144 x 856 pixels")
    
    tests = [
        # Box prompts - most reliable
        ("box", [100, 100, 200, 200], "Small box (100x100) - testing small region"),
        ("box", [150, 200, 500, 650], "Right lung area (appears left in image)"),
        ("box", [644, 200, 994, 650], "Left lung area (appears right in image)"),
        ("box", [400, 400, 744, 700], "Heart/mediastinum region"),
        
        # Point prompts - for specific locations
        ("point", [325, 425], "Point on right lung center"),
        ("point", [819, 425], "Point on left lung center"),
        ("point", [572, 550], "Point on heart center"),
        
        # Auto segmentation - experimental
        ("auto", [], "Automatic segmentation (no prompts)"),
    ]
    
    results = []
    for prompt_type, coords, description in tests:
        success = test_medsam2(prompt_type, coords, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Tests passed: {successful}/{total}")
    
    if successful < total:
        print("\nFailed tests:")
        for desc, success in results:
            if not success:
                print(f"  • {desc}")
    
    print("\n💡 Tips:")
    print("  • Box prompts are most reliable")
    print("  • MedSAM2 generates 3 masks - check all of them")
    print("  • Higher confidence (>70%) usually means better segmentation")
    print("  • Files are saved in: web_platform/backend/temp/medsam2/")
    
    return successful == total

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)


