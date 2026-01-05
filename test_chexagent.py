#!/usr/bin/env python3
"""Test CheXagent tool directly."""

import sys
import os
sys.path.insert(0, '/home/jma/Documents/Alankrit/MedRAX2')

# Import directly to avoid __init__ dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "xray_vqa",
    "/home/jma/Documents/Alankrit/MedRAX2/medrax/tools/vqa/xray_vqa.py"
)
xray_vqa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xray_vqa)
CheXagentXRayVQATool = xray_vqa.CheXagentXRayVQATool
from pathlib import Path

# Create tool instance
print("Creating CheXagent tool...")
tool = CheXagentXRayVQATool()

# Test paths
test_path = "/home/jma/Documents/Alankrit/MedRAX2/web_platform/backend/temp/test_uploads/pneumonia3.jpg"
print(f"Test path: {test_path}")
print(f"Exists: {Path(test_path).exists()}")

# Test 1: Call with list (correct)
print("\nTest 1: Calling with list of paths...")
try:
    result = tool._run(
        image_paths=[test_path],
        prompt="What abnormalities are visible?",
        max_new_tokens=512
    )
    print(f"Success! Result: {result[0]}")
    print(f"Metadata: {result[1]}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Call with string (incorrect, but should be handled by defensive code)
print("\nTest 2: Calling with string path (testing defensive code)...")
try:
    result = tool._run(
        image_paths=test_path,  # Passing string instead of list
        prompt="What abnormalities are visible?",
        max_new_tokens=512
    )
    print(f"Success! Result: {result[0]}")
    print(f"Metadata: {result[1]}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Check what happens with relative path
print("\nTest 3: Calling with relative path...")
try:
    result = tool._run(
        image_paths=["temp/test_uploads/pneumonia3.jpg"],
        prompt="What abnormalities are visible?",
        max_new_tokens=512
    )
    print(f"Success! Result: {result[0]}")
    print(f"Metadata: {result[1]}")
except Exception as e:
    print(f"Error: {e}")
