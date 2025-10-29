#!/usr/bin/env python3
"""
Validation script to ensure all tool schemas are Gemini API compatible.

Run this before committing new tools to catch schema issues early.

SCHEMA STANDARD FOR LIST PARAMETERS:
====================================

When you have a List[] field in your tool input schema, you MUST add json_schema_extra:

    from pydantic import BaseModel, Field
    from typing import List, Optional
    
    class MyToolInput(BaseModel):
        # For List[str] - MUST include json_schema_extra
        items: List[str] = Field(
            ...,
            description="List of items to process",
            json_schema_extra={
                "type": "array",
                "items": {"type": "string"}
            }
        )
        
        # For List[int] - MUST include json_schema_extra
        numbers: List[int] = Field(
            ...,
            description="List of numbers",
            json_schema_extra={
                "type": "array",
                "items": {"type": "integer"}
            }
        )
        
        # For single values - NO json_schema_extra needed
        single_path: str = Field(..., description="Single file path")

WHY: Gemini API requires array parameters to have a top-level 'items' field.
     Pydantic v2 only generates 'items' inside 'anyOf' for Optional[List[]].
     Using json_schema_extra adds the required top-level 'items' field.

FIXED TOOLS: ChestXRaySegmentationTool, CheXagentXRayVQATool, MedSAM2Tool, MedGemmaVQATool
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Any

def validate_schema(schema: Dict[str, Any], tool_name: str, input_class_name: str) -> List[str]:
    """
    Validate that a schema is Gemini API compatible.
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    props = schema.get('properties', {})
    
    for field_name, field_schema in props.items():
        # Check if field is an array
        is_array = False
        
        # Check direct type
        if field_schema.get('type') == 'array':
            is_array = True
        
        # Check anyOf for array type
        if 'anyOf' in field_schema:
            for option in field_schema['anyOf']:
                if option.get('type') == 'array':
                    is_array = True
                    break
        
        if is_array:
            # MUST have top-level 'items' field for Gemini compatibility
            if 'items' not in field_schema:
                errors.append(
                    f"  ❌ {tool_name}.{input_class_name}.{field_name}: "
                    f"Array field missing top-level 'items' (will fail with Gemini API)\n"
                    f"     Add: json_schema_extra={{'type': 'array', 'items': {{'type': '...'}}}}"
                )
            else:
                # Verify items has type
                items = field_schema.get('items', {})
                if 'type' not in items:
                    errors.append(
                        f"  ⚠️  {tool_name}.{input_class_name}.{field_name}: "
                        f"'items' field missing 'type'"
                    )
    
    return errors

def find_tool_input_schemas(tools_dir: Path) -> List[tuple]:
    """
    Find all tool input schema classes.
    
    Returns list of (module_path, class_name, class_obj) tuples.
    """
    schemas = []
    
    # Walk through tools directory
    for root, dirs, files in os.walk(tools_dir):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('__')]
        
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                filepath = Path(root) / file
                
                # Convert to module path
                rel_path = filepath.relative_to(tools_dir.parent)
                module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
                
                try:
                    # Import module
                    module = importlib.import_module(module_path)
                    
                    # Find input schema classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if 'Input' in name and hasattr(obj, 'model_json_schema'):
                            schemas.append((module_path, name, obj))
                
                except Exception as e:
                    # Skip files that can't be imported
                    pass
    
    return schemas

def main():
    """Main validation function."""
    print("="*80)
    print("  MedRAX Tool Schema Validation")
    print("="*80)
    print()
    
    # Find tools directory
    script_dir = Path(__file__).parent
    tools_dir = script_dir
    
    print(f"Scanning: {tools_dir}")
    print()
    
    # Find all input schemas
    schemas = find_tool_input_schemas(tools_dir)
    
    if not schemas:
        print("⚠️  No tool input schemas found!")
        return 1
    
    print(f"Found {len(schemas)} tool input schemas\n")
    
    # Validate each schema
    all_errors = []
    valid_count = 0
    
    for module_path, class_name, class_obj in schemas:
        try:
            schema = class_obj.model_json_schema()
            errors = validate_schema(schema, module_path, class_name)
            
            if errors:
                all_errors.extend(errors)
                print(f"❌ {module_path}.{class_name}")
                for error in errors:
                    print(error)
            else:
                print(f"✅ {module_path}.{class_name}")
                valid_count += 1
        
        except Exception as e:
            all_errors.append(f"  ❌ {module_path}.{class_name}: Error generating schema - {e}")
            print(f"❌ {module_path}.{class_name}")
            print(f"  Error: {e}")
    
    # Summary
    print()
    print("="*80)
    print("Summary")
    print("="*80)
    print(f"Total schemas: {len(schemas)}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {len(schemas) - valid_count}")
    print()
    
    if all_errors:
        print("❌ VALIDATION FAILED")
        print()
        print("Fix the errors above before committing.")
        print("See medrax/tools/TOOL_SCHEMA_STANDARD.md for guidance.")
        return 1
    else:
        print("✅ ALL SCHEMAS ARE GEMINI API COMPATIBLE!")
        return 0

if __name__ == "__main__":
    sys.exit(main())

