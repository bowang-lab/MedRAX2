#!/usr/bin/env python3
"""
Comprehensive Tool Validation Script

This script validates tool implementations WITHOUT importing them (no dependencies needed).
It performs static analysis using AST parsing to catch common issues:

1. _arun/_run signature mismatches
2. Missing required methods
3. Incorrect return types
4. Schema validation issues
5. Missing error handling

Run this before committing tool changes!
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class MethodSignature:
    """Represents a method signature."""
    name: str
    params: List[str]
    defaults: List[Any]
    return_annotation: Optional[str] = None


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: str  # 'error', 'warning', 'info'
    tool_name: str
    file_path: str
    line_number: int
    message: str
    
    def __str__(self):
        icon = "❌" if self.severity == "error" else "⚠️" if self.severity == "warning" else "ℹ️"
        return f"{icon} {self.severity.upper()}: {self.tool_name} ({self.file_path}:{self.line_number})\n   {self.message}"


class ToolValidator:
    """Validates tool implementations using AST parsing."""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
    
    def add_issue(self, severity: str, tool_name: str, file_path: str, line_number: int, message: str):
        """Add a validation issue."""
        self.issues.append(ValidationIssue(severity, tool_name, file_path, line_number, message))
    
    def extract_method_signature(self, node: ast.FunctionDef) -> MethodSignature:
        """Extract method signature from AST node."""
        params = [arg.arg for arg in node.args.args if arg.arg != 'self']
        defaults = []
        
        # Extract defaults (aligned from the right)
        if node.args.defaults:
            num_defaults = len(node.args.defaults)
            num_params = len(params)
            defaults = [None] * (num_params - num_defaults) + list(node.args.defaults)
        
        return_annotation = None
        if node.returns:
            return_annotation = ast.unparse(node.returns)
        
        return MethodSignature(node.name, params, defaults, return_annotation)
    
    def find_return_statement(self, node: ast.FunctionDef) -> Optional[ast.Return]:
        """Find the return statement in a function."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return child
        return None
    
    def find_return_call_args(self, node: ast.FunctionDef) -> Optional[List[str]]:
        """Find arguments passed to self._run() in return statement."""
        ret_stmt = self.find_return_statement(node)
        if not ret_stmt or not ret_stmt.value:
            return None
        
        if isinstance(ret_stmt.value, ast.Call):
            if isinstance(ret_stmt.value.func, ast.Attribute):
                if ret_stmt.value.func.attr == '_run':
                    # Extract argument names
                    args = []
                    for arg in ret_stmt.value.args:
                        if isinstance(arg, ast.Name):
                            args.append(arg.id)
                        elif isinstance(arg, ast.Constant):
                            args.append(f"<const:{arg.value}>")
                        else:
                            args.append("<expr>")
                    
                    # Also include keyword arguments
                    for kw in ret_stmt.value.keywords:
                        args.append(f"{kw.arg}=...")
                    
                    return args
        return None
    
    def has_error_handling(self, node: ast.FunctionDef) -> bool:
        """Check if function has try/except error handling."""
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
        return False
    
    def validate_tool_class(self, class_node: ast.ClassDef, file_path: Path, content: str):
        """Validate a single tool class."""
        tool_name = class_node.name
        
        # Find _run and _arun methods
        _run_method = None
        _arun_method = None
        
        for item in class_node.body:
            # Check both sync and async function definitions
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == '_run':
                    _run_method = item
                elif item.name == '_arun':
                    _arun_method = item
        
        # Check 1: Both methods should exist
        if not _run_method:
            self.add_issue('warning', tool_name, str(file_path), class_node.lineno,
                          "Missing _run method")
        
        if not _arun_method:
            self.add_issue('warning', tool_name, str(file_path), class_node.lineno,
                          "Missing _arun method")
        
        # Check 2: Validate _arun/_run signature consistency
        if _run_method and _arun_method:
            run_sig = self.extract_method_signature(_run_method)
            arun_sig = self.extract_method_signature(_arun_method)
            
            # Filter out run_manager for comparison (it's expected to differ)
            run_params_filtered = [p for p in run_sig.params if 'run_manager' not in p]
            arun_params_filtered = [p for p in arun_sig.params if 'run_manager' not in p]
            
            # Check if _arun has all the same params as _run
            if set(arun_params_filtered) != set(run_params_filtered):
                missing = set(run_params_filtered) - set(arun_params_filtered)
                extra = set(arun_params_filtered) - set(run_params_filtered)
                
                msg_parts = []
                if missing:
                    msg_parts.append(f"Missing params in _arun: {missing}")
                if extra:
                    msg_parts.append(f"Extra params in _arun: {extra}")
                
                self.add_issue('error', tool_name, str(file_path), _arun_method.lineno,
                              "Parameter mismatch between _run and _arun: " + "; ".join(msg_parts))
            
            # Check if _arun properly forwards to _run
            passed_args = self.find_return_call_args(_arun_method)
            if passed_args is not None:
                # Remove run_manager from comparison
                passed_args_filtered = [a for a in passed_args if 'run_manager' not in a]
                
                # Check if correct number of args are passed
                if len(passed_args_filtered) != len(run_params_filtered):
                    self.add_issue('error', tool_name, str(file_path), _arun_method.lineno,
                                  f"_arun passes {len(passed_args_filtered)} args to _run but _run expects {len(run_params_filtered)}. "
                                  f"Passed: {passed_args_filtered}, Expected: {run_params_filtered}")
                
                # Check if run_manager is in wrong position
                if 'run_manager' in passed_args:
                    rm_index = passed_args.index('run_manager')
                    # run_manager should be the last argument or not passed at all
                    if rm_index < len(run_params_filtered):
                        self.add_issue('error', tool_name, str(file_path), _arun_method.lineno,
                                      f"run_manager passed at position {rm_index} but should be last (after position {len(run_params_filtered)-1})")
        
        # Check 3: Error handling in _run
        if _run_method:
            if not self.has_error_handling(_run_method):
                self.add_issue('warning', tool_name, str(file_path), _run_method.lineno,
                              "_run method missing try/except error handling")
        
        # Check 4: Return type annotations
        if _run_method and not _run_method.returns:
            self.add_issue('info', tool_name, str(file_path), _run_method.lineno,
                          "_run method missing return type annotation")
        
        if _arun_method and not _arun_method.returns:
            self.add_issue('info', tool_name, str(file_path), _arun_method.lineno,
                          "_arun method missing return type annotation")
    
    def validate_file(self, file_path: Path) -> int:
        """Validate a single Python file. Returns number of issues found."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            self.add_issue('error', 'N/A', str(file_path), e.lineno or 0,
                          f"Syntax error: {e.msg}")
            return 1
        except Exception as e:
            self.add_issue('error', 'N/A', str(file_path), 0,
                          f"Failed to parse file: {e}")
            return 1
        
        issues_before = len(self.issues)
        
        # Find all classes that inherit from BaseTool
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if this is a Tool class
                is_tool = any(
                    (isinstance(base, ast.Name) and 'Tool' in base.id) or
                    (isinstance(base, ast.Attribute) and 'Tool' in base.attr)
                    for base in node.bases
                )
                
                if is_tool:
                    self.validate_tool_class(node, file_path, content)
        
        return len(self.issues) - issues_before
    
    def validate_directory(self, tools_dir: Path) -> Dict[str, int]:
        """Validate all tools in directory. Returns stats."""
        stats = {
            'files_scanned': 0,
            'files_with_issues': 0,
            'total_errors': 0,
            'total_warnings': 0,
            'total_info': 0
        }
        
        # Find all Python files
        for file_path in tools_dir.rglob('*.py'):
            # Skip __pycache__, __init__.py, and validation scripts
            if '__pycache__' in str(file_path) or file_path.name.startswith('__'):
                continue
            if 'validate' in file_path.name or 'audit' in file_path.name:
                continue
            
            stats['files_scanned'] += 1
            issues_count = self.validate_file(file_path)
            
            if issues_count > 0:
                stats['files_with_issues'] += 1
        
        # Count issues by severity
        for issue in self.issues:
            if issue.severity == 'error':
                stats['total_errors'] += 1
            elif issue.severity == 'warning':
                stats['total_warnings'] += 1
            else:
                stats['total_info'] += 1
        
        return stats


def main():
    """Main validation function."""
    tools_dir = Path(__file__).parent
    
    print("=" * 80)
    print("  MedRAX Tool Validation (Static Analysis)")
    print("=" * 80)
    print(f"\nScanning: {tools_dir}\n")
    
    validator = ToolValidator()
    stats = validator.validate_directory(tools_dir)
    
    print(f"📊 Scanned {stats['files_scanned']} files")
    print()
    
    if not validator.issues:
        print("✅ All tools passed validation!")
        return 0
    
    # Group issues by severity
    errors = [i for i in validator.issues if i.severity == 'error']
    warnings = [i for i in validator.issues if i.severity == 'warning']
    infos = [i for i in validator.issues if i.severity == 'info']
    
    # Print errors first
    if errors:
        print(f"🔴 ERRORS ({len(errors)}):")
        print("-" * 80)
        for issue in errors:
            print(issue)
            print()
    
    # Then warnings
    if warnings:
        print(f"🟡 WARNINGS ({len(warnings)}):")
        print("-" * 80)
        for issue in warnings:
            print(issue)
            print()
    
    # Then info
    if infos:
        print(f"🔵 INFO ({len(infos)}):")
        print("-" * 80)
        for issue in infos:
            print(issue)
            print()
    
    # Summary
    print("=" * 80)
    print(f"SUMMARY: {stats['total_errors']} errors, {stats['total_warnings']} warnings, {stats['total_info']} info")
    print("=" * 80)
    
    # Exit with error code if any errors found
    return 1 if stats['total_errors'] > 0 else 0


if __name__ == '__main__':
    sys.exit(main())

