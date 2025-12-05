#!/usr/bin/env python3
"""
Setup script to create all necessary temp directories for MedRAX tools.
Run this after cloning the repository or to ensure all directories exist.
"""

from pathlib import Path
import sys

def setup_temp_directories():
    """Create all temp directories used by MedRAX tools."""
    
    # Define all temp directories used by tools
    temp_dirs = [
        "temp",  # General temp directory
        "temp/segmentation",  # ChestXRaySegmentationTool
        "temp/medsam2",  # MedSAM2Tool
        "temp/grounding",  # XRayPhraseGroundingTool
        "temp/xray_generation",  # ChestXRayGeneratorTool
        "temp/dicom",  # DicomProcessorTool
        "temp/test_uploads",  # For testing
        "temp/visualizations",  # For various visualizations
    ]
    
    # Also ensure backend temp directory exists
    backend_temp_dirs = [
        "web_platform/backend/temp",
        "web_platform/backend/temp/test_uploads",
    ]
    
    all_dirs = temp_dirs + backend_temp_dirs
    
    print("=" * 60)
    print("SETTING UP TEMP DIRECTORIES FOR MEDRAX")
    print("=" * 60)
    
    created_count = 0
    existed_count = 0
    
    for dir_path in all_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ Already exists: {dir_path}")
            existed_count += 1
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_path}")
            created_count += 1
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total directories: {len(all_dirs)}")
    print(f"Already existed: {existed_count}")
    print(f"Newly created: {created_count}")
    
    # Create .gitignore in temp directories to exclude generated files but keep dirs
    gitignore_content = """# Ignore all files in temp directories
*
# But keep the directory structure
!.gitignore
!*/
"""
    
    for dir_path in ["temp", "web_platform/backend/temp"]:
        gitignore_path = Path(dir_path) / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(gitignore_content)
            print(f"\n✅ Created .gitignore in {dir_path}")
    
    print("\n✨ All temp directories are ready!")
    print("\nNOTE: All temporary files are now stored locally in your project:")
    print("  - Tool outputs: ./temp/[tool_name]/")
    print("  - Backend temp: ./web_platform/backend/temp/")
    print("\nNo more /tmp usage! All files stay within your project. 🎉")
    
    return True

if __name__ == "__main__":
    try:
        success = setup_temp_directories()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


