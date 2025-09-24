"""
SRT File Cleanup Utility
Deletes previous SRT files to prevent conflicts
"""

import os
import glob
import shutil
from pathlib import Path

def delete_srt_files(directory="."):
    """
    Delete all SRT files in the specified directory and subdirectories
    
    Args:
        directory (str): Directory to search for SRT files (default: current directory)
    
    Returns:
        dict: Summary of deletion results
    """
    
    deleted_files = []
    errors = []
    
    # Find all SRT files
    srt_patterns = [
        "*.srt",
        "**/*.srt"  # Recursive search
    ]
    
    for pattern in srt_patterns:
        search_path = os.path.join(directory, pattern)
        srt_files = glob.glob(search_path, recursive=True)
        
        for srt_file in srt_files:
            try:
                if os.path.isfile(srt_file):
                    print(f"🗑️  Deleting SRT file: {srt_file}")
                    os.remove(srt_file)
                    deleted_files.append(srt_file)
                    
            except Exception as e:
                error_msg = f"Failed to delete {srt_file}: {str(e)}"
                print(f"❌ {error_msg}")
                errors.append(error_msg)
    
    # Also check for specific common SRT file names
    common_srt_files = [
        "test1.srt",
        "subtitles.srt", 
        "output.srt",
        "transcription.srt",
        "video_subs.srt"
    ]
    
    for srt_name in common_srt_files:
        srt_path = os.path.join(directory, srt_name)
        if os.path.exists(srt_path):
            try:
                print(f"🗑️  Deleting common SRT file: {srt_path}")
                os.remove(srt_path)
                if srt_path not in deleted_files:
                    deleted_files.append(srt_path)
            except Exception as e:
                error_msg = f"Failed to delete {srt_path}: {str(e)}"
                print(f"❌ {error_msg}")
                if error_msg not in errors:
                    errors.append(error_msg)
    
    return {
        "deleted_count": len(deleted_files),
        "deleted_files": deleted_files,
        "errors": errors,
        "success": len(errors) == 0
    }

def delete_temp_files(directory="."):
    """
    Delete temporary files that might conflict with generation
    
    Args:
        directory (str): Directory to clean
        
    Returns:
        dict: Summary of deletion results
    """
    
    deleted_files = []
    errors = []
    
    # Temporary file patterns to delete
    temp_patterns = [
        "*.tmp",
        "*.temp", 
        "CAM_data.pkl",
        "lips.pkl",
        "audio.mp3",
        "audio.wav",
        "extracted_audio.*"
    ]
    
    for pattern in temp_patterns:
        search_path = os.path.join(directory, pattern)
        temp_files = glob.glob(search_path)
        
        for temp_file in temp_files:
            try:
                if os.path.isfile(temp_file):
                    print(f"🗑️  Deleting temp file: {temp_file}")
                    os.remove(temp_file)
                    deleted_files.append(temp_file)
            except Exception as e:
                error_msg = f"Failed to delete {temp_file}: {str(e)}"
                print(f"❌ {error_msg}")
                errors.append(error_msg)
    
    return {
        "deleted_count": len(deleted_files),
        "deleted_files": deleted_files,
        "errors": errors,
        "success": len(errors) == 0
    }

def clean_comic_workspace(directory="."):
    """
    Complete cleanup of workspace for fresh comic generation
    
    Args:
        directory (str): Directory to clean
        
    Returns:
        dict: Comprehensive cleanup results
    """
    
    print("🧹 Starting comic workspace cleanup...")
    
    # Delete SRT files
    srt_results = delete_srt_files(directory)
    
    # Delete temporary files  
    temp_results = delete_temp_files(directory)
    
    # Optionally clean output directories (be careful!)
    output_dirs_to_clean = [
        "frames/final",
        "output/page_images"
    ]
    
    cleaned_dirs = []
    for output_dir in output_dirs_to_clean:
        output_path = os.path.join(directory, output_dir)
        if os.path.exists(output_path):
            try:
                # Only clean if it contains generated files
                files_in_dir = os.listdir(output_path)
                if len(files_in_dir) > 0:
                    print(f"🗂️  Cleaning output directory: {output_path}")
                    shutil.rmtree(output_path)
                    os.makedirs(output_path, exist_ok=True)
                    cleaned_dirs.append(output_path)
            except Exception as e:
                print(f"⚠️  Could not clean {output_path}: {str(e)}")
    
    total_deleted = srt_results["deleted_count"] + temp_results["deleted_count"]
    total_errors = srt_results["errors"] + temp_results["errors"]
    
    print(f"\n✅ Cleanup completed!")
    print(f"   📄 SRT files deleted: {srt_results['deleted_count']}")
    print(f"   🗂️  Temp files deleted: {temp_results['deleted_count']}")
    print(f"   📁 Directories cleaned: {len(cleaned_dirs)}")
    print(f"   ❌ Errors: {len(total_errors)}")
    
    if total_errors:
        print("\n⚠️  Errors encountered:")
        for error in total_errors:
            print(f"   • {error}")
    
    return {
        "srt_cleanup": srt_results,
        "temp_cleanup": temp_results,
        "cleaned_directories": cleaned_dirs,
        "total_deleted": total_deleted,
        "total_errors": len(total_errors),
        "success": len(total_errors) == 0
    }

if __name__ == "__main__":
    # Run cleanup when script is executed directly
    import sys
    
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print(f"🧹 Running SRT cleanup in: {os.path.abspath(directory)}")
    
    # Choose cleanup type based on arguments
    if "--full" in sys.argv:
        results = clean_comic_workspace(directory)
    elif "--temp-only" in sys.argv:
        results = delete_temp_files(directory)
    else:
        results = delete_srt_files(directory)
    
    # Exit with appropriate code
    exit_code = 0 if results.get("success", True) else 1
    sys.exit(exit_code)