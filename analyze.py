import json
import argparse
import sys
from collections import defaultdict
from pathlib import Path

def process_single_file(json_file_path):
    """
    Processes a single JSON results file and returns its accuracy counts.

    Args:
        json_file_path (Path): Path to the ...results.json file.

    Returns:
        defaultdict: A dictionary with the aggregated counts for this file.
                     Returns None if the file cannot be processed.
    """
    
    # These counts are *only* for the file being processed
    counts = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct": 0}))
    keys_to_track = ["reasoning_type", "category", "class", "subcategory"]

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"  - WARNING: Could not decode JSON from '{json_file_path}'. Skipping.")
        return None
    except Exception as e:
        print(f"  - ERROR: Unexpected error loading '{json_file_path}': {e}. Skipping.")
        return None

    # Iterate through each record in the JSON array
    for record in data:
        try:
            is_correct = record.get("is_correct", False)
            metadata = record["metadata"]["data_point_metadata"]

            for key in keys_to_track:
                value = metadata.get(key)
                
                if value is not None:
                    counts[key][value]["total"] += 1
                    if is_correct:
                        counts[key][value]["correct"] += 1
                        
        except KeyError as e:
            print(f"  - WARNING: Record {record.get('data_point_id')} is missing expected key: {e}. Skipping record.")
        except TypeError:
            print(f"  - WARNING: Record {record.get('data_point_id')} has unexpected data structure. Skipping record.")
            
    return counts

def generate_report_dict(counts):
    """
    Converts a counts dictionary into the final, formatted report dictionary.

    Args:
        counts (defaultdict): The aggregated counts from process_single_file.

    Returns:
        dict: A dictionary formatted with percentages and absolute numbers.
    """
    accuracy_report = defaultdict(dict)
    
    for key, values in counts.items():
        # Sort by the sub-category name (e.g., "Negation Assessment")
        sorted_values = sorted(values.items(), key=lambda item: item[0])
        
        for value, tally in sorted_values:
            total = tally["total"]
            correct = tally["correct"]
            
            if total > 0:
                accuracy = (correct / total) * 100
            else:
                accuracy = 0.0
            
            # Store the full results in our report dictionary
            accuracy_report[key][value] = {
                "accuracy_percent": round(accuracy, 2),
                "correct": correct,
                "total": total
            }
    return accuracy_report

def main():
    """
    Main function to find, process, and save individual reports.
    """
    
    parser = argparse.ArgumentParser(
        description="Finds and processes individual benchmarking runs, saving "
                    "a separate accuracy report for each run."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The top-level directory to search within (e.g., 'my_experiments')."
    )
    args = parser.parse_args()

    top_dir = Path(args.directory)
    if not top_dir.is_dir():
        print(f"Error: Path '{args.directory}' is not a valid directory.")
        sys.exit(1)

    # Glob pattern to find all target files
    search_pattern = '*/final_results/*results.json'
    json_files_to_process = list(top_dir.glob(search_pattern))

    if not json_files_to_process:
        print(f"No files matching the pattern '{search_pattern}' were found in '{top_dir}'.")
        sys.exit(0)

    print(f"Found {len(json_files_to_process)} result file(s) to process individually.")

    # --- Loop and process each file ---
    for file_path in json_files_to_process:
        # Use relative path for cleaner logging
        print(f"\n--- Processing: {file_path.relative_to(top_dir.parent)} ---")
        
        # 1. Get counts for this file
        counts = process_single_file(file_path)
        
        if counts is None or not counts:
            print("  - No data processed. Skipping report generation.")
            continue
            
        # 2. Generate the report dictionary
        report = generate_report_dict(counts)
        
        # 3. Determine the output path and save the file
        #    The output is saved in the *same directory* as the input file
        output_filename = file_path.parent / "accuracy_report.json"
        
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, sort_keys=True)
            print(f"  > Successfully saved report to: {output_filename.relative_to(top_dir.parent)}")
        except Exception as e:
            print(f"  > ERROR: Could not save report to '{output_filename}': {e}")
            
    print("\nAll processing complete.")

if __name__ == "__main__":
    main()