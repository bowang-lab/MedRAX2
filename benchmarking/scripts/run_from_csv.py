import argparse
import csv
import json
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _clean_field(row: Dict[str, str], key: str) -> str:
    """Return a stripped string value for a CSV field, handling None safely."""
    return (row.get(key, "") or "").strip()


REQUIRED_COLUMNS = [
    "benchmark",
    "provider",
    "model",
    "system-prompt",
    "max_questions",
    "tools",
    "score",
    "run_id",
]


def load_rows(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        return list(reader), reader.fieldnames

def write_rows(csv_path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_selected_rows(
    csv_path: Path,
    updated_rows_by_idx: Dict[int, Dict[str, str]],
) -> None:
    """Merge selected row updates into CSV atomically under a file lock."""
    if not updated_rows_by_idx:
        return

    try:
        import fcntl
    except ImportError:  # pragma: no cover
        fcntl = None

    with csv_path.open("r+", newline="", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise ValueError(f"CSV '{csv_path}' has no header row.")
            current_rows = list(reader)
            max_idx = max(updated_rows_by_idx)
            if max_idx >= len(current_rows):
                raise ValueError(
                    f"CSV row count changed while running; max selected row index {max_idx + 1} "
                    f"but file only has {len(current_rows)} data rows."
                )

            for idx, row in updated_rows_by_idx.items():
                current_rows[idx] = row

            f.seek(0)
            f.truncate()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(current_rows)
            f.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def list_run_dirs(output_dir: Path) -> Set[Path]:
    if not output_dir.exists():
        return set()
    return {p for p in output_dir.iterdir() if p.is_dir()}


def pick_run_dir(before: Set[Path], after: Set[Path]) -> Optional[Path]:
    new_dirs = after - before
    if new_dirs:
        return sorted(new_dirs, key=lambda p: p.stat().st_mtime)[-1]
    if after:
        return sorted(after, key=lambda p: p.stat().st_mtime)[-1]
    return None


def parse_tools_field(tools_raw: str) -> Optional[List[str]]:
    if tools_raw is None:
        return None
    tools = [t.strip() for t in tools_raw.split(",") if t.strip()]
    return tools or None


def parse_row_selector(rows_raw: str, total_rows: int) -> Set[int]:
    """Parse 1-based row selector like '1,3,5-7' into 0-based row indices."""
    selected: Set[int] = set()
    for token in rows_raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            try:
                start = int(start_raw.strip())
                end = int(end_raw.strip())
            except ValueError as e:
                raise ValueError(f"Invalid row range '{token}'. Use integers like 5-8.") from e
            if start > end:
                raise ValueError(f"Invalid row range '{token}': start must be <= end.")
            row_numbers = range(start, end + 1)
        else:
            try:
                row_num = int(token)
            except ValueError as e:
                raise ValueError(f"Invalid row value '{token}'. Use integers like 5.") from e
            row_numbers = [row_num]

        for row_num in row_numbers:
            if row_num < 1 or row_num > total_rows:
                raise ValueError(
                    f"Row {row_num} is out of range for CSV data rows (1-{total_rows})."
                )
            selected.add(row_num - 1)

    if not selected:
        raise ValueError("Row selector did not include any valid rows.")
    return selected


def build_overrides(row: Dict[str, str]) -> List[str]:
    overrides = []

    benchmark = _clean_field(row, "benchmark")
    provider = _clean_field(row, "provider")
    model = _clean_field(row, "model")
    system_prompt = _clean_field(row, "system-prompt")
    max_questions = _clean_field(row, "max_questions")
    subset = _clean_field(row, "subset")
    temperature = _clean_field(row, "temperature")
    
    overrides.append(f"benchmark={benchmark}")
    overrides.append(f"provider={provider}")
    overrides.append(f"provider.model={model}")
    overrides.append(f"provider.system_prompt={system_prompt}")

    # Allow per-row override of benchmark subset file
    if subset:
        overrides.append(f"benchmark.subset_file={subset}")

    # Allow per-row override of benchmark max questions
    if max_questions:
        overrides.append(f"benchmark.max_questions={max_questions}")

    # Allow per-row override of provider temperature
    if temperature:
        overrides.append(f"provider.temperature={temperature}")

    if provider == "medrax":
        tools_list = parse_tools_field(row.get("tools", ""))
        if tools_list:
            tools_override = ",".join(tools_list)
            overrides.append(f"provider.tools=[{tools_override}]")
        else:
            overrides.append("provider.tools=[]")

    return overrides


def _get_override_value(overrides: List[str], key: str) -> Optional[str]:
    """Return the value for an override key from 'key=value' style overrides."""
    prefix = f"{key}="
    for item in overrides:
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _build_unique_run_id(overrides: List[str]) -> str:
    """Build a unique run_id so parallel workers never share output/log paths."""
    benchmark = _get_override_value(overrides, "benchmark") or "benchmark"
    provider = _get_override_value(overrides, "provider") or "provider"
    model = _get_override_value(overrides, "provider.model") or "model"
    max_questions = _get_override_value(overrides, "benchmark.max_questions") or "None"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique = uuid.uuid4().hex[:8]
    return f"{benchmark}_{provider}_{model}_{max_questions}_{timestamp}_{unique}"


def read_accuracy_from_run(run_dir: Path) -> Optional[float]:
    final_dir = run_dir / "final_results"
    if not final_dir.exists():
        return None
    summaries = sorted(final_dir.glob("*_summary.json"))
    if not summaries:
        return None
    summary_path = summaries[-1]
    try:
        with summary_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", {}).get("accuracy")
    except Exception:
        return None


def run_benchmark(
    overrides: List[str],
    repo_root: Path,
    output_dir: Path,
    timeout: Optional[int] = None,
) -> Dict[str, Optional[object]]:
    run_id = _build_unique_run_id(overrides)
    run_dir = output_dir / run_id
    cmd = ["python", "-m", "benchmarking.cli"] + overrides + [f"hydra.run.dir={run_dir}"]
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            timeout=timeout,
            capture_output=True,
        )
    except subprocess.TimeoutExpired as e:
            return {
                "accuracy": None,
                "stdout": e.stdout,
                "stderr": f"TimeoutExpired: {e}",
                "returncode": -1,
                "run_id": run_id,
            }

    if result.returncode != 0:
        return {
            "accuracy": None,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "run_id": run_id,
        }

    if not run_dir.exists():
        # Fallback if an alternate hydra.run.dir is injected elsewhere.
        discovered_dirs = list_run_dirs(output_dir)
        run_dir = sorted(discovered_dirs, key=lambda p: p.stat().st_mtime)[-1] if discovered_dirs else None
        run_id = run_dir.name if run_dir is not None else run_id
    if run_dir is None or not run_dir.exists():
        return {
            "accuracy": None,
            "stdout": result.stdout,
            "stderr": result.stderr + "\nCould not locate run directory after execution.\n",
            "returncode": result.returncode,
            "run_id": run_id,
        }

    accuracy = read_accuracy_from_run(run_dir)
    return {
        "accuracy": accuracy,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "run_id": run_id,
    }


def process_csv(
    csv_path: Path,
    timeout: Optional[int],
    jobs: int,
    row_selector: Optional[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "benchmark_results"

    csv_rows, fieldnames = load_rows(csv_path)
    selected_indices: Optional[Set[int]] = None
    if row_selector is not None:
        selected_indices = parse_row_selector(row_selector, len(csv_rows))
        selected_rows = ", ".join(str(idx + 1) for idx in sorted(selected_indices))
        print(f"Row filter active. Considering only row(s): {selected_rows}", flush=True)

    # Ensure run_id column exists in output
    if "run_id" not in fieldnames:
        if "score" in fieldnames:
            idx = fieldnames.index("score") + 1
            fieldnames = fieldnames[:idx] + ["run_id"] + fieldnames[idx:]
        else:
            fieldnames.append("run_id")

    final_rows: List[Optional[Dict[str, str]]] = [
        row if (selected_indices is not None and idx not in selected_indices) else None
        for idx, row in enumerate(csv_rows)
    ]

    # Build work list
    work_items = []
    for idx, row in enumerate(csv_rows):
        if selected_indices is not None and idx not in selected_indices:
            continue

        # Skip blank/empty rows (all required fields empty)
        if all((row.get(col, "") or "").strip() == "" for col in REQUIRED_COLUMNS):
            if selected_indices is not None:
                final_rows[idx] = row
            continue

        if _clean_field(row, "score"):
            final_rows[idx] = row
            continue
        work_items.append((idx, row))

    # Process in parallel
    import concurrent.futures

    results_by_idx: Dict[int, Dict[str, object]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(jobs, 1)) as executor:
        future_to_idx = {}
        for idx, row in work_items:
            overrides = build_overrides(row)
            print(f"[{idx + 1}/{len(csv_rows)}] Running with overrides: {overrides}", flush=True)
            future = executor.submit(run_benchmark, overrides, repo_root, output_dir, timeout)
            future_to_idx[future] = idx

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"accuracy": None, "stdout": "", "stderr": str(e), "returncode": -1}
            results_by_idx[idx] = result

    # Apply results in submission order
    for idx, row in work_items:
        result = results_by_idx.get(idx, {"accuracy": None, "stdout": "", "stderr": "", "returncode": -1})
        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""
        returncode = result.get("returncode")
        accuracy = result.get("accuracy")
        run_id = result.get("run_id") or ""

        print(f"\n=== Output for row {idx + 1} (exit {returncode}) ===")
        if stdout.strip():
            print(stdout)
        if stderr.strip():
            sys.stderr.write(stderr + "\n")

        if accuracy is not None:
            row["score"] = str(accuracy)
            print(f"  -> accuracy: {accuracy}")
        else:
            print("  -> failed to obtain accuracy; leaving score empty")
        row["run_id"] = run_id
        if run_id:
            print(f"  -> run_id: {run_id}")

        final_rows[idx] = row

    # When running a selected subset, merge only those rows back to avoid clobbering
    # unrelated updates from other parallel jobs.
    if selected_indices is not None:
        merge_selected_rows(
            csv_path,
            {
                idx: final_rows[idx]
                for idx in selected_indices
                if final_rows[idx] is not None
            },
        )
    else:
        # Default behavior: rewrite rows while filtering blank/empty lines.
        write_rows(csv_path, fieldnames, [r for r in final_rows if r is not None])
 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedRAX benchmarks from a CSV and populate scores."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file with benchmark rows")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-benchmark timeout in seconds (default: no timeout).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of benchmarks to run in parallel (default: 1).",
    )
    parser.add_argument(
        "--rows",
        type=str,
        default=None,
        help="1-based row selector (e.g. '3' or '1,4,7-10'). Only selected rows are processed.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    process_csv(args.csv_path, timeout=args.timeout, jobs=args.jobs, row_selector=args.rows)


if __name__ == "__main__":
    main()
