import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


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


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
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
    before_dirs = list_run_dirs(output_dir)

    cmd = ["python", "-m", "benchmarking.cli"] + overrides
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
                "run_id": None,
            }

    if result.returncode != 0:
        return {
            "accuracy": None,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "run_id": None,
        }

    after_dirs = list_run_dirs(output_dir)
    run_dir = pick_run_dir(before_dirs, after_dirs)
    run_id = run_dir.name if run_dir is not None else None
    if run_dir is None:
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


def process_csv(csv_path: Path, timeout: Optional[int], jobs: int) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "benchmark_results"

    rows, fieldnames = load_rows(csv_path)
    # Ensure run_id column exists in output
    if "run_id" not in fieldnames:
        if "score" in fieldnames:
            idx = fieldnames.index("score") + 1
            fieldnames = fieldnames[:idx] + ["run_id"] + fieldnames[idx:]
        else:
            fieldnames.append("run_id")
    updated_rows: List[Dict[str, str]] = []
    skipped_rows: List[tuple] = []

    # Build work list
    work_items = []
    for idx, row in enumerate(rows):
        # Skip blank/empty rows (all required fields empty)
        if all((row.get(col, "") or "").strip() == "" for col in REQUIRED_COLUMNS):
            continue

        if _clean_field(row, "score"):
            skipped_rows.append((idx, row))
            continue
        work_items.append((idx, row))

    # Process in parallel
    import concurrent.futures

    results_by_idx: Dict[int, Dict[str, object]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(jobs, 1)) as executor:
        future_to_idx = {}
        for idx, row in work_items:
            overrides = build_overrides(row)
            print(f"[{idx + 1}/{len(rows)}] Running with overrides: {overrides}", flush=True)
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
    final_rows: List[Optional[Dict[str, str]]] = [None] * len(rows)
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

    # Append rows that already had scores
    for idx, row in skipped_rows:
        final_rows[idx] = row

    # Filter out any None slots (should not happen) while preserving order
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
    return parser.parse_args()

def main():
    args = parse_args()
    process_csv(args.csv_path, timeout=args.timeout, jobs=args.jobs)


if __name__ == "__main__":
    main()
