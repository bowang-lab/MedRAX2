import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm


METADATA_COLUMNS = {
    "dicom_id",
    "subject_id",
    "study_id",
    "ViewPosition",
    "ViewCodeSequence_CodeMeaning",
    "fpath",
    "split",
}
DEFAULT_PHYSIONET_BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/"
PHYSIONET_WGET_USER_AGENT = "Wget/1.20.3 (linux-gnu)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a JSONL manifest for MIMIC benchmarking from a specific CXR-LT CSV file."
        )
    )
    parser.add_argument(
        "--source-csv",
        type=Path,
        required=True,
        help=(
            "Path to one source CSV file (e.g., test_labeled_task2.csv, "
            "test_labeled_task3.csv, train_labeled.csv, or labels.csv)."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Root directory used to resolve CSV fpath values. "
            "Default: sibling mimic-cxr-jpg/2.0.0 folder under the same PhysioNet files directory."
        ),
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="single_positive_only",
        choices=["single_positive_only", "allow_multiple_findings"],
        help="Question/answer mode for generated manifest rows.",
    )
    parser.add_argument(
        "--download-missing-images",
        action="store_true",
        help=(
            "If set, attempt to download missing image files from PhysioNet using "
            "PHYSIONET_USERNAME/PHYSIONET_PASSWORD in .env."
        ),
    )
    parser.add_argument(
        "--physionet-base-url",
        type=str,
        default=DEFAULT_PHYSIONET_BASE_URL,
        help="Base URL used to download missing images.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap on final number of manifest rows after filtering.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used only when max_questions is set.",
    )
    return parser.parse_args()


def infer_default_image_root(source_csv: Path) -> Path:
    # .../files/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/test_labeled_task2.csv
    # -> .../files/mimic-cxr-jpg/2.0.0
    data_dir = source_csv.parent
    if len(data_dir.parents) < 3:
        raise ValueError(
            f"Cannot infer default image root from source CSV path: {source_csv}. "
            "Please provide --image-root explicitly."
        )
    files_root = data_dir.parents[2]
    return files_root / "mimic-cxr-jpg" / "2.0.0"


def read_header(csv_path: Path) -> List[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def compute_label_columns(header: Sequence[str]) -> List[str]:
    return [col for col in header if col not in METADATA_COLUMNS]


def count_data_rows(csv_path: Path) -> int:
    # Exclude the header row so tqdm can render progress with percentage.
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        total_lines = sum(1 for _ in f)
    return max(0, total_lines - 1)


def normalize_split(raw_split: str, source_file: Path) -> str:
    raw = (raw_split or "").strip().lower()
    if raw == "val":
        return "development"
    if raw in {"train", "test", "development"}:
        return raw
    raise ValueError(
        f"Unexpected split value '{raw_split}' in {source_file.name}. "
        "Expected train/val/development/test."
    )


def infer_split_from_filename(source_file: Path) -> str:
    name = source_file.name.lower()
    if "development" in name or "dev" in name or "val" in name:
        return "development"
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    return "unknown"


def parse_binary_label(raw_value: str, file_path: Path, row_number: int, column: str) -> int:
    value = (raw_value or "").strip()
    if value == "0":
        return 0
    if value == "1":
        return 1
    raise ValueError(
        f"Invalid non-binary label value in {file_path} row {row_number}, "
        f"column '{column}': '{raw_value}'. Expected integer 0 or 1."
    )


def normalize_fpath(value: str) -> str:
    return (value or "").strip().lstrip("./").lstrip("/")


def resolve_physionet_auth(download_missing_images: bool) -> Optional[Tuple[str, str]]:
    if not download_missing_images:
        return None

    load_dotenv()
    username = os.getenv("PHYSIONET_USERNAME", "").strip()
    password = os.getenv("PHYSIONET_PASSWORD", "").strip()
    if not username or not password:
        raise ValueError(
            "--download-missing-images was set, but PHYSIONET_USERNAME/PHYSIONET_PASSWORD "
            "were not found in the environment/.env."
        )
    return username, password


def maybe_download_image(
    session: requests.Session,
    image_path: Path,
    fpath: str,
    base_url: str,
    auth: Tuple[str, str],
) -> Tuple[bool, str]:
    normalized_fpath = normalize_fpath(fpath)
    if not normalized_fpath:
        return False, "empty fpath; cannot download image"

    image_url = f"{base_url.rstrip('/')}/{normalized_fpath}"
    tmp_path = image_path.with_suffix(f"{image_path.suffix}.part")

    try:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        with session.get(image_url, auth=auth, stream=True, timeout=(10, 60)) as response:
            if response.status_code != 200:
                return (
                    False,
                    f"HTTP {response.status_code} while downloading {image_url}",
                )
            with tmp_path.open("wb") as out:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
        tmp_path.replace(image_path)
    except requests.RequestException as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False, f"request error while downloading {image_url}: {e}"
    except OSError as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False, f"file write error for {image_path}: {e}"

    if not image_path.exists():
        return False, f"download completed but image still missing: {image_path}"
    return True, ""


def build_question_text(class_to_token: Dict[str, str], label_mode: str) -> str:
    options = [f"{token}. {cls}" for cls, token in class_to_token.items()]
    options_block = "\n".join(options)

    if label_mode == "single_positive_only":
        question = "Which class best describes this chest X-ray?"
        instruction = "Choose exactly one option."
        answer_hint = "Respond with your final answer in the format: \\boxed{12}"
    else:
        question = "Which classes are present in this chest X-ray?"
        instruction = "One or more options may be correct. Select all that apply."
        answer_hint = "Respond with your final answer in the format: \\boxed{3,12}"

    return (
        f"{question}\n"
        f"{instruction}\n\n"
        "Options:\n"
        f"{options_block}\n\n"
        f"{answer_hint}"
    )


def build_manifest_rows(
    source_file: Path,
    image_root: Path,
    label_columns: Sequence[str],
    label_mode: str,
    download_missing_images: bool,
    physionet_base_url: str,
    physionet_auth: Optional[Tuple[str, str]],
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    class_to_token = {label: str(i + 1) for i, label in enumerate(label_columns)}
    question_text = build_question_text(class_to_token, label_mode=label_mode)
    inferred_split = infer_split_from_filename(source_file)
    has_split_column = "split" in read_header(source_file)
    total_rows = count_data_rows(source_file)

    counts = {
        "rows_read": 0,
        "rows_missing_image_initial": 0,
        "download_attempted": 0,
        "download_succeeded": 0,
        "download_failed": 0,
        "rows_skipped_missing_after_download": 0,
        "rows_dropped_label_mode": 0,
        "rows_written": 0,
    }
    split_counts: Dict[str, int] = {
        "train": 0,
        "development": 0,
        "test": 0,
        "unknown": 0,
    }

    rows: List[Dict[str, object]] = []
    with source_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        session = requests.Session() if download_missing_images else None
        if session is not None:
            # PhysioNet accepts wget-style auth requests more consistently than default python-requests UA.
            session.headers.update({"User-Agent": PHYSIONET_WGET_USER_AGENT})
        try:
            for row_idx, row in enumerate(
                tqdm(
                    reader,
                    total=total_rows,
                    desc=f"Processing {source_file.name}",
                    unit="row",
                ),
                start=2,
            ):
                counts["rows_read"] += 1

                if has_split_column:
                    split = normalize_split(row.get("split", ""), source_file=source_file)
                    split_source = "csv"
                else:
                    split = inferred_split
                    split_source = "filename"

                fpath = normalize_fpath(row.get("fpath", ""))
                image_path = image_root / fpath if fpath else image_root
                image_available = bool(fpath) and image_path.exists()

                if not image_available:
                    counts["rows_missing_image_initial"] += 1
                    if download_missing_images:
                        if session is None or physionet_auth is None:
                            raise RuntimeError(
                                "download_missing_images=True but requests session/auth was not initialized."
                            )
                        counts["download_attempted"] += 1
                        downloaded, error = maybe_download_image(
                            session=session,
                            image_path=image_path,
                            fpath=fpath,
                            base_url=physionet_base_url,
                            auth=physionet_auth,
                        )
                        if downloaded:
                            counts["download_succeeded"] += 1
                            image_available = True
                        else:
                            counts["download_failed"] += 1
                            print(f"Download failed for row {row_idx} ({fpath}): {error}")

                if not image_available:
                    counts["rows_skipped_missing_after_download"] += 1
                    continue

                positives: List[str] = []
                for label in label_columns:
                    label_value = parse_binary_label(
                        row.get(label, ""),
                        file_path=source_file,
                        row_number=row_idx,
                        column=label,
                    )
                    if label_value == 1:
                        positives.append(label)

                if label_mode == "single_positive_only":
                    if len(positives) != 1:
                        counts["rows_dropped_label_mode"] += 1
                        continue
                else:
                    if not positives:
                        counts["rows_dropped_label_mode"] += 1
                        continue

                positive_tokens = sorted(
                    (class_to_token[p] for p in positives), key=lambda x: int(x)
                )
                correct_answer = (
                    positive_tokens[0]
                    if label_mode == "single_positive_only"
                    else ",".join(positive_tokens)
                )
                split_counts[split] = split_counts.get(split, 0) + 1

                dicom_id = (row.get("dicom_id") or "").strip()
                subject_id = (row.get("subject_id") or "").strip()
                study_id = (row.get("study_id") or "").strip()
                data_point_id = f"mimic_{subject_id}_{study_id}_{dicom_id}"

                manifest_row = {
                    "id": data_point_id,
                    "text": question_text,
                    "images": [str(image_path)],
                    "correct_answer": correct_answer,
                    "metadata": {
                        "dataset": "mimic_cxr_lt",
                        "split": split,
                        "split_source": split_source,
                        "source_csv": source_file.name,
                        "dicom_id": dicom_id,
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "view_position": (row.get("ViewPosition") or "").strip(),
                        "view_code_meaning": (row.get("ViewCodeSequence_CodeMeaning") or "").strip(),
                        "fpath": fpath,
                        "classes": list(label_columns),
                        "class_to_option": class_to_token,
                        "positive_classes": positives,
                        "positive_options": positive_tokens,
                        "multi_select": label_mode == "allow_multiple_findings",
                        "label_validation": "binary_int_0_1",
                        "validated_label_columns": len(label_columns),
                    },
                }
                rows.append(manifest_row)
                counts["rows_written"] += 1
        finally:
            if session is not None:
                session.close()

    counts["written_train"] = split_counts.get("train", 0)
    counts["written_development"] = split_counts.get("development", 0)
    counts["written_test"] = split_counts.get("test", 0)
    counts["written_unknown"] = split_counts.get("unknown", 0)
    return rows, counts


def maybe_sample_rows(
    rows: List[Dict[str, object]],
    max_questions: Optional[int],
    seed: int,
) -> List[Dict[str, object]]:
    if max_questions is None or len(rows) <= max_questions:
        return rows

    rng = random.Random(seed)
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:max_questions]


def main() -> None:
    args = parse_args()
    source_file = args.source_csv.resolve()
    if not source_file.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_file}")

    image_root = args.image_root.resolve() if args.image_root else infer_default_image_root(source_file)
    output_file = args.output_file.resolve()
    label_columns = compute_label_columns(read_header(source_file))
    if not label_columns:
        raise ValueError("No label columns were detected after excluding metadata columns.")

    physionet_auth = resolve_physionet_auth(args.download_missing_images)
    rows, counts = build_manifest_rows(
        source_file=source_file,
        image_root=image_root,
        label_columns=label_columns,
        label_mode=args.label_mode,
        download_missing_images=args.download_missing_images,
        physionet_base_url=args.physionet_base_url,
        physionet_auth=physionet_auth,
    )
    rows = maybe_sample_rows(rows, max_questions=args.max_questions, seed=args.random_seed)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Source file: {source_file}")
    print(f"Image root: {image_root}")
    print(f"Label columns: {len(label_columns)}")
    print(f"Rows read: {counts['rows_read']}")
    print(f"Rows missing image (initial): {counts['rows_missing_image_initial']}")
    print(f"Download attempted: {counts['download_attempted']}")
    print(f"Download succeeded: {counts['download_succeeded']}")
    print(f"Download failed: {counts['download_failed']}")
    print(
        "Rows skipped missing image (after download attempts): "
        f"{counts['rows_skipped_missing_after_download']}"
    )
    print(f"Rows dropped by label_mode: {counts['rows_dropped_label_mode']}")
    print(f"Manifest rows written: {len(rows)}")
    print(
        "Written split counts: "
        f"train={counts['written_train']}, "
        f"development={counts['written_development']}, "
        f"test={counts['written_test']}, "
        f"unknown={counts['written_unknown']}"
    )
    print(f"Manifest path: {output_file}")


if __name__ == "__main__":
    main()
