import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import Benchmark, BenchmarkDataPoint


METADATA_COLUMNS = {
    "dicom_id",
    "subject_id",
    "study_id",
    "ViewPosition",
    "ViewCodeSequence_CodeMeaning",
    "fpath",
    "split",
}


def _safe_strip(value: Optional[str]) -> str:
    return (value or "").strip()


class MimicCXRBenchmark(Benchmark):
    """MIMIC-CXR benchmark using manifest-first, source-CSV fallback loading."""

    def __init__(self, data_dir: str, **kwargs):
        self.manifest_path = kwargs.get("manifest_path", None)
        self.source_csv = kwargs.get("source_csv", None)
        self.image_root = kwargs.get("image_root", None)
        self.label_mode = kwargs.get("label_mode", "single_positive_only")

        if self.label_mode not in {"single_positive_only", "allow_multiple_findings"}:
            raise ValueError(
                "label_mode must be one of: single_positive_only, allow_multiple_findings"
            )

        super().__init__(data_dir, **kwargs)

    def _load_data(self) -> None:
        manifest_path = self._resolve_manifest_path()
        if manifest_path is not None:
            self._load_from_manifest(manifest_path)
            return

        source_csv = self._resolve_source_csv_path()
        image_root = self._resolve_image_root(source_csv)
        self._load_from_source_csv(source_csv=source_csv, image_root=image_root)

    def _resolve_manifest_path(self) -> Optional[Path]:
        if not self.manifest_path:
            return None

        manifest_path = Path(self.manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = Path(self.data_dir) / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        return manifest_path

    def _resolve_source_csv_path(self) -> Path:
        if not self.source_csv:
            raise ValueError(
                "source_csv is required when manifest_path is not provided for mimic_cxr benchmark."
            )

        source_csv_path = Path(self.source_csv)
        if not source_csv_path.is_absolute():
            source_csv_path = Path(self.data_dir) / source_csv_path
        if not source_csv_path.exists():
            raise FileNotFoundError(f"Source CSV not found: {source_csv_path}")
        return source_csv_path

    def _resolve_image_root(self, source_csv_path: Path) -> Path:
        if self.image_root:
            image_root_path = Path(self.image_root)
            if not image_root_path.is_absolute():
                image_root_path = Path(self.data_dir) / image_root_path
            return image_root_path

        # .../files/cxr-lt-iccv-workshop-cvamd/2.0.0/cxr-lt-2024/<csv>
        # -> .../files/mimic-cxr-jpg/2.0.0
        data_dir = source_csv_path.parent
        if len(data_dir.parents) < 3:
            raise ValueError(
                f"Cannot infer image_root from source_csv path: {source_csv_path}. "
                "Please set benchmark.image_root explicitly."
            )
        files_root = data_dir.parents[2]
        return files_root / "mimic-cxr-jpg" / "2.0.0"

    def _load_from_manifest(self, manifest_path: Path) -> None:
        print(f"Loading MIMIC-CXR benchmark from manifest: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed manifest row {i}: {e}")
                    continue

                data_point = BenchmarkDataPoint(
                    id=item.get("id", f"mimic_manifest_{i}"),
                    text=item.get("text", ""),
                    images=item.get("images"),
                    correct_answer=item.get("correct_answer", ""),
                    metadata=item.get("metadata", {}),
                )
                self.data_points.append(data_point)

    def _load_from_source_csv(self, source_csv: Path, image_root: Path) -> None:
        header = self._read_header(source_csv)
        label_columns = self._compute_label_columns(header)
        if not label_columns:
            raise ValueError(
                f"No label columns detected in {source_csv} after excluding metadata columns."
            )

        class_to_option = {label: str(i + 1) for i, label in enumerate(label_columns)}
        question_text = self._build_question_text(class_to_option)
        has_split_column = "split" in header
        inferred_split = self._infer_split_from_filename(source_csv)

        counts = {
            "rows_read": 0,
            "rows_missing_image": 0,
            "rows_dropped_label_mode": 0,
            "rows_written": 0,
        }
        split_counts: Dict[str, int] = {
            "train": 0,
            "development": 0,
            "test": 0,
            "unknown": 0,
        }

        with source_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=2):
                counts["rows_read"] += 1

                if has_split_column:
                    split = self._normalize_split(row.get("split", ""), source_csv)
                    split_source = "csv"
                else:
                    split = inferred_split
                    split_source = "filename"

                fpath = self._normalize_fpath(row.get("fpath", ""))
                image_path = image_root / fpath if fpath else image_root
                if not fpath or not image_path.exists():
                    counts["rows_missing_image"] += 1
                    continue

                positives: List[str] = []
                for label in label_columns:
                    label_value = self._parse_binary_label(
                        row.get(label, ""),
                        file_path=source_csv,
                        row_number=row_idx,
                        column=label,
                    )
                    if label_value == 1:
                        positives.append(label)

                if self.label_mode == "single_positive_only":
                    if len(positives) != 1:
                        counts["rows_dropped_label_mode"] += 1
                        continue
                elif not positives:
                    counts["rows_dropped_label_mode"] += 1
                    continue

                positive_options = sorted((class_to_option[p] for p in positives), key=int)
                correct_answer = (
                    positive_options[0]
                    if self.label_mode == "single_positive_only"
                    else ",".join(positive_options)
                )

                split_counts[split] = split_counts.get(split, 0) + 1

                dicom_id = _safe_strip(row.get("dicom_id"))
                subject_id = _safe_strip(row.get("subject_id"))
                study_id = _safe_strip(row.get("study_id"))
                data_point_id = f"mimic_{subject_id}_{study_id}_{dicom_id}"

                metadata = {
                    "dataset": "mimic_cxr_lt",
                    "split": split,
                    "split_source": split_source,
                    "source_csv": source_csv.name,
                    "dicom_id": dicom_id,
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "view_position": _safe_strip(row.get("ViewPosition")),
                    "view_code_meaning": _safe_strip(row.get("ViewCodeSequence_CodeMeaning")),
                    "fpath": fpath,
                    "classes": list(label_columns),
                    "class_to_option": class_to_option,
                    "positive_classes": positives,
                    "positive_options": positive_options,
                    "multi_select": self.label_mode == "allow_multiple_findings",
                    "label_validation": "binary_int_0_1",
                    "validated_label_columns": len(label_columns),
                }

                self.data_points.append(
                    BenchmarkDataPoint(
                        id=data_point_id,
                        text=question_text,
                        images=[str(image_path)],
                        correct_answer=correct_answer,
                        metadata=metadata,
                    )
                )
                counts["rows_written"] += 1

        print(f"Source CSV: {source_csv}")
        print(f"Image root: {image_root}")
        print(f"Label columns: {len(label_columns)}")
        print(f"Rows read: {counts['rows_read']}")
        print(f"Rows missing image (skipped): {counts['rows_missing_image']}")
        print(f"Rows dropped by label_mode: {counts['rows_dropped_label_mode']}")
        print(f"Rows written: {counts['rows_written']}")
        print(
            "Written split counts: "
            f"train={split_counts['train']}, "
            f"development={split_counts['development']}, "
            f"test={split_counts['test']}, "
            f"unknown={split_counts['unknown']}"
        )

    def _read_header(self, csv_path: Path) -> List[str]:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames or [])

    def _compute_label_columns(self, header: Sequence[str]) -> List[str]:
        return [col for col in header if col not in METADATA_COLUMNS]

    def _parse_binary_label(
        self, raw_value: Optional[str], file_path: Path, row_number: int, column: str
    ) -> int:
        value = _safe_strip(raw_value)
        if value == "0":
            return 0
        if value == "1":
            return 1
        raise ValueError(
            f"Invalid non-binary label value in {file_path} row {row_number}, "
            f"column '{column}': '{raw_value}'. Expected integer 0 or 1."
        )

    def _normalize_split(self, raw_split: Optional[str], source_file: Path) -> str:
        raw = _safe_strip(raw_split).lower()
        if raw == "val":
            return "development"
        if raw in {"train", "development", "test"}:
            return raw
        raise ValueError(
            f"Unexpected split value '{raw_split}' in {source_file.name}. "
            "Expected train/val/development/test."
        )

    def _infer_split_from_filename(self, source_file: Path) -> str:
        name = source_file.name.lower()
        if "development" in name or "dev" in name or "val" in name:
            return "development"
        if "train" in name:
            return "train"
        if "test" in name:
            return "test"
        return "unknown"

    def _normalize_fpath(self, value: Optional[str]) -> str:
        return _safe_strip(value).lstrip("./").lstrip("/")

    def _build_question_text(self, class_to_option: Dict[str, str]) -> str:
        option_lines = [f"{token}. {class_name}" for class_name, token in class_to_option.items()]
        options_block = "\n".join(option_lines)

        if self.label_mode == "single_positive_only":
            instruction = "Choose exactly one option."
            response_format = "Respond with your final answer in the format: \\boxed{12}"
            question = "Which class best describes this chest X-ray?"
        else:
            instruction = "One or more options may be correct. Select all that apply."
            response_format = "Respond with your final answer in the format: \\boxed{3,12}"
            question = "Which classes are present in this chest X-ray?"

        return (
            f"{question}\n"
            f"{instruction}\n\n"
            "Options:\n"
            f"{options_block}\n\n"
            f"{response_format}"
        )

    def _filter_subset(self, subset_file: Path) -> None:
        """Filter questions by data point ID from a text file (one ID per line)."""
        if not subset_file.exists():
            raise FileNotFoundError(f"Could not find subset file: {subset_file}")

        subset_ids = set()
        with open(subset_file, "r", encoding="utf-8") as f:
            for line in f:
                data_point_id = line.strip()
                if data_point_id:
                    subset_ids.add(data_point_id)

        original_count = len(self.data_points)
        self.data_points = [dp for dp in self.data_points if dp.id in subset_ids]

        print(
            f"Filtered MIMIC-CXR benchmark from {original_count} to {len(self.data_points)} "
            f"using subset file {subset_file}"
        )
