import json
from pathlib import Path
import zipfile
from typing import Dict, Optional, Any
from .base import Benchmark, BenchmarkDataPoint


class ChestAgentBenchBenchmark(Benchmark):
    """ChestAgentBench benchmark for complex CXR interpretation and reasoning.

    Loads the dataset from a local metadata.jsonl file and parses each entry into a BenchmarkDataPoint.
    """

    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, **kwargs)

    def _load_data(self) -> None:
        # Ensure figures are available (extract if only the zip is present)
        self._ensure_figures_extracted()

        # Check if metadata.jsonl exists
        metadata_path = Path(self.data_dir) / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Could not find metadata.jsonl in {self.data_dir}")
        print(f"Loading ChestAgentBench from local file: {metadata_path}")

        # Load metadata.jsonl
        with open(metadata_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)
                    data_point = self._parse_item(item, i)
                    if data_point:
                        self.data_points.append(data_point)
                except Exception as e:
                    print(f"Error loading item {i}: {e}")
                    continue

    def _parse_item(self, item: Dict[str, Any], index: int) -> Optional[BenchmarkDataPoint]:
        # Extract required fields
        question_id = item.get("full_question_id")
        question = item.get("question", "")
        correct_answer = item.get("answer", "")

        # Map image paths to local figures directory
        images = item.get("images", [])
        local_images = None
        if images:
            local_images = []
            for img in images:
                full_path = Path(self.data_dir) / img
                local_images.append(str(full_path))

        # Extract metadata
        metadata = dict(item)
        metadata["dataset"] = "chestagentbench"

        # Return data point
        return BenchmarkDataPoint(
            id=question_id,
            text=question,
            images=local_images,
            correct_answer=correct_answer,
            metadata=metadata,
        )

    def _ensure_figures_extracted(self) -> None:
        """Ensure the ChestAgentBench figures directory exists, extracting from the zip if needed."""
        figures_dir = Path(self.data_dir) / "figures"
        if figures_dir.exists():
            return

        zip_path = Path(self.data_dir) / "figures.zip"
        if not zip_path.exists():
            print(
                f"Warning: figures directory not found at {figures_dir}, and no figures.zip present. "
                "Image paths will be invalid."
            )
            return

        try:
            print(f"Extracting ChestAgentBench figures from {zip_path} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(Path(self.data_dir))
            print(f"Figures extracted to {figures_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract ChestAgentBench figures from {zip_path}: {e}") from e

    def _filter_subset(self, subset_file: Path) -> None:
        """Load a subset of data points based on IDs specified in a text file.

        Args:
            subset_file: Path to a text file containing question_id values (one per line)
        """
        if not subset_file.exists():
            raise FileNotFoundError(f"Could not find subset file: {subset_file}")

        # Read IDs from the subset file
        subset_ids = set()
        with open(subset_file, "r", encoding="utf-8") as f:
            for line in f:
                question_id = line.strip()
                if question_id:  # Skip empty lines
                    subset_ids.add(question_id)

        print(f"Filtering to {len(subset_ids)} IDs from subset file: {subset_file}")

        # Filter data_points to only include those with matching question_id
        # Note: subset file contains question_id format (e.g., "2364_8353802248292")
        # but data_point.id is full_question_id (e.g., "2364_2364_8353802248292")
        # So we need to check metadata.question_id instead
        original_count = len(self.data_points)
        filtered_points = []
        for data_point in self.data_points:
            # Check question_id from metadata (stored from original JSON)
            question_id = data_point.metadata.get("question_id") if data_point.metadata else None
            if question_id in subset_ids:
                filtered_points.append(data_point)

        self.data_points = filtered_points
        print(f"Filtered from {original_count} to {len(self.data_points)} data points")
