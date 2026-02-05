# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Data loading utilities for diffgentor."""

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


@dataclass
class DatasetStats:
    """Statistics for a dataset scan."""

    total: int
    completed: int
    pending: int
    source_file: str

    def __str__(self) -> str:
        return (
            f"Total: {self.total}, Completed: {self.completed}, "
            f"Pending: {self.pending}"
        )


class LazyParquetDataset:
    """
    Lazy-loading parquet dataset with fast index scanning.

    Features:
    - Fast index-only scan mode (no image decoding)
    - On-demand row loading
    - Resume support with completed task filtering
    - Two-level task distribution (node + process)
    - Custom output filename column support
    """

    def __init__(
        self,
        data_path: str,
        load_mode: str = "index_only",
        output_name_column: Optional[str] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to parquet file or directory containing parquet files
            load_mode: "index_only" for fast scanning, "full" for immediate loading
            output_name_column: Optional column name to use for custom output filenames
        """
        self.data_path = data_path
        self.load_mode = load_mode
        self.output_name_column = output_name_column
        self._path = Path(data_path)

        # Internal state
        self._parquet_files: List[str] = []
        self._index_to_parquet: Dict[int, str] = {}
        self._all_indices: List[int] = []
        self._index_to_output_name: Dict[int, str] = {}  # Maps index to custom output name
        self._loaded_data: Optional[List[Dict[str, Any]]] = None

        self._scan_parquet_files()

    def _scan_parquet_files(self) -> None:
        """Scan parquet files and build index mapping."""
        if self._path.is_file() and self._path.suffix == ".parquet":
            self._parquet_files = [str(self._path)]
        elif self._path.is_dir():
            # Find all parquet files
            parquet_files = sorted(self._path.glob("*.parquet"))
            if not parquet_files:
                parquet_files = sorted(self._path.glob("data-*.parquet"))
            self._parquet_files = [str(f) for f in parquet_files]
        else:
            warnings.warn(f"Invalid data path: {self.data_path}")
            return

        if not self._parquet_files:
            warnings.warn(f"No parquet files found in: {self.data_path}")
            return

        # Fast index scan using PyArrow
        self._scan_indices()

        # If full mode, load all data immediately
        if self.load_mode == "full":
            self._load_all_data()

    def _scan_indices(self) -> None:
        """Fast scan of parquet index column (and optional output name column)."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            warnings.warn("pyarrow not available, falling back to full load")
            self.load_mode = "full"
            self._load_all_data()
            return

        # Determine which columns to read
        columns_to_read = ["index"]
        if self.output_name_column:
            columns_to_read.append(self.output_name_column)

        for pf in self._parquet_files:
            try:
                # Read only the necessary columns - very fast
                table = pq.read_table(pf, columns=columns_to_read)
                indices = table.column("index").to_pylist()

                # Read custom output names if specified
                output_names = None
                if self.output_name_column:
                    try:
                        output_names = table.column(self.output_name_column).to_pylist()
                    except KeyError:
                        warnings.warn(
                            f"Column '{self.output_name_column}' not found in {pf}, "
                            f"falling back to index-based naming"
                        )
                        output_names = None

                for i, idx in enumerate(indices):
                    try:
                        idx_int = int(idx)
                        self._all_indices.append(idx_int)
                        if idx_int not in self._index_to_parquet:
                            self._index_to_parquet[idx_int] = pf
                        # Store custom output name if available
                        if output_names is not None and i < len(output_names):
                            output_name = output_names[i]
                            if output_name is not None and str(output_name).strip():
                                self._index_to_output_name[idx_int] = str(output_name).strip()
                    except (TypeError, ValueError):
                        continue
            except Exception as e:
                warnings.warn(f"Failed to scan parquet index {pf}: {e}")

    def _load_all_data(self) -> None:
        """Load all data from parquet files."""
        if self._loaded_data is not None:
            return

        self._loaded_data = []

        # Try HuggingFace datasets first (better image handling)
        try:
            from datasets import Dataset

            for pf in self._parquet_files:
                ds = Dataset.from_parquet(pf)
                for i in range(len(ds)):
                    row = dict(ds[i])
                    if "index" not in row:
                        row["index"] = len(self._loaded_data)
                    if "input_images" in row and not isinstance(row.get("input_images"), str):
                        row["_images_embedded"] = True
                    self._loaded_data.append(row)
            return
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback to pandas
        try:
            import pandas as pd

            for pf in self._parquet_files:
                df = pd.read_parquet(pf)
                rows = df.to_dict("records")
                for row in rows:
                    if "index" not in row:
                        row["index"] = len(self._loaded_data)
                    if "input_images" in row and not isinstance(row["input_images"], str):
                        row["_images_embedded"] = True
                    self._loaded_data.append(row)
        except ImportError:
            raise ImportError("datasets or pandas+pyarrow are required for parquet support")

    def __len__(self) -> int:
        """Return the number of samples."""
        if self._loaded_data is not None:
            return len(self._loaded_data)
        return len(self._all_indices)

    def get_all_indices(self) -> List[int]:
        """Get all sample indices."""
        if self._loaded_data is not None:
            return [int(row.get("index", i)) for i, row in enumerate(self._loaded_data)]
        return list(self._all_indices)

    def get_output_name(self, sample_index: int) -> Optional[str]:
        """Get custom output name for a sample index.

        Args:
            sample_index: The sample index

        Returns:
            Custom output name if available, None otherwise
        """
        # Check pre-scanned mapping first
        if sample_index in self._index_to_output_name:
            return self._index_to_output_name[sample_index]

        # If data is loaded, check the row
        if self._loaded_data is not None and self.output_name_column:
            for row in self._loaded_data:
                if int(row.get("index", -1)) == sample_index:
                    output_name = row.get(self.output_name_column)
                    if output_name is not None and str(output_name).strip():
                        return str(output_name).strip()
                    break

        return None

    def get_by_index(self, sample_index: int) -> Optional[Dict[str, Any]]:
        """
        Get a sample by its index field.

        For index_only mode, this will load the specific parquet file on demand.

        Args:
            sample_index: The 'index' field value

        Returns:
            Row dict if found, None otherwise
        """
        # If data is already loaded, search directly
        if self._loaded_data is not None:
            for row in self._loaded_data:
                if int(row.get("index", -1)) == sample_index:
                    return row
            return None

        # Lazy load from specific parquet file
        pf = self._index_to_parquet.get(sample_index)
        if pf is None:
            return None

        return self._load_row_from_parquet(pf, sample_index)

    def _load_row_from_parquet(self, parquet_file: str, sample_index: int) -> Optional[Dict[str, Any]]:
        """Load a specific row from a parquet file."""
        try:
            from datasets import Dataset

            ds = Dataset.from_parquet(parquet_file)
            for i in range(len(ds)):
                row = dict(ds[i])
                if int(row.get("index", -1)) == sample_index:
                    if "input_images" in row and not isinstance(row.get("input_images"), str):
                        row["_images_embedded"] = True
                    return row
            return None
        except ImportError:
            pass

        # Fallback to pandas
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_file)
            matches = df[df["index"] == sample_index]
            if not matches.empty:
                row = matches.iloc[0].to_dict()
                if "input_images" in row and not isinstance(row["input_images"], str):
                    row["_images_embedded"] = True
                return row
            return None
        except Exception:
            return None

    def load_rows_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Load multiple rows by their indices.

        Optimized to batch load from the same parquet file.

        Args:
            indices: List of index values to load

        Returns:
            List of row dicts (in same order as indices, None entries filtered out)
        """
        if self._loaded_data is not None:
            # Data already loaded, filter by indices
            index_set = set(indices)
            result = []
            for row in self._loaded_data:
                if int(row.get("index", -1)) in index_set:
                    result.append(row)
            return result

        # Group indices by parquet file for efficient loading
        from collections import defaultdict

        pf_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx in indices:
            pf = self._index_to_parquet.get(idx)
            if pf:
                pf_to_indices[pf].append(idx)

        result = []
        for pf, pf_indices in pf_to_indices.items():
            rows = self._load_rows_from_parquet(pf, set(pf_indices))
            result.extend(rows)

        return result

    def _load_rows_from_parquet(self, parquet_file: str, indices: set) -> List[Dict[str, Any]]:
        """Load multiple rows from a single parquet file."""
        rows = []

        try:
            from datasets import Dataset

            ds = Dataset.from_parquet(parquet_file)
            for i in range(len(ds)):
                row = dict(ds[i])
                if int(row.get("index", -1)) in indices:
                    if "input_images" in row and not isinstance(row.get("input_images"), str):
                        row["_images_embedded"] = True
                    rows.append(row)
            return rows
        except ImportError:
            pass

        # Fallback to pandas
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_file)
            for _, row_data in df.iterrows():
                row = row_data.to_dict()
                if int(row.get("index", -1)) in indices:
                    if "input_images" in row and not isinstance(row["input_images"], str):
                        row["_images_embedded"] = True
                    rows.append(row)
        except Exception as e:
            warnings.warn(f"Failed to load rows from {parquet_file}: {e}")

        return rows

    def scan_and_filter(
        self,
        output_dir: str,
        get_output_filename: Callable[[int], str],
        node_rank: int = 0,
        num_nodes: int = 1,
        process_rank: int = 0,
        num_processes: int = 1,
    ) -> tuple[DatasetStats, List[int]]:
        """
        Fast scan and filter dataset for pending tasks.

        This method:
        1. Scans all indices (fast, no image decoding)
        2. Checks which outputs already exist (supports custom output names)
        3. Distributes remaining tasks across nodes/processes

        Args:
            output_dir: Output directory to check for completed tasks
            get_output_filename: Function to generate output filename from index
                (used as fallback when output_name_column is not set or value is missing)
            node_rank: Current node rank
            num_nodes: Total number of nodes
            process_rank: Current process rank within node
            num_processes: Number of processes per node

        Returns:
            Tuple of (stats, pending_indices_for_this_process)
        """
        from diffgentor.utils.image import normalize_custom_output_path

        all_indices = self.get_all_indices()
        total = len(all_indices)

        # Check completed tasks
        output_path = Path(output_dir)
        completed_indices = set()
        pending_indices = []

        for idx in all_indices:
            # Determine output filename: use custom name if available, otherwise default
            custom_name = self.get_output_name(idx)
            if custom_name:
                # Use custom output name with normalized path
                output_filename = normalize_custom_output_path(custom_name)
            else:
                # Fallback to index-based naming
                output_filename = get_output_filename(idx)

            output_file = output_path / output_filename
            if output_file.exists():
                completed_indices.add(idx)
            else:
                pending_indices.append(idx)

        # Distribute pending tasks across nodes
        if num_nodes > 1:
            chunk_size = (len(pending_indices) + num_nodes - 1) // num_nodes
            node_start = node_rank * chunk_size
            node_end = min(node_start + chunk_size, len(pending_indices))
            node_indices = pending_indices[node_start:node_end]
        else:
            node_indices = pending_indices

        # Distribute within node across processes
        if num_processes > 1:
            chunk_size = (len(node_indices) + num_processes - 1) // num_processes
            proc_start = process_rank * chunk_size
            proc_end = min(proc_start + chunk_size, len(node_indices))
            process_indices = node_indices[proc_start:proc_end]
        else:
            process_indices = node_indices

        stats = DatasetStats(
            total=total,
            completed=len(completed_indices),
            pending=len(process_indices),
            source_file=self.data_path,
        )

        return stats, process_indices


class LazyCsvDataset:
    """
    Lazy-loading CSV dataset with fast scanning.

    Similar interface to LazyParquetDataset but for CSV files.
    Supports custom output filename column.
    """

    def __init__(
        self,
        data_path: str,
        load_mode: str = "index_only",
        output_name_column: Optional[str] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to CSV file
            load_mode: "index_only" for fast scanning, "full" for immediate loading
            output_name_column: Optional column name to use for custom output filenames
        """
        self.data_path = data_path
        self.load_mode = load_mode
        self.output_name_column = output_name_column
        self._path = Path(data_path)

        self._all_indices: List[int] = []
        self._index_to_output_name: Dict[int, str] = {}  # Maps index to custom output name
        self._loaded_data: Optional[List[Dict[str, Any]]] = None

        self._scan_file()

    def _scan_file(self) -> None:
        """Scan CSV file for indices and optional output names."""
        if not self._path.is_file():
            warnings.warn(f"CSV file not found: {self.data_path}")
            return

        if self.load_mode == "index_only":
            # Fast scan: read index column and optionally output_name_column
            with open(self._path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    idx = int(row.get("index", i))
                    self._all_indices.append(idx)
                    # Store custom output name if column is specified and has value
                    if self.output_name_column:
                        output_name = row.get(self.output_name_column)
                        if output_name is not None and str(output_name).strip():
                            self._index_to_output_name[idx] = str(output_name).strip()
        else:
            self._load_all_data()

    def _load_all_data(self) -> None:
        """Load all data from CSV."""
        if self._loaded_data is not None:
            return

        self._loaded_data = []
        with open(self._path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if "index" not in row:
                    row["index"] = i
                self._loaded_data.append(dict(row))

        self._all_indices = [int(row.get("index", i)) for i, row in enumerate(self._loaded_data)]

    def __len__(self) -> int:
        return len(self._all_indices)

    def get_all_indices(self) -> List[int]:
        return list(self._all_indices)

    def get_output_name(self, sample_index: int) -> Optional[str]:
        """Get custom output name for a sample index.

        Args:
            sample_index: The sample index

        Returns:
            Custom output name if available, None otherwise
        """
        # Check pre-scanned mapping first
        if sample_index in self._index_to_output_name:
            return self._index_to_output_name[sample_index]

        # If data is loaded, check the row
        if self._loaded_data is not None and self.output_name_column:
            for row in self._loaded_data:
                if int(row.get("index", -1)) == sample_index:
                    output_name = row.get(self.output_name_column)
                    if output_name is not None and str(output_name).strip():
                        return str(output_name).strip()
                    break

        return None

    def get_by_index(self, sample_index: int) -> Optional[Dict[str, Any]]:
        """Get a sample by its index field."""
        if self._loaded_data is None:
            self._load_all_data()

        for row in self._loaded_data:
            if int(row.get("index", -1)) == sample_index:
                return row
        return None

    def load_rows_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Load multiple rows by their indices."""
        if self._loaded_data is None:
            self._load_all_data()

        index_set = set(indices)
        return [row for row in self._loaded_data if int(row.get("index", -1)) in index_set]

    def scan_and_filter(
        self,
        output_dir: str,
        get_output_filename: Callable[[int], str],
        node_rank: int = 0,
        num_nodes: int = 1,
        process_rank: int = 0,
        num_processes: int = 1,
    ) -> tuple[DatasetStats, List[int]]:
        """Fast scan and filter dataset for pending tasks."""
        from diffgentor.utils.image import normalize_custom_output_path

        all_indices = self.get_all_indices()
        total = len(all_indices)

        # Check completed tasks
        output_path = Path(output_dir)
        completed_indices = set()
        pending_indices = []

        for idx in all_indices:
            # Determine output filename: use custom name if available, otherwise default
            custom_name = self.get_output_name(idx)
            if custom_name:
                # Use custom output name with normalized path
                output_filename = normalize_custom_output_path(custom_name)
            else:
                # Fallback to index-based naming
                output_filename = get_output_filename(idx)

            output_file = output_path / output_filename
            if output_file.exists():
                completed_indices.add(idx)
            else:
                pending_indices.append(idx)

        # Distribute pending tasks
        if num_nodes > 1:
            chunk_size = (len(pending_indices) + num_nodes - 1) // num_nodes
            node_start = node_rank * chunk_size
            node_end = min(node_start + chunk_size, len(pending_indices))
            node_indices = pending_indices[node_start:node_end]
        else:
            node_indices = pending_indices

        if num_processes > 1:
            chunk_size = (len(node_indices) + num_processes - 1) // num_processes
            proc_start = process_rank * chunk_size
            proc_end = min(proc_start + chunk_size, len(node_indices))
            process_indices = node_indices[proc_start:proc_end]
        else:
            process_indices = node_indices

        stats = DatasetStats(
            total=total,
            completed=len(completed_indices),
            pending=len(process_indices),
            source_file=self.data_path,
        )

        return stats, process_indices


def create_lazy_dataset(
    data_path: str,
    load_mode: str = "index_only",
    output_name_column: Optional[str] = None,
) -> Union[LazyParquetDataset, LazyCsvDataset]:
    """
    Create appropriate lazy dataset based on file type.

    Args:
        data_path: Path to data file or directory
        load_mode: "index_only" for fast scanning, "full" for immediate loading
        output_name_column: Optional column name to use for custom output filenames

    Returns:
        LazyParquetDataset or LazyCsvDataset
    """
    path = Path(data_path)

    if path.is_dir():
        return LazyParquetDataset(data_path, load_mode, output_name_column)
    elif path.suffix.lower() == ".parquet":
        return LazyParquetDataset(data_path, load_mode, output_name_column)
    elif path.suffix.lower() == ".csv":
        return LazyCsvDataset(data_path, load_mode, output_name_column)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_prompts(file_path: str) -> List[Dict[str, Any]]:
    """Load prompts from file.

    Supports:
    - JSONL: One JSON object per line with 'prompt' field
    - JSON: Array of objects with 'prompt' field
    - TXT: One prompt per line
    - CSV: With 'prompt' column

    Args:
        file_path: Path to prompts file

    Returns:
        List of prompt dictionaries with at least 'prompt' key
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return _load_jsonl(file_path)
    elif suffix == ".json":
        return _load_json(file_path)
    elif suffix == ".txt":
        return _load_txt(file_path)
    elif suffix == ".csv":
        return _load_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, str):
                data = {"prompt": data}
            if "index" not in data:
                data["index"] = i
            prompts.append(data)
    return prompts


def _load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        prompts = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                item = {"prompt": item}
            if "index" not in item:
                item["index"] = i
            prompts.append(item)
        return prompts
    else:
        raise ValueError("JSON file must contain an array")


def _load_txt(file_path: str) -> List[Dict[str, Any]]:
    """Load TXT file (one prompt per line)."""
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            prompts.append({"prompt": line, "index": i})
    return prompts


def _load_csv(file_path: str) -> List[Dict[str, Any]]:
    """Load CSV file."""
    prompts = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if "index" not in row:
                row["index"] = i
            prompts.append(dict(row))
    return prompts


def load_editing_data(file_path: str) -> List[Dict[str, Any]]:
    """Load editing data from CSV or Parquet file.

    Args:
        file_path: Path to data file (CSV or Parquet directory)

    Returns:
        List of data dictionaries
    """
    path = Path(file_path)

    if path.is_dir():
        # Parquet directory
        return _load_parquet_dir(file_path)
    elif path.suffix.lower() == ".parquet":
        return _load_parquet(file_path)
    elif path.suffix.lower() == ".csv":
        return _load_csv_data(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _load_csv_data(file_path: str) -> List[Dict[str, Any]]:
    """Load CSV data file."""
    rows = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if "index" not in row:
                row["index"] = i
            rows.append(dict(row))
    return rows


def _load_parquet(file_path: str) -> List[Dict[str, Any]]:
    """Load single parquet file.

    Tries to use HuggingFace datasets first (better image handling),
    falls back to pandas if not available.
    """
    # Try HuggingFace datasets first (better for embedded images)
    try:
        from datasets import Dataset

        ds = Dataset.from_parquet(file_path)
        rows = []
        for i in range(len(ds)):
            row = dict(ds[i])
            if "index" not in row:
                row["index"] = i
            # Mark that images are embedded (for parquet with image columns)
            if "input_images" in row and not isinstance(row.get("input_images"), str):
                row["_images_embedded"] = True
            rows.append(row)
        return rows
    except ImportError:
        pass

    # Fallback to pandas
    try:
        import pandas as pd

        df = pd.read_parquet(file_path)
        rows = df.to_dict("records")

        # Add index if not present
        for i, row in enumerate(rows):
            if "index" not in row:
                row["index"] = i
            # Mark that images are embedded (for parquet with image columns)
            if "input_images" in row and not isinstance(row["input_images"], str):
                row["_images_embedded"] = True

        return rows

    except ImportError:
        raise ImportError("datasets or pandas+pyarrow are required for parquet support")


def _load_parquet_dir(dir_path: str) -> List[Dict[str, Any]]:
    """Load parquet files from directory.

    Tries to use HuggingFace datasets first (better image handling),
    falls back to pandas if not available.
    """
    path = Path(dir_path)
    parquet_files = sorted(path.glob("*.parquet"))

    if not parquet_files:
        # Try glob pattern for data-*.parquet style
        parquet_files = sorted(path.glob("data-*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {dir_path}")

    # Try HuggingFace datasets first (better for embedded images)
    try:
        from datasets import Dataset

        parquet_pattern = str(path / "*.parquet")
        ds = Dataset.from_parquet(parquet_pattern)
        rows = []
        for i in range(len(ds)):
            row = dict(ds[i])
            row["index"] = i  # Re-index across all files
            if "input_images" in row and not isinstance(row.get("input_images"), str):
                row["_images_embedded"] = True
            rows.append(row)
        return rows
    except ImportError:
        pass
    except Exception:
        # datasets may fail on some parquet structures, fallback to pandas
        pass

    # Fallback to pandas
    try:
        import pandas as pd

        all_rows = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            rows = df.to_dict("records")
            for row in rows:
                if "input_images" in row and not isinstance(row["input_images"], str):
                    row["_images_embedded"] = True
            all_rows.extend(rows)

        # Re-index
        for i, row in enumerate(all_rows):
            row["index"] = i

        return all_rows

    except ImportError:
        raise ImportError("datasets or pandas+pyarrow are required for parquet support")


def save_csv_data(
    file_path: str,
    rows: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Save data to CSV file.

    Args:
        file_path: Output file path
        rows: List of data dictionaries
        fieldnames: Column names (auto-detected if None)
    """
    if not rows:
        return

    if fieldnames is None:
        # Collect all keys from all rows
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys)

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
