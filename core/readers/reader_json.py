import json
from typing import Any

from core.data.objects import Dataset

from .gec_reader import BaseDataReaderWriter


class JsonDataReaderWriter(BaseDataReaderWriter):
    """Reader and writer for JSON format datasets.

    Handles reading from and writing to JSON files.
    """

    def _read(self, file_input: str, max_sample: int = -1) -> Dataset:
        """Read dataset from a JSON file.

        Args:
            file_input: Path to the JSON file
            max_sample: Maximum number of samples to read

        Returns:
            Dataset: The dataset read from the JSON file
        """
        with open(file_input, "r", encoding="utf-8") as f:
            dataset = Dataset.model_validate_json(json.load(f))

        if max_sample > 0:
            dataset.samples = dataset.samples[:max_sample]
        return dataset

    def _write(self, dataset: Dataset, file_output: Any, max_sample: int = -1) -> None:
        """Write dataset to a JSON file.

        Args:
            dataset: The dataset to write
            file_output: Path to the output JSON file
            max_sample: Maximum number of samples to write
        """
        dataset_json = dataset.model_dump()
        if max_sample > 0:
            dataset_json["samples"] = dataset_json["samples"][:max_sample]

        with open(file_output, "w", encoding="utf-8") as f:
            json.dump(dataset_json, f, ensure_ascii=False, indent=2)
