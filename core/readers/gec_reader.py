from abc import ABC, abstractmethod
from typing import Any, List, Union

from core.utils import get_logger, remove_space

from core.data.objects import Dataset, Sample

LOGGER = get_logger(__name__)


class BaseDataReaderWriter(ABC):
    """Base abstract class for data readers and writers.

    Provides common functionality for reading and writing datasets with various processing options.
    """

    def classname(self) -> str:
        """Returns the class name of the current instance.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__

    def read(
        self,
        file_input: Any,
        max_sample: int = -1,
        flatten: bool = False,
        remove_empty: bool = False,
        remove_unchanged: bool = False,
        remove_whitespace: bool = False,
        **kwargs: Any,
    ) -> Dataset:
        """Reads data from input file(s) and applies post-processing.

        Args:
            file_input: Path to input file or list of file paths
            max_sample: Maximum number of samples to read (-1 for all)
            flatten: Whether to flatten multi-target samples
            remove_empty: Whether to remove empty targets
            remove_unchanged: Whether to remove unchanged targets
            remove_whitespace: Whether to remove whitespace from text
            **kwargs: Additional arguments for specific reader implementations

        Returns:
            Dataset: The processed dataset
        """
        dataset = self._read(file_input=file_input, max_sample=max_sample, **kwargs)
        dataset = self.read_post(
            dataset,
            flatten=flatten,
            remove_empty=remove_empty,
            remove_unchanged=remove_unchanged,
            remove_whitespace=remove_whitespace,
        )

        LOGGER.info(f"{self.classname()}: Read {len(dataset)} samples from {file_input}.")
        return dataset

    def write(
        self, dataset: Dataset, file_output: Any, flatten: bool = False, max_sample: int = -1, **kwargs: Any
    ) -> None:
        """Writes dataset to output file(s).

        Args:
            dataset: The dataset to write
            file_output: Path to output file or list of file paths
            flatten: Whether to flatten multi-target samples before writing
            max_sample: Maximum number of samples to write (-1 for all)
            **kwargs: Additional arguments for specific writer implementations
        """
        dataset = self.flatten_dataset(dataset) if flatten else dataset
        num_sample = min(len(dataset), max_sample) if max_sample > 0 else len(dataset)
        LOGGER.info(f"{self.classname()}: Write {num_sample} samples to {file_output}")
        self._write(dataset=dataset, file_output=file_output, max_sample=max_sample, **kwargs)

    @abstractmethod
    def _read(self, file_input: Union[str, List[str]], **kwargs: Any) -> Dataset:
        """Abstract method to be implemented by subclasses for reading data.

        Args:
            file_input: Path to input file or list of file paths
            **kwargs: Additional arguments for specific reader implementations

        Returns:
            Dataset: The read dataset
        """
        raise NotImplementedError()

    @abstractmethod
    def _write(self, dataset: Dataset, file_output: Any, max_sample: int = -1, **kwargs: Any) -> None:
        """Abstract method to be implemented by subclasses for writing data.

        Args:
            dataset: The dataset to write
            file_output: Path to output file or other output destination
            max_sample: Maximum number of samples to write
            **kwargs: Additional arguments for specific writer implementations
        """
        raise NotImplementedError

    def read_post(
        self,
        dataset: Dataset,
        flatten: bool = False,
        remove_empty: bool = False,
        remove_unchanged: bool = False,
        remove_whitespace: bool = False,
    ) -> Dataset:
        """Post-process the dataset.

        Args:
            dataset: The dataset to process
            flatten: Whether to flatten multi-target samples
            remove_empty: Remove empty sentences
            remove_unchanged: Remove unchanged targets
            remove_whitespace: Remove whitespaces of sentences

        Returns:
            Dataset: New processed dataset
        """
        if not sum([flatten, remove_empty, remove_unchanged, remove_whitespace]):
            return dataset

        new_dataset = Dataset()
        for sample in dataset:
            source = sample.source.copy()
            target = sample.target.copy()
            if not source or not source[0]:
                raise ValueError(f"Source can not be empty: {sample}")

            if remove_empty:
                target = filter(None, target)

            if remove_unchanged:
                target = filter(lambda x: x != source[0], target)

            if remove_whitespace:
                source = remove_space(source)
                target = remove_space(target)

            new_sample = Sample(index=len(new_dataset), source=source, target=target)
            new_dataset.append(new_sample)

        if flatten:
            new_dataset = self.flatten_dataset(new_dataset)
        return new_dataset

    def flatten_dataset(self, dataset: Dataset) -> Dataset:
        """Flatten the dataset to an one-target dataset.

        Args:
            dataset: Input dataset with potentially multiple targets per sample

        Returns:
            Dataset: Output dataset with only one-target samples
        """
        new_dataset = Dataset()
        for sample in dataset:
            for target in sample.target:
                new_sample = Sample(index=len(new_dataset), source=sample.source.copy(), target=[target])
                new_dataset.append(new_sample)
        return new_dataset
