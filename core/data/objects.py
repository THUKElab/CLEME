import copy
from typing import Any, Iterator, List, Optional

from pydantic import BaseModel, Field

EDIT_TEMPLATE = "A {i1} {i2}|||{types}|||{target}|||REQUIRED|||-NONE-|||{index}"


class Edit(BaseModel):
    """Represents an edit operation between source and target text.

    This class models a single edit operation with information about the source and target
    intervals, tokens, and edit types. It provides methods for comparing edits and
    generating M2 format representations.
    """

    src_interval: List[int] = Field(default=None, description="Source interval")
    tgt_interval: List[int] = Field(default=None, description="Target interval")
    src_tokens: List[str] = Field(default=None, description="Source tokens")
    tgt_tokens: List[str] = Field(default=None, description="Target tokens")
    src_tokens_tok: Optional[Any] = Field(default=None, description="Tokenized source tokens")
    tgt_tokens_tok: Optional[Any] = Field(default=None, description="Tokenized Target tokens")
    tgt_index: int = Field(default=0, description="Target index")
    types: Optional[List[str]] = Field(default_factory=list, description="Edit types")
    weight: Optional[float] = Field(default=None, description="Edit weight")

    @property
    def source(self, limiter: str = " ") -> str:
        """Get the source text by joining source tokens.

        Args:
            limiter (str): The delimiter to join tokens with

        Returns:
            str: The joined source tokens
        """
        return limiter.join(self.src_tokens)

    @property
    def target(self, limiter: str = " ") -> str:
        """Get the target text by joining target tokens.

        Args:
            limiter (str): The delimiter to join tokens with

        Returns:
            str: The joined target tokens
        """
        return limiter.join(self.tgt_tokens)

    @property
    def m2(self) -> str:
        """Generate the M2 format representation of this edit.

        Returns:
            str: The edit in M2 format
        """
        return EDIT_TEMPLATE.format(
            i1=self.src_interval[0],
            i2=self.src_interval[1],
            types=", ".join(self.types),
            target=self.target or "-NONE-",
            index=self.tgt_index,
        )

    def is_valid(self) -> bool:
        """Check if the edit is valid by verifying target intervals.

        Returns:
            bool: True if the edit is valid, False otherwise
        """
        tgt_beg_idx = self.tgt_interval[0]
        tgt_end_idx = self.tgt_interval[1]
        return self.tgt_tokens[tgt_beg_idx:tgt_end_idx] == self.tgt_tokens

    def __eq__(self, other: "Edit") -> bool:
        """Compare this edit with another edit.

        Args:
            other (Edit): Another edit to compare with

        Returns:
            bool: True if edits are equal, False otherwise

        Raises:
            ValueError: If source intervals match but source tokens don't match
        """
        if self.src_interval == other.src_interval:
            if self.src_tokens != other.src_tokens:
                raise ValueError(f"Invalid Edit comparison:\nEdit 1:{self}\nEdit 2:{other}")
            elif self.tgt_tokens == other.tgt_tokens:
                return True
        return False

    def __hash__(self) -> int:
        return (
            hash(tuple(self.src_interval))
            + hash(tuple(self.tgt_tokens))
            + hash(tuple(self.src_tokens))
            + hash(tuple(self.types))
        )

    def __deepcopy__(self, memodict={}) -> "Edit":
        return Edit(
            src_interval=self.src_interval.copy(),
            tgt_interval=self.tgt_interval.copy(),
            src_tokens=self.src_tokens.copy(),
            tgt_tokens=self.tgt_tokens.copy(),
            src_tokens_tok=copy.deepcopy(self.src_tokens_tok),
            tgt_tokens_tok=copy.deepcopy(self.tgt_tokens_tok),
            types=self.types.copy(),
            weight=self.weight,
        )


class Chunk(Edit):
    """Represents a chunk of text with edit information.

    This class extends the Edit class with additional chunk index information,
    useful for chunk-level edit operations.
    """

    chunk_index: int = Field(default=None, description="Chunk index")

    def __eq__(self, other: "Chunk") -> bool:
        """Compare this chunk with another chunk.

        Args:
            other (Chunk): Another chunk to compare with

        Returns:
            bool: True if chunks are equal, False otherwise
        """
        if self.chunk_index == other.chunk_index:
            return super().__eq__(other)
        return False

    def __hash__(self) -> int:
        return super().__hash__() + hash(self.chunk_index)

    def __deepcopy__(self, memodict={}) -> "Chunk":
        return Chunk(
            src_interval=self.src_interval.copy(),
            tgt_interval=self.tgt_interval.copy(),
            src_tokens=self.src_tokens.copy(),
            tgt_tokens=self.tgt_tokens.copy(),
            src_tokens_tok=copy.deepcopy(self.src_tokens_tok),
            tgt_tokens_tok=copy.deepcopy(self.tgt_tokens_tok),
            types=self.types.copy(),
            weight=self.weight,
            chunk_index=self.chunk_index,
        )


class Sample(BaseModel):
    """Represents a sample with source and target sentences and their edits.

    This class models a sample containing source sentences, target sentences,
    and the edits that transform the source into the target. It provides methods
    for M2 format generation and sample validation.
    """

    index: int = Field(default=None, description="Sample index")
    source: List[str] = Field(default=None, description="Source sentences")
    target: List[str] = Field(default=None, description="Target sentences")
    edits: Optional[List[List[List[Edit]]]] = Field(default=None, description="Edits extracted from source to target")
    chunks: Optional[List[List[List[Chunk]]]] = Field(default=None, description="Chunks extracted by CLEME")

    def dump_m2(self) -> str:
        """Generate M2 format representation of this sample.

        Returns:
            str: The sample in M2 format

        Raises:
            ValueError: If source or edits are invalid
        """
        if len(self.source) != 1 or not self.source:
            raise ValueError(f"Error source: {self.source}")
        if not self.edits or len(self.edits[0]) != len(self.target):
            raise ValueError(f"Error edits: {self.edits}")

        result = f"S {self.source[0]}\n"
        for idx, (sample_target, sample_edits) in enumerate(zip(self.target, self.edits[0])):
            result += f"T{idx} {sample_target}\n"
            for edit in sample_edits:
                if idx != edit.tgt_index:
                    raise ValueError(f"Inconsistent target index: {idx} != {edit.tgt_index}")
                result += edit.m2 + "\n"
        return result

    def contains_empty(self) -> bool:
        """Check if the sample contains any empty source or target sentences.

        Returns:
            bool: True if there are empty sentences, False otherwise
        """
        return any([not x for x in self.source + self.target])

    def __deepcopy__(self, memodict={}) -> "Sample":
        return Sample(
            index=self.index,
            source=self.source.copy(),
            target=self.target.copy(),
            edits=copy.deepcopy(self.edits),
            chunks=copy.deepcopy(self.chunks),
        )

    def has_repeated_target(self) -> bool:
        """Check if the sample has repeated target sentences.

        Returns:
            bool: True if there are repeated targets, False otherwise
        """
        return len(self.target) != len(set(self.target))

    def has_unchanged_target(self) -> bool:
        """Check if any target sentence is identical to the source.

        Returns:
            bool: True if there are unchanged targets, False otherwise
        """
        return any([x == self.source[0] for x in self.target])


class Dataset(BaseModel):
    """Represents a collection of samples.

    This class provides methods for managing a collection of samples,
    including iteration, indexing, appending, extending, merging, and
    flattening operations.
    """

    samples: List[Sample] = Field(default_factory=list, description="Samples included")

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        """Get an iterator over the samples."""
        return iter(self.samples)

    def __getitem__(self, item: int) -> Sample:
        """Get a sample by index."""
        return self.samples[item]

    def append(self, sample: Sample) -> None:
        """Append a sample to the dataset."""
        self.samples.append(sample)

    def extend(self, dataset: "Dataset") -> None:
        """Extend this dataset with samples from another dataset.

        Updates the indices of the appended samples to maintain consistency.

        Args:
            dataset (Dataset): Dataset to extend from
        """
        orig_len = len(self)
        self.samples.extend(dataset.samples)
        for sample_idx in range(orig_len, len(self.samples)):
            self.samples[sample_idx].index = sample_idx

    def merge(self, dataset: "Dataset"):
        """Merge this dataset with another dataset.

        Combines targets and edits from corresponding samples in both datasets.

        Args:
            dataset (Dataset): Dataset to merge with

        Raises:
            ValueError: If datasets have different lengths or incompatible sources
        """
        if len(self) != len(dataset):
            raise ValueError(f"Both datasets must contain equal samples: {len(self)} != {len(dataset)}.")

        for sample1, sample2 in zip(self, dataset):
            if sample1.source != sample2.source:
                raise ValueError(
                    f"Both samples must contain same sources:\n"
                    f"Source 1: {sample1.source}\n"
                    f"Source 2: {sample2.source}"
                )
            sample1.target.extend(sample2.target)
            sample1.edits[0].extend(sample2.edits[0])

    def reorder(self) -> None:
        """Reorder the sample indices to match their position in the dataset."""
        for idx, sample in enumerate(self.samples):
            sample.index = idx

    def flatten(self) -> "Dataset":
        """Flatten the dataset by creating individual samples for each source-target pair.

        Returns:
            Dataset: A new flattened dataset
        """
        new_dataset = Dataset()
        for sample in self.samples:
            for sid, src in enumerate(sample.source):
                for tid, tgt in enumerate(sample.target):
                    new_edits = [[copy.deepcopy(sample.edits[sid][tid])]] if sample.edits is not None else None
                    new_chunks = [[copy.deepcopy(sample.chunks[sid][tid])]] if sample.chunks is not None else None
                    new_sample = Sample(
                        index=len(new_dataset), source=[src], target=[tgt], edits=new_edits, chunks=new_chunks
                    )
                    new_dataset.append(new_sample)
        return new_dataset


def apply_edits(src_tokens: List[str], edits: List[Edit]) -> List[str]:
    """Generate target tokens by applying edits to source tokens.

    This function applies a sequence of edits to the source tokens to produce
    the target tokens. It performs sanity checks to ensure edit consistency.

    Args:
        src_tokens (List[str]): Source tokens to apply edits to
        edits (List[Edit]): List of edits to apply

    Returns:
        List[str]: The resulting target tokens after applying all edits

    Raises:
        ValueError: If source or target tokens in edits are inconsistent
    """
    tgt_offset, tgt_tokens = 0, src_tokens.copy()
    for edit in edits:
        src_beg_idx, src_end_idx = edit.src_interval[0], edit.src_interval[1]
        tgt_beg_idx = src_beg_idx + tgt_offset
        tgt_end_idx = tgt_beg_idx + len(edit.tgt_tokens)

        tgt_tokens[tgt_beg_idx : src_end_idx + tgt_offset] = edit.tgt_tokens
        tgt_offset += len(edit.tgt_tokens) - len(edit.src_tokens)

        # Sanity Check
        if edit.src_tokens != src_tokens[src_beg_idx:src_end_idx]:
            raise ValueError(f"Inconsistent Source Tokens: {edit} != {src_tokens[src_beg_idx: src_end_idx]}")
        if edit.tgt_tokens != tgt_tokens[tgt_beg_idx:tgt_end_idx]:
            raise ValueError(f"Inconsistent Target Tokens: {edit} != {tgt_tokens[src_beg_idx: src_end_idx]}")
    return tgt_tokens
