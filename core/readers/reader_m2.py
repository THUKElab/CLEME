import logging
import re
from typing import List, Tuple

from core.data.objects import Dataset, Edit, Sample
from core.utils import get_logger, remove_space

from .gec_reader import BaseDataReaderWriter

LOGGER = get_logger(__name__, level=logging.INFO)

DELIMITER_M2 = "|||"
EDIT_NONE_TYPE = {"noop", "NA"}
EDIT_NONE_CORRECTION = {"-NONE-"}


class M2DataReaderWriter(BaseDataReaderWriter):
    """Reader and writer for M2 format datasets used in grammatical error correction.

    Handles parsing and generating M2 format files with edit annotations.
    """

    def _read(self, file_input: str, max_sample: int = -1, char_level: bool = False) -> Dataset:
        """Read dataset from an M2 format file.

        Args:
            file_input: Path to the M2 file
            max_sample: Maximum number of samples to read
            char_level: Whether to segment by character instead of word

        Returns:
            Dataset: The dataset read from the M2 file
        """
        with open(file_input, "r", encoding="utf-8") as f:
            content = f.read().strip()
        sample_lines_list: List[List[str]] = [x.split("\n") for x in content.split("\n\n")]

        dataset = Dataset()
        for sample_lines in sample_lines_list:
            if 0 <= max_sample <= len(dataset):
                break
            sample = self.parse_sample_lines(sample_lines, char_level=char_level)
            dataset.append(sample)
        dataset.reorder()
        return dataset

    def _write(self, dataset: Dataset, file_output: str, max_sample: int = -1) -> None:
        """Write dataset to an M2 format file.

        Args:
            dataset: The dataset to write
            file_output: Path to the output M2 file
            max_sample: Maximum number of samples to write
        """
        with open(file_output, "w", encoding="utf-8") as f:
            for idx, sample in enumerate(dataset):
                if 0 < max_sample <= idx:
                    break
                f.write(sample.dump_m2() + "\n")

    def parse_sample_lines(self, sample_lines: List[str], char_level: bool = False) -> Sample:
        """Parse M2 lines of a sample into a sample object.

        Args:
            sample_lines: List of lines from M2 file for one sample
            segment_char: Whether to segment by character instead of word

        Returns:
            Sample: The parsed sample object
        """

        # Source line
        if not sample_lines[0].startswith("S"):
            raise ValueError(f"Not begin with source: {sample_lines}")
        src_sent = sample_lines[0].replace("S ", "", 1)

        if char_level:
            src_tokens = [char for char in remove_space(src_sent.strip())]
        else:
            src_tokens = src_sent.split()

        if not src_tokens:
            raise ValueError(f"Empty source sentence: {sample_lines}")

        # Split target lines
        edit_lines_list: List[List[str]] = []
        edit_lines = []
        target_index = -1
        skip_flag, prev_target = False, False
        for line in sample_lines[1:]:
            if line.startswith("T"):
                if prev_target:
                    edit_lines_list.append([])
                rex = re.search("T(\\d+)-A(\\d+)", line.split()[0])
                if rex is not None:
                    skip_flag = True if int(rex.group(2)) != 0 else False
                prev_target = True

            elif line.startswith("A"):
                curr_target_index = int(line.split(DELIMITER_M2)[-1])
                if curr_target_index == target_index + 1:
                    if edit_lines:
                        edit_lines_list.append(edit_lines)
                    edit_lines = []
                    target_index = curr_target_index

                if not skip_flag:
                    # Add target line
                    edit_lines.append(line)
                else:
                    LOGGER.warning(f"Skip overlapped lines: {line}")
                prev_target = False
            else:
                raise ValueError(f"Error line: {line}")

        # Don't forget the last target
        edit_lines_list.append(edit_lines)

        # Build target sentences
        sample_targets, sample_edits = [], []
        for edit_lines in edit_lines_list:
            tgt_tokens, edits = self.parse_edit_lines(src_tokens=src_tokens, lines=edit_lines)
            if char_level:
                tgt_sent = "".join(tgt_tokens)
            else:
                tgt_sent = " ".join(tgt_tokens)
            sample_targets.append(tgt_sent)
            sample_edits.append(edits)

        return Sample(source=[src_sent], target=sample_targets, edits=[sample_edits])

    def parse_edit_lines(self, src_tokens: List[str], lines: List[str]) -> Tuple[List[str], List[Edit]]:
        """Parse target lines into target tokens and edits.

        Args:
            src_tokens: Source tokens
            lines: Target lines with edit annotations

        Returns:
            Tuple containing target tokens and list of edits
        """
        edits = []
        tgt_offset = 0
        tgt_tokens = src_tokens.copy()

        if not lines:
            return tgt_tokens, edits

        def parse_line(m2_line: str) -> Tuple:
            """Parse a single M2 format edit line into its components.

            This function splits an M2 format line into its constituent parts:
            source indices, error type, correction tokens, and target index.

            Args:
                m2_line: A string in M2 format representing an edit annotation

            Returns:
                Tuple containing:
                - src_beg_idx: Beginning index in source tokens
                - src_end_idx: Ending index in source tokens
                - error_type: Type of grammatical error
                - edit_src_tokens: Source tokens affected by this edit
                - edit_tgt_tokens: Target tokens that replace the source tokens
                - target_index: Index of the target sentence this edit belongs to

            Raises:
                ValueError: If the M2 line format is invalid
            """
            # Split the line by the delimiter, handling the special case for the middle part
            elements = m2_line.split(DELIMITER_M2, 2)
            elements = elements[:2] + elements[-1].rsplit(DELIMITER_M2, 3)

            # Validate that we have the expected number of elements
            if len(elements) != 6:
                raise ValueError(f"Error line: {m2_line}")

            # Extract source indices from the first element (after removing the 'A' marker)
            src_beg_idx, src_end_idx = map(int, elements[0].split()[1:])
            # Get the affected source tokens based on the indices
            edit_src_tokens = src_tokens[src_beg_idx:src_end_idx].copy()
            # Extract target tokens from the third element
            edit_tgt_tokens = elements[2].strip().split()
            # Extract error type and target index
            error_type, target_index = elements[1], int(elements[-1])

            # Handle special case: if target token is a placeholder for "no correction"
            if len(edit_tgt_tokens) and edit_tgt_tokens[0] in EDIT_NONE_CORRECTION:
                edit_tgt_tokens = []

            # Validate error type consistency with indices
            if error_type in EDIT_NONE_TYPE and not (src_beg_idx == src_end_idx == -1):
                raise ValueError(f"Error line: {line}")
            return src_beg_idx, src_end_idx, error_type, edit_src_tokens, edit_tgt_tokens, target_index

        for line in lines:
            src_beg_idx, src_end_idx, error_type, edit_src_tokens, edit_tgt_tokens, target_index = parse_line(line)

            # Skip unchanged sample
            if src_beg_idx == src_end_idx == -1:
                continue

            tgt_beg_idx = src_beg_idx + tgt_offset
            tgt_end_idx = tgt_beg_idx + len(edit_tgt_tokens)
            tgt_tokens[tgt_beg_idx : src_end_idx + tgt_offset] = edit_tgt_tokens
            tgt_offset += len(edit_tgt_tokens) - len(edit_src_tokens)

            edit = Edit(
                src_interval=[src_beg_idx, src_end_idx],
                tgt_interval=[tgt_beg_idx, tgt_end_idx],
                src_tokens=edit_src_tokens,
                tgt_tokens=edit_tgt_tokens,
                tgt_index=target_index,
                types=[error_type],
            )
            edits.append(edit)
            LOGGER.debug(f"Build Edit: {edit}")

            # Sanity Check
            if tgt_tokens[tgt_beg_idx:tgt_end_idx] != edit_tgt_tokens:
                raise ValueError(
                    f"Error Parsing in {src_tokens}:\n"
                    f"Parsed: {' '.join(tgt_tokens[tgt_beg_idx:tgt_end_idx])}\n"
                    f"Origin: {' '.join(edit_tgt_tokens)}"
                )
        return tgt_tokens, edits
