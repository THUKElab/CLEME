import copy
import logging
from typing import List

from core.data.objects import Sample
from core.utils import get_logger

from ...schema import BaseChunkMetricResult, SampleMetricResult
from .cleme_base import CLEME
from .cleme_utils import chunk_list_to_text

LOGGER = get_logger(__name__, level=logging.INFO)


class DependentCLEME(CLEME):
    """DependentCLEME implements a dependent evaluation strategy for grammatical error correction.

    In this approach, the hypothesis chunks are compared sequentially with the corresponding
    reference chunks at the same position. The evaluation is position-dependent, meaning that
    each hypothesis chunk is matched only with the reference chunk from the same index.
    """

    def evaluate_sample_correction(
        self, sample_hyp: Sample, sample_ref: Sample, in_place: bool = False
    ) -> SampleMetricResult:
        """Calculate true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN)
        for grammatical error correction by performing a position-dependent matching.

        For each reference version (found in sample_ref.chunks[0]), iterate through each chunk in the
        hypothesis (located at sample_hyp.chunks[0][0]). Then, for each position:
        - If the hypothesis chunk is marked (chunk.types is non-empty) and matches exactly the corresponding
            reference chunk, it is counted as TP.
        - If the hypothesis chunk is marked but does not match the reference, it is counted as FP. Additionally,
            it distinguishes between necessary (fp_ne) and unnecessary (fp_un) false positives based on whether
            the reference chunk is marked.
        - If the hypothesis chunk is not marked while the reference chunk has a mark, it is a FN.
        - If neither is marked, it is a TN.

        The result statistics for each reference version are wrapped inside BaseChunkMetricResult objects.
        Optionally, the chunks can be modified in place or deep-copied based on the 'in_place' parameter.

        Args:
            sample_hyp (Sample): The hypothesis sample to evaluate.
            sample_ref (Sample): The reference sample containing one or more references.
            in_place (bool, optional): If True, modifications are done in place; otherwise, deep copies are used.
                Defaults to False.

        Returns:
            SampleMetricResult: An object containing a list of BaseChunkMetricResult for each reference version.
        """

        ref_results: List[BaseChunkMetricResult] = []
        # Iterate over each reference version (each list of chunks is in sample_ref.chunks[0])
        for ref_chunks in sample_ref.chunks[0]:
            # Initialize lists for different evaluation categories.
            tp_chunks, fp_chunks, fn_chunks, tn_chunks = [], [], [], []
            fp_ne_chunks, fp_un_chunks = [], []

            # Evaluate each hypothesis chunk against the corresponding reference chunk.
            for chunk_index, hyp_chunk in enumerate(sample_hyp.chunks[0][0]):
                # If the hypothesis chunk is marked (indicating an error correction or detection)
                if hyp_chunk.types:
                    # If the hypothesis chunk exactly matches the reference chunk at the same position, count as TP.
                    if hyp_chunk == ref_chunks[chunk_index]:
                        tp_chunks.append(hyp_chunk)
                    else:
                        # Otherwise, count as FP and further classify as necessary or unnecessary FP.
                        fp_chunks.append(hyp_chunk)
                        if ref_chunks[chunk_index].types:
                            # Reference indicates that the error should be corrected: necessary FP.
                            fp_ne_chunks.append(hyp_chunk)
                        else:
                            # Reference indicates no error at this position: unnecessary FP.
                            fp_un_chunks.append(hyp_chunk)
                else:
                    # If the hypothesis chunk is not marked, check if it missed a correction that is present in the reference.
                    if hyp_chunk != ref_chunks[chunk_index]:
                        fn_chunks.append(hyp_chunk)
                    else:
                        tn_chunks.append(hyp_chunk)

            # Assemble the evaluation result using either in-place data or deep copies.
            if in_place:
                ref_result = BaseChunkMetricResult(
                    tp_chunks=tp_chunks,
                    fp_chunks=fp_chunks,
                    fn_chunks=fn_chunks,
                    tn_chunks=tn_chunks,
                    fp_ne_chunks=fp_ne_chunks,
                    fp_un_chunks=fp_un_chunks,
                )
            else:
                ref_result = BaseChunkMetricResult(
                    tp_chunks=copy.deepcopy(tp_chunks),
                    fp_chunks=copy.deepcopy(fp_chunks),
                    fn_chunks=copy.deepcopy(fn_chunks),
                    tn_chunks=copy.deepcopy(tn_chunks),
                    fp_ne_chunks=copy.deepcopy(fp_ne_chunks),
                    fp_un_chunks=copy.deepcopy(fp_un_chunks),
                )
            ref_results.append(ref_result)

            # Convert the chunk list into textual form for debugging output.
            src, ref = chunk_list_to_text(ref_chunks)
            LOGGER.debug(f"SRC: {src}")
            LOGGER.debug(f"REF: {ref}")
            LOGGER.debug(f"tp={len(tp_chunks)}, fp={len(fp_chunks)}, fn={len(fn_chunks)}, tn={len(tn_chunks)}")

        return SampleMetricResult(ref_results=ref_results)

    def evaluate_sample_detection(self, sample_hyp: Sample, sample_ref: Sample) -> SampleMetricResult:
        """Evaluate error detection performance in a position-dependent manner.

        For each reference, the function compares hypothesis chunks to reference chunks at the same position
        to determine detection performance. Specifically, for each position:
        - If both hypothesis and reference indicate an error (chunk.types is non-empty), it is TP.
        - If the hypothesis indicates an error but the reference does not, it is FP.
        - If the hypothesis does not indicate an error while the reference does, it is FN.
        - Otherwise, it is TN.

        Note: In detection evaluation, necessary and unnecessary false positives are not distinguished, so fp_ne
        and fp_un are set empty.

        Args:
            sample_hyp (Sample): The hypothesis sample containing detected error chunks.
            sample_ref (Sample): The reference sample containing the ground truth for errors.

        Returns:
            SampleMetricResult: An object containing evaluation statistics (TP, FP, FN, TN) for each reference version.
        """
        ref_results: List[BaseChunkMetricResult] = []
        # Extract hypothesis chunks from the first block (assumed to be stored at sample_hyp.chunks[0][0])
        hyp_chunks = sample_hyp.chunks[0][0]

        # Evaluate performance against each reference version.
        for ref_chunks in sample_ref.chunks[0]:
            tp_chunks, fp_chunks, fn_chunks, tn_chunks = [], [], [], []

            # Compare hypothesis and reference chunks one-by-one by index.
            for chunk_index, hyp_chunk in enumerate(hyp_chunks):
                # Determine whether the hypothesis has marked an error.
                pred_error = bool(hyp_chunk.types)
                # Determine whether the corresponding reference chunk is marked as an error.
                ref_error = bool(ref_chunks[chunk_index].types)
                if pred_error and ref_error:
                    tp_chunks.append(hyp_chunk)
                elif pred_error and not ref_error:
                    fp_chunks.append(hyp_chunk)
                elif (not pred_error) and ref_error:
                    fn_chunks.append(hyp_chunk)
                else:
                    tn_chunks.append(hyp_chunk)

            # Create a result object for this reference version.
            ref_result = BaseChunkMetricResult(
                tp_chunks=copy.deepcopy(tp_chunks),
                fp_chunks=copy.deepcopy(fp_chunks),
                fn_chunks=copy.deepcopy(fn_chunks),
                tn_chunks=copy.deepcopy(tn_chunks),
            )
            ref_results.append(ref_result)
        return SampleMetricResult(ref_results=ref_results)
