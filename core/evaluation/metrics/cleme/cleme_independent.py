import copy

from core.data.objects import Sample
from core.utils import get_logger

from ...schema import BaseChunkMetricResult, SampleMetricResult
from .cleme_base import CLEME
from .cleme_utils import all_correct, any_correct

LOGGER = get_logger(__name__)


class IndependentCLEME(CLEME):
    """IndependentCLEME implements a position-independent evaluation strategy for grammatical error correction.

    This approach evaluates each hypothesis chunk by considering all corresponding reference
    chunks across multiple references at the same position. If the hypothesis chunk matches
    any one of the reference chunks, it is considered a correct correction.
    """

    def evaluate_sample_correction(
        self, sample_hyp: Sample, sample_ref: Sample, in_place: bool = False
    ) -> SampleMetricResult:
        """Calculate TP, FP, FN, and TN counts for grammatical error correction using a
        position-independent matching strategy.

        For each hypothesis chunk, all reference chunks at the same position are gathered.
        The hypothesis chunk is then compared with these reference chunks:
        - If the hypothesis chunk is marked (i.e., chunk.types is non-empty) and it appears in the
            collection of reference chunks, it is counted as TP.
        - If the hypothesis chunk is marked but does not match any reference chunk, it is counted
            as FP. Furthermore, it is subdivided into necessary (fp_ne) or unnecessary (fp_un) FP
            based on the reference chunks.
        - If the hypothesis chunk is not marked yet any reference chunk is marked, it is a FN.
        - Otherwise, it is counted as TN.

        Finally, the result is wrapped into BaseChunkMetricResult objects, one for each reference
        version.

        Args:
            sample_hyp (Sample): The hypothesis sample.
            sample_ref (Sample): The reference sample with one or more references.
            in_place (bool, optional): If True, result lists are not deep-copied. Defaults to False.

        Returns:
            SampleMetricResult: An object that encapsulates evaluation results for all references.
        """
        tp_chunks, fp_chunks, fn_chunks, tn_chunks = [], [], [], []
        fp_ne_chunks, fp_un_chunks = [], []

        # Extract hypothesis chunks sequence (assumed in sample_hyp.chunks[0][0])
        hyp_chunks = sample_hyp.chunks[0][0]

        # Evaluate each hypothesis chunk based on all references at the same position.
        for chunk_idx, hyp_chunk in enumerate(hyp_chunks):
            # Retrieve all reference chunks at the current index across all references.
            ref_chunks_at_idx = [x[chunk_idx] for x in sample_ref.chunks[0]]

            if hyp_chunk.types:
                # If hypothesis is marked, check if any reference chunk exactly matches.
                if hyp_chunk in ref_chunks_at_idx:
                    tp_chunks.append(hyp_chunk)
                else:
                    fp_chunks.append(hyp_chunk)
                    # Classify FP as necessary or unnecessary based on if any reference chunk is marked.
                    if any_correct(ref_chunks_at_idx):
                        fp_ne_chunks.append(hyp_chunk)
                    else:
                        fp_un_chunks.append(hyp_chunk)
            else:
                # If hypothesis is not marked, determine if the reference expects a correction.
                if all_correct(ref_chunks_at_idx):
                    fn_chunks.append(hyp_chunk)
                else:
                    tn_chunks.append(hyp_chunk)

        # Create results for each reference version.
        ref_results = []
        for _ in sample_ref.target:
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
        return SampleMetricResult(ref_results=ref_results)

    def evaluate_sample_detection(self, sample_hyp: Sample, sample_ref: Sample) -> SampleMetricResult:
        """Evaluate error detection performance using a position-independent strategy.

        For each hypothesis chunk (located in sample_hyp.chunks[0][0]), the method gathers all
        reference chunks at the same position across all reference versions. If any reference chunk
        is marked (i.e., chunk.types is non-empty), then the real error status for that position is True.
        It then compares with the hypothesis detection status:
        - If both hypothesis and reference indicate an error, count as TP.
        - If hypothesis indicates an error but reference does not, count as FP.
        - If hypothesis does not indicate an error while references indicate one, count as FN.
        - Otherwise, count as TN.

        Finally, a BaseChunkMetricResult is created with these statistics. This result is then duplicated
        for every reference in the sample, and all these results are wrapped in a SampleMetricResult.

        Args:
            sample_hyp (Sample): The hypothesis sample containing detected error chunks.
            sample_ref (Sample): The reference sample containing the ground truth errors.

        Returns:
            SampleMetricResult: An object containing detection evaluation metrics replicated for each reference.
        """
        # Extract hypothesis chunks sequence
        hyp_chunks = sample_hyp.chunks[0][0]

        tp_chunks, fp_chunks, fn_chunks, tn_chunks = [], [], [], []
        # Evaluate each chunk irrespective of the reference order.
        for chunk_idx, hyp_chunk in enumerate(hyp_chunks):
            # Check if hypothesis marks an error at this position.
            pred_error = bool(hyp_chunk.types)
            # Gather all reference chunks at the current index.
            ref_chunks_at_idx = [ref_version[chunk_idx] for ref_version in sample_ref.chunks[0]]
            # Determine the any error status: True if any reference chunk is changed.
            any_error = any(bool(ref_chunk.types) for ref_chunk in ref_chunks_at_idx)
            # Determine the all error status: True if all reference chunks are changed.
            all_error = all(bool(ref_chunk.types) for ref_chunk in ref_chunks_at_idx)

            if pred_error and any_error:
                tp_chunks.append(hyp_chunk)
            elif pred_error and not any_error:
                fp_chunks.append(hyp_chunk)
            elif (not pred_error) and all_error:
                fn_chunks.append(hyp_chunk)
            else:
                tn_chunks.append(hyp_chunk)

        # Replicate the same result for all references.
        ref_results = []
        for _ in sample_ref.target:
            result = BaseChunkMetricResult(
                tp_chunks=copy.deepcopy(tp_chunks),
                fp_chunks=copy.deepcopy(fp_chunks),
                fn_chunks=copy.deepcopy(fn_chunks),
                tn_chunks=copy.deepcopy(tn_chunks),
            )
            ref_results.append(result)
        return SampleMetricResult(ref_results=ref_results)
