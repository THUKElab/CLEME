from typing import Any, List

from core.data.objects import Sample

from ..schema import BaseChunkMetricResult, BaseEditMetricResult, SampleMetricResult


class BaseWeigher:
    """Base class for acquiring edit weights.

    The edit weigher is designed to adjust the weights associated with edits (or chunks)
    in order to improve correlations with human judgments. By default, this class sets all
    weights to 1.0. Subclasses should override necessary methods to implement more
    sophisticated weighting strategies.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def signature(self) -> str:
        """Returns a signature for the tokenizer."""
        return "none"

    def setup(self, **kwargs: Any) -> None:
        """Perform any necessary setup for the weigher.

        This function is intended to be overridden if the weigher requires initialization
        (e.g., loading resources or computing statistics).
        """
        pass

    def __call__(
        self, sample_hyp: Sample, sample_ref: Sample, metric_result: SampleMetricResult, **kwargs: Any
    ) -> None:
        """Apply the weigher to assign weights to all edits/chunks in the metric result.

        This method iterates through each reference result in the given SampleMetricResult.
        Depending on whether the result is a BaseEditMetricResult or BaseChunkMetricResult,
        it sets the weight of each edit/chunk to 1.0 by default.

        Args:
            sample_hyp (Sample): Hypothesis sample.
            sample_ref (Sample): Reference sample.
            metric_result (SampleMetricResult): The result containing edit/chunk metrics.

        Raises:
            ValueError: If the reference result is not recognized.
        """
        for ref_result in metric_result.ref_results:
            # Set weight for true positives, false positives, false negatives, true negatives, etc.

            if isinstance(ref_result, BaseEditMetricResult):
                for tp_edit in ref_result.tp_edits:
                    tp_edit.weight = 1.0
                for fp_edit in ref_result.fp_edits:
                    fp_edit.weight = 1.0
                for fn_edit in ref_result.fn_edits:
                    fn_edit.weight = 1.0
                for tn_edit in ref_result.tn_edits:
                    tn_edit.weight = 1.0
                for fp_ne_edits in ref_result.fp_ne_edits:
                    fp_ne_edits.weight = 1.0
                for fp_un_edits in ref_result.fp_un_edits:
                    fp_un_edits.weight = 1.0
            elif isinstance(ref_result, BaseChunkMetricResult):
                for tp_chunk in ref_result.tp_chunks:
                    tp_chunk.weight = 1.0
                for fp_chunk in ref_result.fp_chunks:
                    fp_chunk.weight = 1.0
                for fn_chunk in ref_result.fn_chunks:
                    fn_chunk.weight = 1.0
                for tn_chunk in ref_result.tn_chunks:
                    tn_chunk.weight = 1.0
                for fp_ne_chunks in ref_result.fp_ne_chunks:
                    fp_ne_chunks.weight = 1.0
                for fp_un_chunks in ref_result.fp_un_chunks:
                    fp_un_chunks.weight = 1.0
            else:
                raise ValueError()

    def get_weights_batch(
        self, samples_hyp: List[Sample], samples_ref: List[Sample], metric_results: List[SampleMetricResult]
    ) -> None:
        """Apply the weigher in batch for multiple samples.

        Iterates through the provided samples and corresponding metric results, applying
        the __call__ method to each sample.

        Args:
            samples_hyp (List[Sample]): List of hypothesis samples.
            samples_ref (List[Sample]): List of reference samples.
            metric_results (List[SampleMetricResult]): List of associated metric results.
        """
        for sample_hyp, sample_ref, metric_result in zip(samples_hyp, samples_ref, metric_results):
            self.__call__(sample_hyp=sample_hyp, sample_ref=sample_ref, metric_result=metric_result)
