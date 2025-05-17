import sys
from collections import defaultdict
from typing import Any, List, Optional, TextIO, Tuple

import numpy as np
from pydantic import Field
from tabulate import tabulate

from core.data.objects import Dataset, Sample
from core.utils import get_logger

from ..schema import (
    BaseChunkMetricResult,
    EditScorerResult,
    HEUOEditScorerResult,
    OverallScorerResult,
    SampleMetricResult,
)
from .scorer_base import BaseScorer
from .scorer_utils import gt_numbers

LOGGER = get_logger(__name__)


class HEUOEditScorer(BaseScorer):
    """Heuristic-based edit scorer that computes interpretable scores using a weighted linear combination.

    This scorer calculates multiple indicators:
      - hit: The ratio of true positive edits.
      - err: The error ratio for false positive (necessary) edits.
      - und: The under-generation ratio for false negatives.
      - ove: The over-generation ratio for unnecessary false positives.
    A comprehensive (com) indicator is then computed as a linear weighted sum of these metrics.

    Attributes:
        factor_hit (float): Weighting factor for the 'hit' component.
        factor_err (float): Weighting factor for the 'err' component.
        factor_und (float): Weighting factor for the 'und' component.
        factor_ove (float): Weighting factor for the 'ove' component.
        table_print (bool): Flag indicating whether to print the results in a table.
        max_value (float): A maximum clip value used in weighting (if required).
    """

    factor_hit: float = Field(default=0.50)
    factor_err: float = Field(default=0.40)
    factor_und: float = Field(default=0.05)
    factor_ove: float = Field(default=0.05)
    table_print: bool = Field(default=True)
    max_value: Optional[float] = Field(default=None)

    def __init__(
        self,
        factor_hit: float = 0.50,
        factor_err: float = 0.40,
        factor_und: float = 0.05,
        factor_ove: float = 0.05,
        max_value: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """Initialize the HEUOEditScorer.

        The sum of factor_hit, factor_err, factor_und, and factor_ove must equal 1.0.

        Args:
            factor_hit (float): Weight for true positive ratio.
            factor_err (float): Weight for error component.
            factor_und (float): Weight for under-generation.
            factor_ove (float): Weight for over-generation.
            max_value (float): Maximum value threshold for weighting.

        Raises:
            ValueError: If the sum of factors is not equal to 1.0.
        """

        factor_combined = factor_hit + factor_err + factor_und + factor_ove
        if abs(factor_combined - 1.0) > 1e-5:
            raise ValueError("Invalid factors: combination of factors must be 1.0")
        super().__init__(
            factor_hit=factor_hit,
            factor_err=factor_err,
            factor_und=factor_und,
            factor_ove=factor_ove,
            max_value=max_value,
            **kwargs,
        )

    def compute_comprehensive_indicator(self, tp: float, fn: float, fp_ne: float, fp_un: float) -> Tuple:
        """Compute the comprehensive indicator and related metrics using a weighted linear combination.

        The method calculates four indicators:
            - hit: hit-correction ratio of true positives over the count of necessary edits.
            - err: wrong-correction ratio of false positives among necessary edits.
            - und: under-correction ratio of false negatives among necessary edits.
            - ove: over-correction ratio of unnecessary false positives.
        Finally, a comprehensive score ('com') is computed by applying the linear weights.

        Args:
            tp (float): Number (or weighted sum) of true positive chunks.
            fn (float): Number (or weighted sum) of false negative chunks.
            fp_ne (float): Number (or weighted sum) of false positive chunks considered "necessary".
            fp_un (float): Number (or weighted sum) of false positive chunks considered "unnecessary".

        Returns:
            Tuple: A tuple in the format (hit, err, und, ove, com) where each element is a float.
        """
        # Calculate the total count of necessary edits.
        ne = tp + fp_ne + fn
        # Total false positives is sum of necessary and unnecessary FP.
        fp = fp_ne + fp_un

        # Calculate hit ratio if ne is non-zero, otherwise set to 0.0.
        hit = float(tp) / ne if ne else 0.0
        # Calculate error ratio among necessary examples if ne is non-zero.
        err = float(fp_ne) / ne if ne else 0.0
        # Calculate under-generation ratio if ne is non-zero.
        und = float(fn) / ne if ne else 0.0
        # Calculate over-generation ratio if the sum (tp + fp) is non-zero.
        ove = float(fp_un) / (tp + fp) if (tp + fp) else 0.0

        # Compute the comprehensive score as a weighted sum of the indicators.
        com = (
            self.factor_hit * hit
            + self.factor_err * (1 - err)
            + self.factor_und * (1 - und)
            + self.factor_ove * (1 - ove)
        )
        # Note: Although rounding may cause slight differences in the result,
        # the decision was made to return the computed float values directly.
        return hit, err, und, ove, com

    def score(
        self, dataset_hyp: Dataset, dataset_ref: Dataset, metric_results: List[SampleMetricResult], adapt: bool = False
    ) -> OverallScorerResult:
        """Score a set of hypothesis and reference datasets using HEUO metrics.

        This method computes both corpus level and sentence level HEUO scores. If the adapt flag
        is set to True, the weighting factors are adapted for each scoring stage.

        Args:
            dataset_hyp (Dataset): The hypothesis dataset containing predicted samples.
            dataset_ref (Dataset): The reference dataset containing ground truth samples.
            metric_results (List[SampleMetricResult]): List of metric results for each sample.
            adapt (bool): Flag to indicate if the weighting factors should be adapted (default is False).

        Returns:
            OverallScorerResult: An object containing the overall scoring results, including separate
                scores for corpus and sentence levels, for both unweighted and weighted evaluations.
        """
        LOGGER.info(self)
        dataset_scorer_results, dataset_scorer_results_weighted = [], []

        # Process each sample independently and accumulate scoring results.
        for sample_idx, sample_metric_result in enumerate(metric_results):
            sample_scorer_result, sample_scorer_result_weighted = self.score_sample(
                sample_hyp=dataset_hyp[sample_idx],
                sample_ref=dataset_ref[sample_idx],
                metric_result=sample_metric_result,
            )
            dataset_scorer_results.append(sample_scorer_result)
            dataset_scorer_results_weighted.append(sample_scorer_result_weighted)

        # Adjust the weights if adapt flag is set for corpus-level unweighted score.
        if adapt:
            self.factor_hit = 0.60
            self.factor_err = 0.10
            self.factor_und = 0.10
            self.factor_ove = 0.20
        score_corpus = self.score_corpus(dataset_scorer_results)

        # Adjust the weights if adapt flag is set for corpus-level weighted score.
        if adapt:
            self.factor_hit = 0.60
            self.factor_err = 0.30
            self.factor_und = 0.05
            self.factor_ove = 0.05
        score_corpus_weighted = self.score_corpus(dataset_scorer_results_weighted)

        # Adjust the weights if adapt flag is set for sentence-level unweighted score.
        if adapt:
            self.factor_hit = 0.60
            self.factor_err = 0.10
            self.factor_und = 0.05
            self.factor_ove = 0.25
        score_sentence = self.score_sentence(dataset_scorer_results)

        # Adjust the weights if adapt flag is set for sentence-level weighted score.
        if adapt:
            self.factor_hit = 0.15
            self.factor_err = 0.35
            self.factor_und = 0.35
            self.factor_ove = 0.15
        score_sentence_weighted = self.score_sentence(dataset_scorer_results_weighted)

        # Construct OverallScorerResult containing all computed scores.
        result = OverallScorerResult(
            num_sample=len(metric_results),
            scores={
                "heuo_corpus_unweighted": score_corpus,
                "heuo_corpus_weighted": score_corpus_weighted,
                "heuo_sentence_unweighted": score_sentence,
                "heuo_sentence_weighted": score_sentence_weighted,
            },
        )

        # Print the table of results if table_print is enabled.
        if self.table_print:
            self.print_result_table(result, num_sample=result.num_sample)
        return result

    def score_sample(
        self, sample_hyp: Sample, sample_ref: Sample, metric_result: SampleMetricResult
    ) -> Tuple[List[EditScorerResult], List[EditScorerResult]]:
        """Compute sentence-level HEUO scores for a single sample.

        Iterates through each reference result for the sample and computes both unweighted and weighted
        HEUO scores based on chunk metrics. For unweighted metrics, counts are derived by simply taking
        the number of chunks. For weighted metrics, each chunk's weight is summed.

        Args:
            sample_hyp (Sample): The hypothesis sample.
            sample_ref (Sample): The reference sample.
            metric_result (SampleMetricResult): The metric results for the sample.

        Returns:
            Tuple[List[EditScorerResult], List[EditScorerResult]]:
            - A list of unweighted HEUO scoring results.
            - A list of weighted HEUO scoring results.
        """

        results, results_weighted = [], []

        # Iterate over each reference result in the sample.
        for ref_result in metric_result.ref_results:
            if not isinstance(ref_result, BaseChunkMetricResult):
                raise ValueError("Expected instance of BaseChunkMetricResult")

            # Get chunk counts for unweighted computation.
            tp = len(ref_result.tp_chunks)
            fp = len(ref_result.fp_chunks)
            fn = len(ref_result.fn_chunks)
            tn = len(ref_result.tn_chunks)
            fp_ne = len(ref_result.fp_ne_chunks)
            fp_un = len(ref_result.fp_un_chunks)

            # Compute the HEUO indicators and comprehensive score.
            hit, err, und, ove, com = self.compute_comprehensive_indicator(tp=tp, fn=fn, fp_ne=fp_ne, fp_un=fp_un)
            result = HEUOEditScorerResult(
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                fp_ne=fp_ne,
                fp_un=fp_un,
                necessary=tp + fp_ne + fn,
                unnecessary=tp + fp,
                hit=hit,
                err=err,
                und=und,
                ove=ove,
                com=com,
            )
            results.append(result)

            # Compute weighted counts using each chunk's weight.
            tp = sum([x.weight for x in ref_result.tp_chunks])
            fp = sum([x.weight for x in ref_result.fp_chunks])
            fn = sum([x.weight for x in ref_result.fn_chunks])
            tn = sum([x.weight for x in ref_result.tn_chunks])
            fp_ne = sum([x.weight for x in ref_result.fp_ne_chunks])
            fp_un = sum([x.weight for x in ref_result.fp_un_chunks])

            # Compute HEUO indicators for the weighted counts.
            hit, err, und, ove, com = self.compute_comprehensive_indicator(tp=tp, fn=fn, fp_ne=fp_ne, fp_un=fp_un)
            result_weighted = HEUOEditScorerResult(
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                fp_ne=fp_ne,
                fp_un=fp_un,
                necessary=tp + fp_ne + fn,
                unnecessary=fp_un,
                hit=hit,
                err=err,
                und=und,
                ove=ove,
                com=com,
            )
            results_weighted.append(result_weighted)
        return results, results_weighted

    def score_corpus(self, dataset_scorer_results: List[List[HEUOEditScorerResult]]) -> HEUOEditScorerResult:
        """Compute corpus-level HEUO scores by aggregating sample-level results.

        For each sample, the best reference result is chosen (based on the comprehensive score and tie-breakers)
        and its counts are accumulated. After iterating over all samples, the overall HEUO score and its indicators
        are computed from the total accumulated counts.

        Args:
            dataset_scorer_results (List[List[HEUOEditScorerResult]]): A list where each element is a list
                of HEUO scoring results for a sample.

        Returns:
            HEUOEditScorerResult: The aggregated corpus-level HEUO scoring result.
        """
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_fp_ne, total_fp_un = 0, 0

        # Loop over each sample's results.
        for sample_scorer_result in dataset_scorer_results:
            best_com = -1.0
            best_tp, best_fp, best_fn, best_tn = 0, 0, 0, 0
            best_fp_ne, best_fp_un = 0, 0
            for ref_scorer_result in sample_scorer_result:
                _tp = ref_scorer_result.tp
                _fp = ref_scorer_result.fp
                _fn = ref_scorer_result.fn
                _tn = ref_scorer_result.tn
                _fp_ne = ref_scorer_result.fp_ne
                _fp_un = ref_scorer_result.fp_un

                # Compute the cumulative comprehensive score if these counts were added.
                hit, err, und, ove, com = self.compute_comprehensive_indicator(
                    tp=total_tp + _tp, fn=total_fn + _fn, fp_ne=total_fp_ne + _fp_ne, fp_un=total_fp_un + _fp_un
                )

                # Use gt_numbers to determine if this reference result is better than the best so far.
                # TODO: Maybe sorting by HEUO would be better.
                if gt_numbers([com, _tp, -_fp, -_fn, _tn], [best_com, best_tp, -best_fp, -best_fn, best_tn]):
                    best_com = com
                    best_tp = _tp
                    best_fp = _fp
                    best_fn = _fn
                    best_tn = _tn
                    best_fp_ne = _fp_ne
                    best_fp_un = _fp_un

            # Accumulate the best results for the current sample.
            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
            total_tn += best_tn
            total_fp_ne += best_fp_ne
            total_fp_un += best_fp_un

        # Calculate overall indicators from the total accumulated counts.
        hit, err, und, ove, com = self.compute_comprehensive_indicator(
            tp=total_tp, fn=total_fn, fp_ne=total_fp_ne, fp_un=total_fp_un
        )
        return HEUOEditScorerResult(
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            tn=total_tn,
            fp_ne=total_fp_ne,
            fp_un=total_fp_un,
            necessary=total_tp + total_fp_ne + total_fn,
            unnecessary=total_fp_un,
            com=com,
            hit=hit,
            err=err,
            und=und,
            ove=ove,
        )

    def score_sentence(self, dataset_scorer_results: List[List[HEUOEditScorerResult]]) -> HEUOEditScorerResult:
        """Calculate sentence-level HEUO Scores by choosing the best scoring result for each sample.

        For each sample, the best result is chosen based on comprehensive metrics and indicator values.
        The scores are then aggregated and averaged (where appropriate) to produce an overall sentence-level score.

        Args:
            dataset_scorer_results (List[List[HEUOEditScorerResult]]):
                A list where each element is a list of HEUOEditScorerResult for a sample.

        Returns:
            HEUOEditScorerResult: The aggregated sentence-level scoring result.
        """
        total_com, total_hit, total_err, total_und, total_ove = [], [], [], [], []
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_fp_ne, total_fp_un = 0, 0

        # Iterate over each sample's scorer results.
        for sample_scorer_result in dataset_scorer_results:
            if not len(sample_scorer_result):
                continue

            best_com = -1.0
            best_hit, best_err, best_und, best_ove = -1.0, -1.0, -1.0, -1.0
            best_tp, best_fp, best_fn, best_tn = 0, 0, 0, 0
            best_fp_ne, best_fp_un = 0, 0

            # Evaluate each reference result to select the best one for the sample.
            for ref_scorer_result in sample_scorer_result:
                _tp = ref_scorer_result.tp
                _fp = ref_scorer_result.fp
                _fn = ref_scorer_result.fn
                _tn = ref_scorer_result.tn
                _fp_ne = ref_scorer_result.fp_ne
                _fp_un = ref_scorer_result.fp_un
                hit, err, und, ove, com = self.compute_comprehensive_indicator(
                    tp=_tp, fn=_fn, fp_ne=_fp_ne, fp_un=_fp_un
                )

                # Choose the result with the highest (com, hit, etc.) as defined by gt_numbers.
                if gt_numbers([com, hit, -err, -und, -ove], [best_com, best_hit, -best_err, -best_und, -best_ove]):
                    best_com = com
                    best_hit = hit
                    best_err = err
                    best_und = und
                    best_ove = ove
                    best_tp = _tp
                    best_fp = _fp
                    best_fn = _fn
                    best_tn = _tn
                    best_fp_ne = _fp_ne
                    best_fp_un = _fp_un

            # Accumulate the best scoring values across samples.
            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
            total_tn += best_tn
            total_fp_ne += best_fp_ne
            total_fp_un += best_fp_un
            total_com.append(best_com)
            total_hit.append(best_hit)
            total_err.append(best_err)
            total_und.append(best_und)
            total_ove.append(best_ove)

        # Create a HEUOEditScorerResult with aggregated counts and average indicators.
        return HEUOEditScorerResult(
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            tn=total_tn,
            fp_ne=total_fp_ne,
            fp_un=total_fp_un,
            necessary=total_tp + total_fp_ne + total_fn,
            unnecessary=total_fp_un,
            com=np.average(total_com),
            hit=np.average(total_hit),
            err=np.average(total_err),
            und=np.average(total_und),
            ove=np.average(total_ove),
        )

    def print_result_table(self, result: OverallScorerResult, sout: TextIO = sys.stdout, **kwargs: Any) -> None:
        """Visualize the overall scoring results in a formatted table.

        This method uses the tabulate library to print results such as the comprehensive score,
        individual indicators (HIT, ERR, UND, OVE), and chunk counts for each metric.

        Args:
            result (OverallScorerResult): The overall scorer result containing various score metrics.
            sout (TextIO, optional): The output stream to print the table. Defaults to sys.stdout.
            **kwargs: Additional key-value pairs that will be printed below the table.
        """
        tabular_data = defaultdict(list)
        # Populate table data for each metric score.
        for key, score_result in result.scores.items():
            assert isinstance(score_result, HEUOEditScorerResult)
            tabular_data["metric"].append(key)
            tabular_data["COM"].append(score_result.com)
            tabular_data["HIT"].append(score_result.hit)
            tabular_data["ERR"].append(score_result.err)
            tabular_data["UND"].append(score_result.und)
            tabular_data["OVE"].append(score_result.ove)

            tabular_data["TP"].append(score_result.tp)
            tabular_data["FP"].append(score_result.fp)
            tabular_data["FN"].append(score_result.fn)
            tabular_data["TN"].append(score_result.tn)
            tabular_data["FP_NE"].append(score_result.fp_ne)
            tabular_data["FP_UN"].append(score_result.fp_un)
            tabular_data["NE"].append(score_result.necessary)

        # Generate and print the formatted table.
        table = tabulate(
            tabular_data, tablefmt="fancy_grid", headers="keys", floatfmt=".4f", missingval="N/A", numalign="left"
        )
        sout.write("\n" + table + "\n")
        # Print any additional keyword arguments.
        for k, v in kwargs.items():
            sout.write(f"{k}: {v}\n")
        sout.write("\n")
