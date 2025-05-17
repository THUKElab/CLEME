import sys
from collections import defaultdict
from typing import Any, List, TextIO, Tuple

import numpy as np
from pydantic import Field
from tabulate import tabulate

from core.data.objects import Dataset

from ..schema import (
    BaseChunkMetricResult,
    BaseEditMetricResult,
    EditScorerResult,
    OverallScorerResult,
    SampleMetricResult,
)
from .scorer_base import BaseScorer
from .scorer_utils import compute_acc, compute_prf, gt_numbers


class PRFEditScorer(BaseScorer):
    """Traditional edit-based scorer for computing precision, recall, F-measure, and accuracy.

    This scorer works by computing both unweighted and weighted scores for edits or chunks.
    The weighted scores utilize the importance (or weight) attribute on individual edits/chunks.

    Attributes:
        factor_beta (float): Trade-off parameter between precision and recall.
        table_print (bool): Flag to determine if results are printed in a table.
    """

    factor_beta: float = Field(default=0.5, description="Trade-off factor of Precision and Recall")
    table_print: bool = Field(default=True)

    def score(
        self, dataset_hyp: Dataset, dataset_ref: Dataset, metric_results: List[SampleMetricResult]
    ) -> OverallScorerResult:
        """Compute overall PRF scores at both corpus and sentence levels.

        For each sample, this method computes both unweighted and weighted edit scores,
        then aggregates the results at corpus-level and sentence-level.

        Args:
            dataset_hyp (Dataset): The hypothesis dataset.
            dataset_ref (Dataset): The reference dataset.
            metric_results (List[SampleMetricResult]): List of sample metric results each containing
                reference-based scores.

        Returns:
            OverallScorerResult: The overall scores including both unweighted and weighted measurements.
        """
        dataset_scorer_results, dataset_scorer_results_weighted = [], []

        # Iterate over sample-level metric results.
        for sample_metric_result in metric_results:
            sample_scorer_result, sample_scorer_result_weighted = self.score_sample(sample_metric_result)
            dataset_scorer_results.append(sample_scorer_result)
            dataset_scorer_results_weighted.append(sample_scorer_result_weighted)

        # Aggregate scores at the corpus‐level
        score_corpus = self.score_corpus(dataset_scorer_results)
        score_corpus_weighted = self.score_corpus(dataset_scorer_results_weighted)

        # Aggregate scores at the sentence-level
        score_sentence = self.score_sentence(dataset_scorer_results)
        score_sentence_weighted = self.score_sentence(dataset_scorer_results_weighted)

        result = OverallScorerResult(
            num_sample=len(metric_results),
            scores={
                "prf_corpus_unweighted": score_corpus,
                "prf_corpus_weighted": score_corpus_weighted,
                "prf_sentence_unweighted": score_sentence,
                "prf_sentence_weighted": score_sentence_weighted,
            },
        )
        if self.table_print:
            self.print_result_table(result, num_sample=result.num_sample)
        return result

    def score_sample(self, metric_result: SampleMetricResult) -> Tuple[List[EditScorerResult], List[EditScorerResult]]:
        """Compute sample-level scores for edits or chunks.

        For each reference result (which must be either a BaseEditMetricResult or BaseChunkMetricResult),
        the method computes the counts of TP, FP, FN and TN. Then, precision, recall, F-measure, and accuracy
        are computed for both the raw counts and weighted counts.

        Args:
            metric_result (SampleMetricResult): The metric results for a sample containing one or more reference results.

        Returns:
            Tuple[List[EditScorerResult], List[EditScorerResult]]: A tuple containing:
            - A list of unweighted edit scoring results.
            - A list of weighted edit scoring results.
        """

        results, results_weighted = [], []
        for ref_result in metric_result.ref_results:
            # Depending on whether the reference result is edit-based or chunk-based, count the items.
            if isinstance(ref_result, BaseEditMetricResult):
                tp = len(ref_result.tp_edits)
                fp = len(ref_result.fp_edits)
                fn = len(ref_result.fn_edits)
                tn = len(ref_result.tn_edits)
            elif isinstance(ref_result, BaseChunkMetricResult):
                tp = len(ref_result.tp_chunks)
                fp = len(ref_result.fp_chunks)
                fn = len(ref_result.fn_chunks)
                tn = len(ref_result.tn_chunks)
            else:
                raise ValueError(f"Unknown ref_result: {type(ref_result)}")

            # Calculate precision, recall, F-measure using provided beta factor
            p, r, f = compute_prf(tp, fp, fn, beta=self.factor_beta)
            # Compute accuracy
            acc = compute_acc(tp, fp, fn, tn)
            # Create result for unweighted computation
            result = EditScorerResult(tp=tp, fp=fp, fn=fn, tn=tn, p=p, r=r, f=f, acc=acc)
            results.append(result)

            # Process weighted counts (if weight is not provided, default to 1.0)
            if isinstance(ref_result, BaseEditMetricResult):
                tp_weight = sum([x.weight or 1.0 for x in ref_result.tp_edits])
                fp_weight = sum([x.weight or 1.0 for x in ref_result.fp_edits])
                fn_weight = sum([x.weight or 1.0 for x in ref_result.fn_edits])
                tn_weight = sum([x.weight or 1.0 for x in ref_result.tn_edits])
            elif isinstance(ref_result, BaseChunkMetricResult):
                tp_weight = sum([x.weight or 1.0 for x in ref_result.tp_chunks])
                fp_weight = sum([x.weight or 1.0 for x in ref_result.fp_chunks])
                fn_weight = sum([x.weight or 1.0 for x in ref_result.fn_chunks])
                tn_weight = sum([x.weight or 1.0 for x in ref_result.tn_chunks])
            else:
                raise ValueError(f"Unknown ref_result: {type(ref_result)}")
            # Compute weighted precision, recall, F-measure and accuracy
            p, r, f = compute_prf(tp_weight, fp_weight, fn_weight, beta=self.factor_beta)
            acc = compute_acc(tp_weight, fp_weight, fn_weight, tn_weight)
            result_weighted = EditScorerResult(
                tp=tp_weight, fp=fp_weight, fn=fn_weight, tn=tn_weight, p=p, r=r, f=f, acc=acc
            )
            results_weighted.append(result_weighted)
        return results, results_weighted

    def score_corpus(self, dataset_scorer_results: List[List[EditScorerResult]]) -> EditScorerResult:
        """Aggregate sample-level scores to compute a corpus-level score.

        The aggregation is performed by examining the reference results for each sample and choosing
        the one with the highest F-measure (with additional tie-breaking on TP, FP, FN, TN in order).
        The cumulative counts are then used to compute the final precision, recall, F-measure and accuracy.

        Args:
            dataset_scorer_results (List[List[EditScorerResult]]): A list where each element is a list
                of EditScorerResults for a sample.

        Returns:
            EditScorerResult: The aggregated corpus-level scoring result.
        """
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        for sample_scorer_result in dataset_scorer_results:
            # Initialize best metric values for current sample
            best_f, best_tp, best_fp, best_fn, best_tn = -1.0, 0, 0, 0, 0

            # Search for the best reference result in the current sample
            for ref_scorer_result in sample_scorer_result:
                _tp = ref_scorer_result.tp
                _fp = ref_scorer_result.fp
                _fn = ref_scorer_result.fn
                _tn = ref_scorer_result.tn
                # Compute F-measure using cumulative counts (so far plus current)
                _p, _r, _f = compute_prf(tp=total_tp + _tp, fp=total_fp + _fp, fn=total_fn + _fn, beta=self.factor_beta)

                # Use gt_numbers to compare tuples of metrics
                if gt_numbers([_f, _tp, -_fp, -_fn, _tn], [best_f, best_tp, -best_fp, -best_fn, best_tn]):
                    best_f, best_tp, best_fp, best_fn, best_tn = _f, _tp, _fp, _fn, _tn

            # Accumulate the best counts for the sample
            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
            total_tn += best_tn

        # Compute final aggregated precision, recall, F-measure and accuracy
        final_p, final_r, final_f = compute_prf(total_tp, total_fp, total_fn, beta=self.factor_beta)
        final_acc = compute_acc(total_tp, total_fp, total_fn, total_tn)
        return EditScorerResult(
            tp=total_tp, fp=total_fp, fn=total_fn, tn=total_tn, p=final_p, r=final_r, f=final_f, acc=final_acc
        )

    def score_sentence(self, dataset_scorer_results: List[List[EditScorerResult]]) -> EditScorerResult:
        """Compute sentence-level scores by aggregating per-sample results.

        The sentence-level score is computed by averaging the best scoring reference result from
        each sample.

        Args:
            dataset_scorer_results (List[List[EditScorerResult]]): List of scoring results per sample.

        Returns:
            EditScorerResult: The aggregated sentence-level scoring result.
        """

        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_f, total_p, total_r, total_acc = [], [], [], []
        for sample_scorer_result in dataset_scorer_results:
            best_f, best_p, best_r, best_acc = -1.0, -1.0, -1.0, -1.0
            best_tp, best_fp, best_fn, best_tn = 0, 0, 0, 0

            # Find the best result among the reference results for the current sample
            for ref_scorer_result in sample_scorer_result:
                _tp = ref_scorer_result.tp
                _fp = ref_scorer_result.fp
                _fn = ref_scorer_result.fn
                _tn = ref_scorer_result.tn
                _p, _r, _f = compute_prf(_tp, _fp, _fn)
                _acc = compute_acc(_tp, _fp, _fn, _tn)

                if gt_numbers([_f, _p, _r, _acc], [best_f, best_p, best_r, best_acc]):
                    best_f, best_p, best_r, best_acc = _f, _p, _r, _acc
                    best_tp = _tp
                    best_fp = _fp
                    best_fn = _fn
                    best_tn = _tn

            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
            total_tn += best_tn
            total_f.append(best_f)
            total_p.append(best_p)
            total_r.append(best_r)
            total_acc.append(best_acc)

        # Return average scores across sentences
        return EditScorerResult(
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            tn=total_tn,
            p=np.average(total_p),
            r=np.average(total_r),
            f=np.average(total_f),
            acc=np.average(total_acc),
        )

    def print_result_table(self, result: OverallScorerResult, sout: TextIO = sys.stdout, **kwargs: Any) -> None:
        """Print the scoring results in a formatted table.

        Args:
            result (OverallScorerResult): The overall scoring result containing multiple metric scores.
            sout (TextIO, optional): The stream where output is sent (defaults to sys.stdout).
        """
        tabular_data = defaultdict(list)
        for key, score_result in result.scores.items():
            assert isinstance(score_result, EditScorerResult)
            tabular_data["metric"].append(key)
            tabular_data["F"].append(score_result.f)
            tabular_data["P"].append(score_result.p)
            tabular_data["R"].append(score_result.r)
            tabular_data["ACC"].append(score_result.acc)

            tabular_data["TP"].append(score_result.tp)
            tabular_data["FP"].append(score_result.fp)
            tabular_data["FN"].append(score_result.fn)
            tabular_data["TN"].append(score_result.tn)

        table = tabulate(
            tabular_data, tablefmt="fancy_grid", headers="keys", floatfmt=".4f", missingval="N/A", numalign="left"
        )
        sout.write("\n" + table + "\n")
        for k, v in kwargs.items():
            sout.write(f"{k}: {v}\n")
        sout.write("\n")
