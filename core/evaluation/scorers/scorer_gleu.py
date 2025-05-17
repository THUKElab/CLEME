import math
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, TextIO

import numpy as np
from pydantic import Field
from tabulate import tabulate

from core.data.objects import Dataset

from ..schema import GLEUScorerResult, OverallScorerResult
from .scorer_base import BaseScorer

# Constants to be used as keys in the metric results dictionary
KEY_HYP_LEN = "hyp_len"
KEY_REF_LEN = "ref_len"
KEY_NGRAMS = "ngrams"


class GLEUScorer(BaseScorer):
    """Scorer for computing the GLEU metric.

    The GLEU scorer calculates both corpus-level and sentence-level GLEU scores. The metric
    is based on the comparison of n-gram precisions between hypothesis and reference texts.

    Attributes:
        order (int): Maximum order of ngrams considered.
        num_iter (int): Number of iterations for sampling.
        smoothing (bool): Whether to perform smoothing.
        table_print (bool): Controls tabular printing of results.
    """

    order: int = Field(default=4, description="Maximum order of ngrams")
    num_iter: int = Field(default=500, description="Number of iterations to run")
    smoothing: bool = Field(default=False, description="Smoothing factor")
    table_print: bool = Field(default=True)

    def score(self, dataset_hyp: Dataset, dataset_ref: Dataset, metric_results: List[List[Dict]]) -> Dict[str, Any]:
        """Compute the overall GLEU score combining corpus-level and sentence-level evaluation.

        This method first computes corpus-level metric followed by sentence-level metrics,
        and then aggregates them into an OverallScorerResult.

        Args:
            dataset_hyp (Dataset): Hypothesis dataset.
            dataset_ref (Dataset): Reference dataset.
            metric_results (List[List[Dict]]): Precomputed metric result dictionaries for each sample
                (each element is a list of dictionaries corresponding to references).

        Returns:
            Dict[str, Any]: The overall scoring result with corpus and sentence GLEU scores.
        """
        # Compute corpus-level GLEU score
        score_corpus = self.score_corpus(metric_results)
        # Compute sentence-level GLEU score
        score_sentence = self.score_sentence(metric_results)

        # Assemble overall result
        result = OverallScorerResult(
            num_sample=len(metric_results),
            scores={"gleu_corpus": score_corpus, "gleu_sentence": score_sentence},
        )

        # Optionally print result as a formatted table
        if self.table_print:
            self.print_result_table(result, num_sample=result.num_sample)
        return result

    def score_corpus(self, metric_results: List[List[Dict]]) -> float:
        """Compute the corpus-level GLEU score.

        For each sample and for a number of iterations, a random reference is selected and
        metrics such as hypothesis length, reference length, and ngram counts are accumulated.
        Finally, the logarithmic precision is computed and returned as the corpus score.

        Args:
            metric_results (List[List[Dict]]): List of metric result lists, one per sample.

        Returns:
            float: The computed corpus-level GLEU score.
        """

        total_hyp_len, total_ref_len = 0, 0
        # total_ngrams is a flattened list: even indices hold numerator sums and odd indices hold denominator sums
        total_ngrams = [0] * (len(metric_results[0][0][KEY_NGRAMS]) * 2)

        for sample_result in metric_results:
            # Use random sampling on available references for each sample
            for _ in range(self.num_iter):
                ref_idx = random.randint(0, len(sample_result) - 1)
                total_hyp_len += sample_result[ref_idx][KEY_HYP_LEN]
                total_ref_len += sample_result[ref_idx][KEY_REF_LEN]
                # Accumulate n-gram counts for each n
                for n, precision in sample_result[ref_idx][KEY_NGRAMS].items():
                    # Each precision is a tuple of (numerator, denominator)
                    assert len(precision) == 2
                    total_ngrams[2 * n - 2] += precision[0]
                    total_ngrams[2 * n - 1] += precision[1]

        # Apply smoothing if enabled; avoid division by zero by converting zeros to ones
        if self.smoothing:
            total_hyp_len = [x if x != 0 else 1 for x in total_ngrams]

        # Ensure no zero denominators remain
        assert len(list(filter(lambda x: x == 0, total_ngrams))) == 0
        # Compute average log precision over all ngrams
        log_gleu_prec = sum([math.log(float(x) / y) for x, y in zip(total_ngrams[0::2], total_ngrams[1::2])]) / 4
        # Compute final corpus score with length penalty
        score = math.exp(min([0, 1 - float(total_ref_len) / total_hyp_len]) + log_gleu_prec)
        return GLEUScorerResult(score=score)

    def score_sentence(self, scorer_inputs: List[List[Dict]]) -> float:
        """Compute the sentence-level GLEU score.

        For each sentence in the metric results, compute a score that aggregates n-gram
        precisions and length ratios. The sentence-level score is the average and standard deviation
        of all sentence scores.

        Args:
            scorer_inputs (List[List[Dict]]): List of metric result lists for each sentence.

        Returns:
            GLEUScorerResult: The sentence-level GLEU scoring result (with mean and std).
        """
        total_scores = []
        for sample_result in scorer_inputs:
            # Per sample, iterate over each reference result
            for ref_result in sample_result:
                # Ensure lengths are at least 1 to avoid division by zero
                ref_len = ref_result[KEY_REF_LEN] if ref_result[KEY_REF_LEN] != 0 else 1
                hyp_len = ref_result[KEY_HYP_LEN] if ref_result[KEY_HYP_LEN] != 0 else 1
                log_gleu_prec = 0.0

                # Sum log precision for each ngram order
                for n, precision in ref_result[KEY_NGRAMS].items():
                    numerator = precision[0] if precision[0] != 0 else 1
                    denominator = precision[1] if precision[1] != 0 else 1
                    log_gleu_prec += math.log(float(numerator) / denominator)
                log_gleu_prec /= self.order

                # Apply length penalty and compute score for this sentence
                ref_score = math.exp(min([0, 1 - float(ref_len) / hyp_len]) + log_gleu_prec)
                total_scores.append(ref_score)

        # Confidence interval can be added if necessary.
        return GLEUScorerResult(score=np.average(total_scores), std=np.std(total_scores))

    def print_result_table(self, result: OverallScorerResult, sout: TextIO = sys.stdout, **kwargs: Any) -> None:
        """Print the GLEU results in a formatted table.

        Args:
            result (OverallScorerResult): The result object containing scoring metrics.
            sout (TextIO, optional): Output stream (default is sys.stdout).
            **kwargs: Additional parameters to be printed after the table.
        """

        tabular_data = defaultdict(list)
        # Build table data from the scores dictionary
        for key, score_result in result.scores.items():
            # Make sure the score result is a valid GLEUScorerResult instance
            assert isinstance(score_result, GLEUScorerResult)
            tabular_data["metric"].append(key)
            tabular_data["score"].append(score_result.score)
            tabular_data["std"].append(score_result.std)

        table = tabulate(
            tabular_data, tablefmt="fancy_grid", headers="keys", floatfmt=".4f", missingval="N/A", numalign="left"
        )
        sout.write("\n" + table + "\n")
        for k, v in kwargs.items():
            sout.write(f"{k}: {v}\n")
        sout.write("\n")
