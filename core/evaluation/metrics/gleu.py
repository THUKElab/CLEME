# This file is referenced from:
# Ground Truth for Grammatical Error Correction Metrics [ACL 2015]
# by Courtney Napoles, Keisuke Sakaguchi, Matt Post, and Joel Tetreault.
# <https://github.com/alopez/en600.468/blob/master/reranker/>

from collections import Counter, defaultdict
from typing import Any, Dict, List

from core.data.objects import Sample
from core.utils import get_logger

from ..scorers import BaseScorer, GLEUScorer, ScorerType
from .base import BaseMetric

LOGGER = get_logger(__name__)

KEY_HYP_LEN = "hyp_len"
KEY_REF_LEN = "ref_len"
KEY_NGRAMS = "ngrams"


class GLEU(BaseMetric):
    """Implementation of the GLEU (Generalized Language Evaluation Understanding) metric.

    GLEU is an n-gram based metric similar to BLEU but adapted for grammatical error correction.
    It penalizes n-grams that were correct in the source but changed incorrectly in the hypothesis.

    Args:
        lang (str): Language code
        scorer (BaseScorer, optional): Scorer for calculating scores
        scorer_type (ScorerType, optional): Type of scorer to use. Defaults to ScorerType.GLEU.
        **kwargs: Additional arguments
    """

    def __init__(
        self, lang: str, scorer: BaseScorer = None, scorer_type: ScorerType = ScorerType.GLEU, **kwargs: Any
    ) -> None:
        super().__init__(lang=lang, scorer=scorer, scorer_type=scorer_type, **kwargs)

    @property
    def order(self) -> int:
        """Gets the n-gram order used for GLEU calculation.

        Returns:
            int: N-gram order
        """
        assert isinstance(self.scorer, GLEUScorer)
        return self.scorer.order

    def evaluate_sample(self, sample_hyp: Sample, sample_ref: Sample) -> List[Dict]:
        """Evaluates a sample using GLEU metric.

        Args:
            sample_hyp (Sample): Hypothesis sample
            sample_ref (Sample): Reference sample

        Returns:
            List[Dict]: GLEU evaluation results
        """
        src_split: List[str] = self.tokenizer(sample_hyp.source[0], plain=True)
        hyp_split: List[str] = self.tokenizer(sample_hyp.target[0], plain=True)
        refs_split: List[List[str]] = [self.tokenizer(x, plain=True) for x in sample_ref.target]

        src_ngrams = [self.get_ngram_counts(src_split, n) for n in range(1, self.order + 1)]
        hyp_ngrams = [self.get_ngram_counts(hyp_split, n) for n in range(1, self.order + 1)]
        refs_len = [len(x) for x in refs_split]

        results = []
        for ref_idx, ref_split in enumerate(refs_split):
            ngrams_precision = defaultdict()
            for n in range(1, self.order + 1):
                _src_ngrams = src_ngrams[n - 1]
                _hyp_ngrams = hyp_ngrams[n - 1]
                _ref_ngrams = self.get_ngram_counts(ref_split, n)
                src_ref_diff = self.get_ngram_diff(_src_ngrams, _ref_ngrams)

                numerator = sum((_hyp_ngrams & _ref_ngrams).values()) - sum((_hyp_ngrams & src_ref_diff).values())
                numerator = numerator if numerator > 0 else 0
                denominator = max([len(hyp_split) + 1 - n, 0])
                ngrams_precision[n] = [numerator, denominator]
            results.append(
                {
                    KEY_HYP_LEN: len(hyp_split),
                    KEY_REF_LEN: refs_len[ref_idx],
                    KEY_NGRAMS: ngrams_precision,
                }
            )
        return results

    @staticmethod
    def get_ngram_counts(sentence: List[str], n: int) -> Counter:
        """Counts n-grams in a sentence.

        Args:
            sentence (List[str]): Tokenized sentence
            n (int): N-gram size

        Returns:
            Counter: N-gram counts
        """
        return Counter([tuple(sentence[i : i + n]) for i in range(len(sentence) + 1 - n)])

    @staticmethod
    def get_ngram_diff(a: Counter, b: Counter) -> Counter:
        """Returns n-grams in Counter 'a' but not in Counter 'b'.

        Args:
            a (Counter): First counter
            b (Counter): Second counter

        Returns:
            Counter: Difference counter
        """
        diff = Counter(a)
        for k in set(a) & set(b):
            del diff[k]
        return diff

    def persist(self) -> None:
        """Saves GLEU evaluation results to disk.

        Args:
            dataset_hyp (Dataset): Hypothesis dataset
            persist_path (str): Path to save results
            score_result (OverallScorerResult): Overall evaluation scores
            metric_results (List[SampleMetricResult]): Per-sample evaluation results
        """
        raise NotImplementedError()
