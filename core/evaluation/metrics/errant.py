from typing import Any, List

from core.data.objects import Edit, Sample

from ..aligners import BaseAligner, get_aligner
from ..classifers import BaseClassifier, get_classifier
from ..mergers import BaseMerger, get_merger
from ..schema import BaseEditMetricResult, SampleMetricResult
from ..scorers import BaseScorer, ScorerType
from ..tokenizers import BaseTokenizer
from .base import BaseEditMetric


class Errant(BaseEditMetric):
    """Implementation of the ERRANT (ERRor ANnotation Toolkit) metric.

    ERRANT is a grammatical error correction metric that categorizes edits into error types
    and evaluates system performance based on these categorizations.

    Args:
        lang (str): Language code
        scorer (BaseScorer, optional): Scorer for calculating scores
        scorer_type (ScorerType, optional): Type of scorer to use. Defaults to ScorerType.PRF.
        tokenizer (BaseTokenizer, optional): Tokenizer for text processing
        aligner (BaseAligner, optional): Aligner for aligning source and target texts
        merger (BaseMerger, optional): Merger for merging aligned sequences into edits
        classifier (BaseClassifier, optional): Classifier for classifying error types
        enable_tqdm (bool, optional): Whether to show progress bars. Defaults to True.
    """

    def __init__(
        self,
        lang: str,
        scorer: BaseScorer = None,
        scorer_type: ScorerType = ScorerType.PRF,
        tokenizer: BaseTokenizer = None,
        aligner: BaseAligner = None,
        merger: BaseMerger = None,
        classifier: BaseClassifier = None,
        enable_tqdm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            lang=lang, tokenizer=tokenizer, scorer=scorer, scorer_type=scorer_type, enable_tqdm=enable_tqdm, **kwargs
        )
        self.aligner = aligner or get_aligner(aligner_type=lang)
        self.merger = merger or get_merger(merger_type=lang)
        self.classifier = classifier or get_classifier(classifier_type=lang)

    def pickable_edits(self, sample_edits: List[List[List[Edit]]]) -> List[List[List[Edit]]]:
        """Makes edit objects pickable by removing non-serializable attributes.

        Args:
            sample_edits (List[List[List[Edit]]]): Edit objects to make pickable

        Returns:
            List[List[List[Edit]]]: Pickable edit objects
        """
        for edits in sample_edits[0]:
            for edit in edits:
                edit.src_tokens_tok = None
                edit.tgt_tokens_tok = None
        return sample_edits

    def parallel_to_edits(self, sample: Sample) -> List[List[List[Edit]]]:
        """Generates edits from parallel source and target texts.

        Args:
            sample (Sample): Sample with source and target texts

        Returns:
            List[List[List[Edit]]]: Generated edits
        """
        # source, target = source.strip(), target.strip()
        if not sample.target:
            return [[[]]]
        src_tok = self.tokenizer(sample.source[0])
        src_detok = self.tokenizer.detokenize(src_tok).strip()

        sample_edits = [[]]
        for idx, target in enumerate(sample.target):
            if src_detok != target:
                # Tokenize target
                tgt_tok = self.tokenizer(sample.target[idx])
                # Align source and target
                align_seq = self.aligner(src_tok, tgt_tok)
                # Merge alignment
                edits = self.merger(src_tok, tgt_tok, align_seq, idx)
                # Update Edit object with an updated error type
                for edit in edits:
                    self.classifier(src_tok, tgt_tok, edit)
                sample_edits[0].append(edits)
            else:
                sample_edits[0].append([])
        return self.pickable_edits(sample_edits)

    def evaluate_sample_correction(self, sample_hyp: Sample, sample_ref: Sample) -> SampleMetricResult:
        """Evaluates correction quality by comparing hypothesis and reference edits.

        Calculates true positives (TP), false positives (FP), and false negatives (FN)
        for correction evaluation.

        Args:
            sample_hyp (Sample): Hypothesis sample
            sample_ref (Sample): Reference sample

        Returns:
            SampleMetricResult: Correction evaluation results
        """
        # Only consider the first source and target of hypothesis
        hyp_edits = sample_hyp.edits[0][0]
        ref_results: List[BaseEditMetricResult] = []

        for ref_edits in sample_ref.edits[0]:
            tp_edits, fp_edits, fn_edits = [], [], []

            # Classify TP and FP edits
            for hyp_edit in hyp_edits:
                if hyp_edit in ref_edits:
                    tp_edits.append(hyp_edit)
                else:
                    fp_edits.append(hyp_edit)
            # Classify FN edits
            for ref_edit in ref_edits:
                if ref_edit not in hyp_edits:
                    fn_edits.append(ref_edit)

            # Save detailed results
            ref_result = BaseEditMetricResult(
                tp_edits=tp_edits.copy(), fp_edits=fp_edits.copy(), fn_edits=fn_edits.copy()
            )
            ref_results.append(ref_result)
        return SampleMetricResult(ref_results=ref_results)

    def evaluate_sample_detection(self, sample_hyp: Sample, sample_ref: Sample) -> SampleMetricResult:
        """Evaluates error detection quality.

        Args:
            sample_hyp (Sample): Hypothesis sample
            sample_ref (Sample): Reference sample

        Returns:
            SampleMetricResult: Detection evaluation results
        """

        def detection_helper(edit: Edit, edits: List[Edit]):
            for e in edits:
                if edit.src_interval == e.src_interval:
                    return True
            return False

        # Only consider the first source and target of hypothesis
        hyp_edits = sample_hyp.edits[0][0]
        ref_results: List[BaseEditMetricResult] = []

        for ref_edits in sample_ref.edits[0]:
            tp_edits, fp_edits, fn_edits = [], [], []

            # Classify TP and FP edits
            for hyp_edit in hyp_edits:
                if detection_helper(hyp_edit, ref_edits):
                    tp_edits.append(hyp_edit)
                else:
                    fp_edits.append(hyp_edit)
            # Classify FN edits
            for ref_edit in ref_edits:
                if not detection_helper(ref_edit, hyp_edits):
                    fn_edits.append(ref_edit)

            # Save detailed results
            ref_result = BaseEditMetricResult(
                tp_edits=tp_edits.copy(), fp_edits=fp_edits.copy(), fn_edits=fn_edits.copy()
            )
            ref_results.append(ref_result)
        return SampleMetricResult(ref_results=ref_results)
