"""`Metric` is an abstract class that enforces the implementation of a set of abstract methods,
so that a correctly implemented metric will work seamlessly with the rest of the codebase.

BaseMetric                            # Abstract Metric Class
├── NGramMetric                       # N-gram based Metric  i.e., GLEU
|   └── GLEUMetric
└── EditMetric                        # Edit-based Metric, including MaxMatch(M2), ERRANT
    ├── MaxMatch                      # Dynamic Programming based Metric
    ├── Errant                        # Linguistic-enhanced Metric
    └── CLEME                         # Chunk-based Metric, i.e, CLEME
        ├── DependentCLEME            # CLEME-dependent
        └── IndependentCLEME          # CLEME-independent
"""

import copy
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from core.data.objects import Dataset, Edit, Sample
from core.utils import get_logger, get_tqdm_iterable

from ..schema import OverallScorerResult, SampleMetricResult
from ..scorers import BaseScorer, ScorerType, get_scorer
from ..tokenizers import BaseTokenizer, get_tokenizer

LOGGER = get_logger(__name__)


class BaseMetric(ABC):
    """Abstract base class for all metrics.

    This class defines the interface that all metric implementations must follow,
    ensuring consistent behavior across different evaluation methods.

    Args:
        lang (str): Language code for the metric
        scorer (BaseScorer, optional): Scorer object for calculating scores
        scorer_type (ScorerType, optional): Type of scorer to use if scorer is not provided
        tokenizer (BaseTokenizer, optional): Tokenizer for text processing
        enable_tqdm (bool, optional): Whether to show progress bars. Defaults to True.
        table_print (bool, optional): Whether to print results in table format. Defaults to True.
        remove_unchanged_reference (bool, optional): Whether to remove unchanged references. Defaults to False.
    """

    def __init__(
        self,
        lang: str,
        scorer: BaseScorer = None,
        scorer_type: ScorerType = None,
        tokenizer: BaseTokenizer = None,
        enable_tqdm: bool = True,
        table_print: bool = True,
        remove_unchanged_reference: bool = False,
    ) -> None:
        self.lang = lang
        self.tokenizer = tokenizer or get_tokenizer(tokenizer_type=lang)
        self.scorer = scorer or get_scorer(scorer_type, table_print=table_print)
        self.enable_tqdm = enable_tqdm
        self.remove_unchanged = remove_unchanged_reference

    @property
    def classname(cls) -> str:
        """Returns the class name of the current instance."""
        return cls.__class__.__name__

    @property
    def delimiter(self) -> str:
        """Returns the delimiter between tokens."""
        return self.tokenizer.delimiter

    def prepare_datasets(self, dataset_hyp: Dataset, dataset_ref: Dataset) -> Tuple[Dataset, Dataset]:
        """Prepares hypothesis and reference datasets for evaluation.

        Args:
            dataset_hyp (Dataset): Hypothesis dataset
            dataset_ref (Dataset): Reference dataset

        Returns:
            Tuple[Dataset, Dataset]: Prepared hypothesis and reference datasets
        """
        self.check_datasets(dataset_hyp, dataset_ref)
        if self.remove_unchanged:
            dataset_ref = self.remove_unchanged_reference(dataset_ref)
        return dataset_hyp, dataset_ref

    @classmethod
    def check_datasets(cls, dataset_hyp: Dataset, dataset_ref: Dataset) -> None:
        """Validates that the hypothesis and reference datasets are compatible.

        Args:
            dataset_hyp (Dataset): Hypothesis dataset
            dataset_ref (Dataset): Reference dataset

        Raises:
            ValueError: If datasets are incompatible
        """
        if len(dataset_hyp) != len(dataset_ref):
            raise ValueError("Unequal source numbers for datasets.")

        for sample_hyp, sample_ref in zip(dataset_hyp, dataset_ref):
            if len(sample_hyp.source) != 1 or len(sample_ref.source) != 1:
                raise ValueError(f"Each hyp_sample must have a single source: {sample_hyp}")
            if sample_hyp.source[0] != sample_ref.source[0]:
                raise ValueError(
                    f"Sources of hyp and ref must be the same:\n"
                    f"hyp_source={sample_hyp.source[0]}\n"
                    f"ref_source={sample_ref.source[0]}"
                )
            if not sample_hyp.source[0]:
                raise ValueError("Source cannot be empty.")
            if len(sample_hyp.target) != 1:
                raise ValueError("The number of hyp target must be one.")
            if len(sample_ref.target) == 0:
                raise ValueError(f"No references for sample_{sample_ref.index}.")

    def remove_unchanged_reference(self, dataset: Dataset) -> Dataset:
        """Removes references that are identical to the source text.

        Args:
            dataset (Dataset): Input dataset

        Returns:
            Dataset: Dataset with unchanged references removed
        """
        new_dataset = copy.deepcopy(dataset)
        num_remove = 0
        for sample in new_dataset:
            if len(sample.target) == 1:
                continue
            src = sample.source[0]
            valid_target_indices = [i for i, x in enumerate(sample.target) if x != src]
            if len(valid_target_indices) != len(sample.target):
                num_remove += 1
                LOGGER.debug(f"Remove unchanged reference in {sample}")
                if len(valid_target_indices) == 0:
                    valid_target_indices = [0]
                sample.target = [sample.target[x] for x in valid_target_indices]
                sample.edits[0] = [sample.edits[0][x] for x in valid_target_indices]
        LOGGER.warning(f"Remove unchanged reference: {num_remove}")
        return new_dataset

    def evaluate(
        self, dataset_hyp: Dataset, dataset_ref: Dataset, persist_path: str = None, **kwargs
    ) -> Tuple[OverallScorerResult, List[SampleMetricResult]]:
        """Evaluates hypothesis dataset against reference dataset.

        Args:
            dataset_hyp (Dataset): Hypothesis dataset
            dataset_ref (Dataset): Reference dataset
            persist_path (str, optional): Path to save evaluation results

        Returns:
            Tuple[OverallScorerResult, List[SampleMetricResult]]: Overall scores and per-sample results
        """
        start_time = time.time()
        dataset_hyp, dataset_ref = self.prepare_datasets(dataset_hyp=dataset_hyp, dataset_ref=dataset_ref)
        prepare_time = time.time() - start_time

        # futures = []
        # executor = ProcessPoolExecutor(max_workers=num_workers)
        # for sample_hyp, sample_ref in zip(dataset_hyp, dataset_ref):
        #     future = executor.submit(
        #         self.evaluate_sample, sample_hyp, sample_ref, **kwargs
        #     )
        #     futures.append(future)

        # iterator = futures
        # if self.enable_tqdm:
        #     iterator = tqdm(
        #         iterator,
        #         total=len(futures),
        #         desc=f"{self.classname} Evaluate by {num_workers} workers",
        #     )
        # results = [future.result() for future in iterator]
        # executor.shutdown(wait=True)

        queue_with_progress = get_tqdm_iterable(
            items=zip(dataset_hyp, dataset_ref), show_progress=self.enable_tqdm, desc=f"{self.classname} Evaluating"
        )

        # Acquire metric results
        metric_results: List[SampleMetricResult] = []
        for sample_hyp, sample_ref in queue_with_progress:
            result = self.evaluate_sample(sample_hyp, sample_ref, **kwargs)
            metric_results.append(result)

        # Post process the metric results
        self.post_metric_evaluation(dataset_hyp=dataset_hyp, dataset_ref=dataset_ref, metric_results=metric_results)

        # Acquire score results
        score_result: OverallScorerResult = self.scorer(
            dataset_hyp=dataset_hyp, dataset_ref=dataset_ref, metric_results=metric_results
        )
        LOGGER.info(
            "{} Total samples: {}, Total time: {:.3f} seconds; Preparation time: {:.3f}".format(
                self.classname, len(dataset_hyp), time.time() - start_time, prepare_time
            )
        )

        # Save datasets and results
        if persist_path is not None:
            self.persist(
                dataset_hyp=dataset_hyp,
                persist_path=persist_path,
                score_result=score_result,
                metric_results=metric_results,
            )
        return score_result, metric_results

    def post_metric_evaluation(self, **kwargs: Any) -> None:
        """Performs post-processing after metric evaluation."""
        pass

    @abstractmethod
    def evaluate_sample(self, *args, **kwargs) -> List[Dict[str, int]]:
        """Evaluates a single sample.

        Returns:
            List[Dict[str, int]]: Evaluation results for the sample
        """
        raise NotImplementedError

    @abstractmethod
    def persist(
        self,
        dataset_hyp: Dataset,
        persist_path: str,
        score_result: OverallScorerResult,
        metric_results: List[SampleMetricResult],
    ) -> None:
        """Saves evaluation results to disk.

        Args:
            dataset_hyp (Dataset): Hypothesis dataset
            persist_path (str): Path to save results
            score_result (OverallScorerResult): Overall evaluation scores
            metric_results (List[SampleMetricResult]): Per-sample evaluation results
        """
        raise NotImplementedError


class BaseEditMetric(BaseMetric):
    """Base class for edit-based metrics. Inherits from BaseMetric.

    Edit-based metrics evaluate text corrections by analyzing the edits (insertions,
    deletions, substitutions) between source and target texts.
    """

    def prepare_dataset(self, dataset: Dataset, num_workers: int = 1, fresh_edit: bool = False) -> Dataset:
        """Prepares a dataset by computing edits for each sample.

        Args:
            dataset (Dataset): Dataset to prepare
            num_workers (int, optional): Number of workers for parallel processing. Defaults to 1.
            fresh_edit (bool, optional): Whether to recompute edits. Defaults to True.

        Returns:
            Dataset: Prepared dataset with computed edits
        """
        # iterator = dataset
        # if self.enable_tqdm:
        #     iterator = tqdm(dataset, total=len(dataset), desc="Tokenizing")
        # for sample in iterator:
        #     sample.source_tokens = [self.tokenizer(source) for source in sample.source]
        #     sample.target_tokens = [self.tokenizer(target) for target in sample.target]

        # futures = []
        # executor = ProcessPoolExecutor(max_workers=num_workers)
        # for sample in dataset:
        #     future = executor.submit(self.prepare_edits, sample)
        #     futures.append(future)

        # iterator = zip(dataset, futures)
        # if self.enable_tqdm:
        #     iterator = tqdm(
        #         iterator,
        #         total=len(futures),
        #         desc=f"{self.classname} Preparing Dataset by {num_workers} workers",
        #     )
        # for sample, future in iterator:
        #     edits = future.result()
        #     if edits is not None:
        #         sample._edits = edits
        # executor.shutdown(wait=True)

        queue_with_progress = get_tqdm_iterable(
            items=dataset.samples, show_progress=self.enable_tqdm, desc=f"{self.classname} preparing edits"
        )

        for sample in queue_with_progress:
            if fresh_edit or not sample.edits:
                sample.edits = self.parallel_to_edits(sample)
        return dataset

    def prepare_datasets(self, dataset_hyp: Dataset, dataset_ref: Dataset) -> Tuple[Dataset, Dataset]:
        """Prepares hypothesis and reference datasets for edit-based evaluation.

        Args:
            dataset_hyp (Dataset): Hypothesis dataset
            dataset_ref (Dataset): Reference dataset

        Returns:
            Tuple[Dataset, Dataset]: Prepared hypothesis and reference datasets
        """
        dataset_hyp, dataset_ref = super().prepare_datasets(dataset_hyp=dataset_hyp, dataset_ref=dataset_ref)
        dataset_hyp = self.prepare_dataset(dataset_hyp)
        dataset_ref = self.prepare_dataset(dataset_ref)
        return dataset_hyp, dataset_ref

    def evaluate_sample(self, sample_hyp: Sample, sample_ref: Sample, detection: bool = False) -> SampleMetricResult:
        """Evaluates a single sample for correction.

        Args:
            sample_hyp (Sample): Hypothesis sample
            sample_ref (Sample): Reference sample

        Returns:
            SampleMetricResult: Evaluation results for the sample
        """
        # Calculate TP, FP and FN counts
        if detection:
            return self.evaluate_sample_detection(sample_hyp=sample_hyp, sample_ref=sample_ref)
        else:
            return self.evaluate_sample_correction(sample_hyp=sample_hyp, sample_ref=sample_ref)

    @abstractmethod
    def parallel_to_edits(self, sample: Sample) -> List[List[List[Edit]]]:
        """Converts parallel texts to edit operations.

        Args:
            sample (Sample): Sample with source and target texts

        Returns:
            List[List[List[Edit]]]: Computed edits
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_sample_correction(self, sample_hyp: Sample, sample_ref: Sample) -> SampleMetricResult:
        """Evaluates a sample for correction quality.

        Args:
            sample_hyp (Sample): Hypothesis sample
            sample_ref (Sample): Reference sample

        Returns:
            SampleMetricResult: Correction evaluation results
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_sample_detection(self, sample_hyp: Sample, sample_ref: Sample) -> SampleMetricResult:
        """Evaluates a sample for error detection quality.

        Args:
            sample_hyp (Sample): Hypothesis sample
            sample_ref (Sample): Reference sample

        Returns:
            SampleMetricResult: Detection evaluation results
        """
        raise NotImplementedError

    def persist(
        self,
        dataset_hyp: Dataset,
        persist_path: str,
        score_result: OverallScorerResult,
        metric_results: List[SampleMetricResult],
    ) -> None:
        """Saves edit-based evaluation results to disk.

        Args:
            dataset_hyp (Dataset): Hypothesis dataset
            persist_path (str): Path to save results
            score_result (OverallScorerResult): Overall evaluation scores
            metric_results (List[SampleMetricResult]): Per-sample evaluation results
        """
        persist_json = {"scores": score_result.model_dump(), "samples": dataset_hyp.model_dump()["samples"]}
        for sample, metric_result in zip(persist_json["samples"], metric_results):
            sample.pop("edits")
            sample.pop("chunks")
            sample["metric_result"] = metric_result.model_dump()
        with open(persist_path, "w", encoding="utf-8") as f:
            json.dump(persist_json, f, indent=2, ensure_ascii=False)
