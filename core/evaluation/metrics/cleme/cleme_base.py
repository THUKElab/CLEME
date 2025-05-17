import copy
import sys
from typing import Any, List, TextIO, Tuple, Union

from tabulate import tabulate

from core.data.objects import Chunk, Dataset, Edit, Sample
from core.utils import get_logger

from ...aligners import BaseAligner
from ...classifers import BaseClassifier
from ...mergers import BaseMerger
from ...schema import SampleMetricResult
from ...scorers import BaseScorer, ScorerType
from ...tokenizers import BaseTokenizer
from ...weighers import BaseWeigher, WeigherType, get_weigher
from ..base import BaseEditMetric
from ..errant import Errant
from .cleme_utils import convert_edit_into_chunk, map_parallel, merge_edits

LOGGER = get_logger(__name__)


class CLEME(BaseEditMetric):
    """Evaluate unbiasedly GEC systems for multi-reference setting.

    For more details, refer to the following paper:
    CLEME: De-biasing Multi-reference Evaluation for Grammatical Error Correction [EMNLP 2023]
    """

    def __init__(
        self,
        lang: str,
        scorer: BaseScorer = None,
        scorer_type: ScorerType = ScorerType.PRF,
        weigher: BaseWeigher = None,
        weigher_type: WeigherType = WeigherType.NONE,
        tokenizer: BaseTokenizer = None,
        aligner: BaseAligner = None,
        merger: BaseMerger = None,
        classifier: BaseClassifier = None,
        enable_tqdm: bool = True,
        merge_distance: int = 0,
        output_visualize: Union[str, TextIO] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CLEME metric for grammatical error correction evaluation.

        Args:
            lang (str): Language code for tokenization and processing
            scorer (BaseScorer, optional): Scorer for evaluation. Defaults to None.
            scorer_type (ScorerType, optional): Type of scorer. Defaults to ScorerType.PRF.
            weigher (BaseWeigher, optional): Weigher for evaluation. Defaults to None.
            weigher_type (WeigherType, optional): Type of weigher. Defaults to WeigherType.NONE.
            tokenizer (BaseTokenizer, optional): Tokenizer for text processing. Defaults to None.
            aligner (BaseAligner, optional): Aligner for edit alignment. Defaults to None.
            merger (BaseMerger, optional): Merger for edit merging. Defaults to None.
            classifier (BaseClassifier, optional): Classifier for edit classification. Defaults to None.
            enable_tqdm (bool, optional): Whether to show progress bars. Defaults to True.
            merge_distance (int, optional): Maximum distance for merging edits. Defaults to 0.
            output_visualize (Union[str, TextIO], optional): Output file or stream for visualization. Defaults to None.
            **kwargs (Any): Additional arguments for initialization.
        """
        super().__init__(
            lang=lang, tokenizer=tokenizer, scorer=scorer, scorer_type=scorer_type, enable_tqdm=enable_tqdm, **kwargs
        )
        self.errant = Errant(
            lang=lang,
            scorer_type=scorer_type,
            tokenizer=tokenizer,
            aligner=aligner,
            merger=merger,
            classifier=classifier,
            enable_tqdm=enable_tqdm,
            **kwargs,
        )
        self.weigher = weigher or get_weigher(weigher_type)
        self.merge_distance = merge_distance
        self.output_visualize = output_visualize

    def parallel_to_edits(self, sample: Sample) -> List[List[List[Edit]]]:
        """Convert parallel sentences to edits using ERRANT.

        Args:
            sample (Sample): Sample containing source and target sentences

        Returns:
            List[List[List[Edit]]]: Extracted edits
        """
        return self.errant.parallel_to_edits(sample=sample)

    def prepare_datasets(self, dataset_hyp: Dataset, dataset_ref: Dataset) -> Tuple[Dataset, Dataset]:
        """Prepare datasets for chunk-level evaluation.

        1) Acquire Edits using Errant.
        2) Chunk partition.
        3) Compute average chunk length.

        Args:
            dataset_hyp (Dataset): Hyp dataset.
            dataset_ref (Dataset): Ref dataset.
            num_workers (int): _description_. Defaults to 1.
            write_hyp_m2 (str): _description_. Defaults to None.
            write_ref_m2 (str): _description_. Defaults to None.

        Returns:
            Tuple[Dataset, Dataset]: Prepared dataset with chunk partition.
        """
        # Extract edits
        dataset_hyp, dataset_ref = self.errant.prepare_datasets(dataset_hyp=dataset_hyp, dataset_ref=dataset_ref)

        # Remove empty references to avoid chunk partition collapse
        for sample in dataset_ref:
            valid_target_indices = [i for i, x in enumerate(sample.target) if x]
            if len(valid_target_indices) != len(sample.target):
                LOGGER.warning(f"Remove empty reference in {sample}")
                if len(valid_target_indices) == 0:
                    valid_target_indices = [0]
                sample.target = [sample.target[x] for x in valid_target_indices]
                sample.edits[0] = [sample.edits[0][x] for x in valid_target_indices]

        # Merge dataset_hyp and dataset_ref into one dataset
        merge_data = copy.deepcopy(dataset_hyp)
        for sample_idx, sample in enumerate(merge_data):
            sample.target.extend(dataset_ref[sample_idx].target)
            sample.edits[0].extend(copy.deepcopy(dataset_ref[sample_idx].edits[0]))

        # Chunk partition
        chunk_dataset = self.chunk_partition(merge_data, merge_distance=self.merge_distance)
        for sample_chunk, sample_hyp, sample_ref in zip(chunk_dataset, dataset_hyp, dataset_ref):
            assert len(chunk_dataset[sample_idx]) > 1
            sample_hyp.chunks = [[sample_chunk[0]]]
            sample_ref.chunks = [sample_chunk[1:]]

        # Visualize chunk partition if possible
        if self.output_visualize:
            sout = self.output_visualize
            if isinstance(sout, str):
                sout = open(sout, "w", encoding="utf-8")
            self.visualize(merge_data, chunk_dataset=chunk_dataset, sout=sout, delimiter=self.delimiter)
            if isinstance(self.output_visualize, str):
                sout.close()

        # Setup edit weigher
        self.weigher.setup(dataset_hyp=dataset_hyp, dataset_ref=dataset_ref)
        return dataset_hyp, dataset_ref

    def post_metric_evaluation(
        self, dataset_hyp: Dataset, dataset_ref: Dataset, metric_results: List[SampleMetricResult]
    ) -> None:
        """Compute edit weights for metric_results.

        Args:
            dataset_hyp (Dataset): Hyp dataset.
            dataset_ref (Dataset): Ref dataset.
            metric_results (List[MetricSampleResult]): Results of metric.
        """
        if self.weigher is not None:
            self.weigher.get_weights_batch(
                samples_hyp=dataset_hyp.samples, samples_ref=dataset_ref.samples, metric_results=metric_results
            )

    def chunk_partition(self, dataset: Dataset, merge_distance: int = 0) -> List[List[List[Chunk]]]:
        """Segment the source, hypothesis and references into chunk sequences.

        1) Construct token_mapping
        2) Merge edits with overlapping interval
        3) Convert edit into chunk

        Args:
            dataset (Dataset): Input dataset
            merge_distance (int): Maximum merging distance of two adjacent edits. Defaults to 0.

        Returns:
            List[List[List[Chunk]]]: Segmented chunks.
        """
        chunk_list_dataset = []
        for sample in dataset:
            # Segment sentence
            src_tokens = self.errant.tokenizer(sample.source[0], plain=True)
            # print(f"src_tokens: {src_tokens}")
            tgt_tokens_list = []
            for target in sample.target:
                tgt_tokens_list.append(self.errant.tokenizer(target, plain=True))
                # print(f"tgt_tokens: {tgt_tokens_list[-1]}")

            # Construct token_mapping
            edits_list, token_mapping_total = [], []
            for tgt_idx in range(len(sample.target)):
                edits = sample.edits[0][tgt_idx]
                edits = sorted(edits, key=lambda x: x.src_interval[0])
                edits_list.append(edits)
                token_mapping = map_parallel(src_tokens, edits)
                token_mapping_total.append(token_mapping)

            # Merge edits with overlapping intervals
            merge_edits_list, shared_interval_list = merge_edits(
                src_tokens, tgt_tokens_list, edits_list, token_mapping_total, merge_distance=merge_distance
            )

            # Convert edits into chunks
            chunk_list_total = convert_edit_into_chunk(
                src_tokens, tgt_tokens_list, merge_edits_list, shared_interval_list, token_mapping_total
            )
            chunk_list_dataset.append(chunk_list_total)
        return chunk_list_dataset

    def visualize(
        self,
        dataset: Dataset,
        chunk_dataset: List[List[List[Chunk]]] = None,
        sout: Union[str, TextIO] = sys.stdout,
        show_types: bool = False,
        delimiter: str = " ",
        **kwargs: Any,
    ) -> None:
        """Visualize the results of chunk partition into output stream.

        Creates a tabular representation of chunks for each sentence:
        tabular_data = {
            "sentence": [],
            "chunk-0": [],
            "chunk-1": [],
            "chunk-T": [],
            "chunk-N": [],
        }

        Args:
            dataset (Dataset): Input dataset
            chunk_dataset (List[List[List[Chunk]]], optional): Pre-computed chunks. Defaults to None.
            sout (Union[str, TextIO], optional): Output stream. Defaults to sys.stdout.
            show_types (bool, optional): Whether to show edit types. Defaults to False.
            delimiter (str, optional): Token delimiter. Defaults to " ".
            **kwargs (Any): Additional information to display
        """

        if chunk_dataset is None:
            chunk_dataset = self.chunk_partition(dataset)

        for chunk_sample in chunk_dataset:
            # Initialize sentences as a source and targets
            tabular_data = {"sentence": ["source"] + [f"target-{x}" for x in range(len(chunk_sample))]}
            for chunk_idx in range(len(chunk_sample[0])):
                chunks = [delimiter.join(chunk_sample[0][chunk_idx].src_tokens)] + [
                    delimiter.join(x[chunk_idx].tgt_tokens) for x in chunk_sample
                ]

                # Highlight changed chunks with stars
                if len(set(chunks)) > 1:
                    head_name = f"chunk-{chunk_idx} *"
                else:
                    head_name = f"chunk-{chunk_idx}"
                tabular_data[head_name] = chunks

                # Print error types of each chunk
                if show_types and len(set(chunks)) > 1:
                    types = [""] + [" ".join(x[chunk_idx].types) for x in chunk_sample]
                    tabular_data[f"Types-{chunk_idx}"] = types

            table = tabulate(
                tabular_data, tablefmt="fancy_grid", headers="keys", floatfmt=".3f", missingval="N/A", numalign="left"
            )
            sout.write("\n" + table + "\n")
            for k, v in kwargs.items():
                sout.write(f"{k}: {v}\n")
            sout.write("\n")
