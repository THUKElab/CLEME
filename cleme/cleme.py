import copy
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
from tabulate import tabulate

from .chunk import (
    Chunk,
    map_parallel,
    merge_edit,
    convert_edit_into_chunk,
    chunk_list_to_text,
    all_correct,
)
from .constants import *
from .data import Dataset
from .scorers import SCORERS
from .utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class LengthWeigher:
    alpha: float = field(
        default=3.0, metadata={"help": "Scale factor"}
    )
    bias: float = field(
        default=0.0, metadata={"help": "Bias factor (default not used)"}
    )
    min_value: float = field(
        default=1.0, metadata={"help": "Clamp factor of minimal value"}
    )
    max_value: float = field(
        default=1.0, metadata={"help": "Clamp factor of maximal value"}
    )
    reverse: bool = field(
        default=False, metadata={"help": "Reverse the value or not"}
    )

    def __call__(self, edit_len: int) -> float:
        if self.reverse:
            weight = self.alpha * (1 / (1 + (self.alpha - 1) * np.exp(edit_len - self.bias)))
        else:
            weight = self.alpha * (1 / (1 + (self.alpha - 1) * np.exp(-edit_len + self.bias)))
        return np.clip(weight, self.min_value, self.max_value).item()

    def __str__(self):
        return f"SigmoidWeigher(alpha={self.alpha}, bias={self.bias}, " \
               f"min_value={self.min_value}, max_value={self.max_value}, reverse={self.reverse})"


class CLEME(ABC):
    """ CLEME: De-biasing Multi-reference Evaluation for Grammatical Error Correction [EMNLP 2023]
        :param scorer: used to compute Precision, Recall and F scores (choices: `corpus`, `sentence`).
        :param weigher_config: configuration of Length Weighting.
    """

    def __init__(self, scorer: str = "corpus", weigher_config: Dict = None):
        super().__init__()
        self.scorer = self.build_scorer(name=scorer)
        self.weigher_config = weigher_config
        self.weigher_tp = None
        self.weigher_fp = None
        self.weigher_fn = None

    @classmethod
    def build_scorer(cls, name: str, *args, **kwargs):
        class_type = SCORERS[name]
        return class_type(*args, **kwargs)

    def evaluate(self, dataset_hyp: Dataset, dataset_ref: Dataset):
        start_time = time.time()
        self.check_datasets(dataset_hyp, dataset_ref)
        dataset_hyp, dataset_ref = self.prepare_datasets(dataset_hyp, dataset_ref)
        prepare_time = time.time() - start_time

        results = []
        for sample_hyp, sample_ref in zip(dataset_hyp, dataset_ref):
            results.append(self.evaluate_sample_correction(sample_hyp, sample_ref))

        score = self.scorer(results)
        LOGGER.info(
            "{} Total samples: {}, Total time: {:.3f} seconds; Preparation time: {:.3f}".format(
                self.__class__, len(dataset_hyp), time.time() - start_time, prepare_time,
            )
        )
        return score, results

    @abstractmethod
    def evaluate_sample_correction(
            self,
            chunks_hyp: List[Chunk],
            chunks_ref: List[List[Chunk]],
    ) -> List[Dict[str, int]]:
        raise NotImplementedError()

    def prepare_datasets(self, dataset_hyp: Dataset, dataset_ref: Dataset):
        """ Prepare for chunk-level evaluation
            1) Acquire Edits using Errant
            2) Chunk partition
            2) Calculate average chunk length
        """
        # Merge dataset_hyp and dataset_ref into one dataset for convenience
        merge_data = copy.deepcopy(dataset_hyp)
        for sample_idx, sample in enumerate(merge_data):
            sample.target.extend(dataset_ref[sample_idx].target)
            sample.edits[0].extend(dataset_ref[sample_idx].edits[0])

        # Chunk partition
        chunk_dataset = self.chunk_partition(merge_data)
        chunk_dataset_hyp, chunk_dataset_ref = [], []
        for sample_idx in range(len(dataset_hyp)):
            assert len(chunk_dataset[sample_idx]) > 1
            chunk_dataset_hyp.append(chunk_dataset[sample_idx][0])
            chunk_dataset_ref.append(chunk_dataset[sample_idx][1:])

        # Prepare for length weighting
        self.setup_length_weighting(chunk_dataset_ref)
        return chunk_dataset_hyp, chunk_dataset_ref

    def setup_length_weighting(self, chunk_dataset_ref: List[List[List[Chunk]]]):
        def construct_length_weight(config, avg_chunk_len):
            return LengthWeigher(
                alpha=config["alpha"],
                bias=avg_chunk_len,
                min_value=config["min_value"],
                max_value=config["max_value"],
                reverse=config["reverse"],
            )

        # Calculate the average length of correct chunks (i.e., Unchanged Chunks)
        # and incorrect chunks (i.e., Corrected/Dummy Chunks)
        chunk_len_correct, chunk_len_incorrect = [], []

        for chunks_list in chunk_dataset_ref:
            for chunks in chunks_list:
                for chunk in chunks:
                    if chunk.type:  # Corrected/Dummy Chunk
                        chunk_len = max(len(chunk.src_tokens), len(chunk.tgt_tokens))
                        chunk_len_incorrect.append(chunk_len)
                    else:  # Unchanged Chunk
                        chunk_len_correct.append(len(chunk.src_tokens))
        avg_chunk_len_correct = np.average(chunk_len_correct)
        avg_chunk_len_incorrect = np.average(chunk_len_incorrect)
        LOGGER.info(
            f"avg_chunk_len_correct={round(avg_chunk_len_correct, 2)}, "
            f"avg_chunk_len_incorrect={round(avg_chunk_len_incorrect, 2)}"
        )

        self.weigher_tp = construct_length_weight(
            self.weigher_config["tp"],
            avg_chunk_len_incorrect,
        )
        self.weigher_fp = construct_length_weight(
            self.weigher_config["fp"],
            avg_chunk_len_incorrect,
        )
        self.weigher_fn = construct_length_weight(
            self.weigher_config["fn"],
            avg_chunk_len_incorrect,
        )
        LOGGER.info(f"length_weight_tp: {self.weigher_tp}")
        LOGGER.info(f"length_weight_fp: {self.weigher_fp}")
        LOGGER.info(f"length_weight_fn: {self.weigher_fn}")

    @classmethod
    def check_datasets(cls, dataset_hyp: Dataset, dataset_ref: Dataset):
        assert len(dataset_hyp) == len(dataset_ref)
        for sample_hyp, sample_ref in zip(dataset_hyp, dataset_ref):
            assert len(sample_hyp.source) == len(sample_hyp.target) == len(sample_hyp.source) == 1, \
                f"{sample_hyp} || {sample_ref}"
            assert sample_hyp.source[0] == sample_ref.source[0]

    def chunk_partition(self, dataset: Dataset) -> List[List[List[Chunk]]]:
        """ Segment the source, hypothesis and references into chunk sequences
            1) Construct token_mapping
            2) Merge edits with overlapping interval
            3) Convert edit into chunk
        """
        chunk_list_dataset = []
        for sample in dataset:
            for tgt_idx, tgt in enumerate(sample.target):
                if not tgt and len(sample.target) > 2:
                    LOGGER.warning(f"Ignore empty reference: {sample.source[0]} || {tgt}")
                    del sample.target[tgt_idx]
                    del sample.edits[0][tgt_idx]

            # Segment sentence
            src_tokens = sample.source[0].split()
            tgt_tokens_list = [x.split() for x in sample.target]

            # Construct token_mapping
            edits_list, token_mapping_total = [], []
            for tgt_idx in range(len(sample.target)):
                edits = sample.edits[0][tgt_idx]
                edits = sorted(edits, key=lambda x: x.src_interval[0])
                edits_list.append(edits)
                token_mapping = map_parallel(src_tokens, edits)
                token_mapping_total.append(token_mapping)
            # Merge edits with overlapping interval
            merge_edits_list, shared_interval_list = merge_edit(
                src_tokens,
                tgt_tokens_list,
                edits_list,
                token_mapping_total,
            )
            # Convert edit into chunk
            chunk_list_total = convert_edit_into_chunk(
                src_tokens,
                tgt_tokens_list,
                merge_edits_list,
                shared_interval_list,
                token_mapping_total,
            )
            chunk_list_dataset.append(chunk_list_total)
        return chunk_list_dataset

    def visualize(
            self,
            dataset_ref: Dataset,
            dataset_hyp: Dataset = None,
            sout=sys.stdout,
            tablefmt: str = "fancy_grid",
            display_types: bool = False,
    ):
        """ Visualize the results of chunk partition.
            tabular_data {
                "sentence": [],
                "chunk-0": [],
                "chunk-1": [],
                ...,
                "chunk-N": [],
            }
        """
        if dataset_hyp is None:
            dataset = dataset_ref
        else:
            dataset = copy.deepcopy(dataset_ref)
            dataset.merge(dataset_hyp)

        chunk_results = self.chunk_partition(dataset)
        for chunk_result in chunk_results:
            tabular_data = {
                "sentence": ["source"] + [f"target-{x}" for x in list(range(len(chunk_result)))],
            }
            for chunk_idx in range(len(chunk_result[0])):
                chunks = [" ".join(chunk_result[0][chunk_idx].src_tokens)] + \
                         [" ".join(x[chunk_idx].tgt_tokens) for x in chunk_result]
                if len(set(chunks)) > 1:
                    head_name = f"chunk-{chunk_idx} *"
                else:
                    head_name = f"chunk-{chunk_idx}"
                tabular_data[head_name] = chunks
                if display_types and len(set(chunks)) > 1:
                    types = [""] + [" ".join(x[chunk_idx].type) for x in chunk_result]
                    tabular_data[f"Types-{chunk_idx}"] = types
            table = tabulate(
                tabular_data,
                tablefmt=tablefmt,
                headers="keys",
                floatfmt='.3f',
                missingval='N/A',
                numalign='left',
            )
            sout.write(table + "\n")


class DependentChunkMetric(CLEME):
    def evaluate_sample_correction(
            self,
            chunks_hyp: List[Chunk],
            chunks_refs: List[List[Chunk]],
    ) -> List[Dict[str, int]]:
        result = []
        for ref_id, chunks_ref in enumerate(chunks_refs):
            src, ref = chunk_list_to_text(chunks_ref)
            LOGGER.debug(f"ref: {ref}")

            tp, fp, fn, tn = 0, 0, 0, 0
            tp_chunks, fp_chunks, fn_chunks, tn_chunks = [], [], [], []
            for chunk_idx, chunk_hyp in enumerate(chunks_hyp):
                chunk_len = max(len(chunk_hyp.src_tokens), len(chunk_hyp.tgt_tokens))
                if chunk_hyp.type:
                    if chunk_hyp == chunks_ref[chunk_idx]:
                        weight = self.weigher_tp(chunk_len)
                        tp += weight
                        tp_chunks.append((chunk_hyp, chunks_ref[chunk_idx]))
                        LOGGER.debug(f"{round(weight, 2)} TP: {chunk_hyp} || {chunks_ref[chunk_idx]}")
                    else:
                        weight = self.weigher_fp(chunk_len)
                        fp += weight
                        fp_chunks.append((chunk_hyp, chunks_ref[chunk_idx]))
                        LOGGER.debug(f"{round(weight, 2)} FP: {chunk_hyp} || {chunks_ref[chunk_idx]}")
                else:
                    if chunk_hyp != chunks_ref[chunk_idx]:
                        weight = self.weigher_fn(chunk_len)
                        fn += weight
                        fn_chunks.append((chunk_hyp, chunks_ref[chunk_idx]))
                        LOGGER.debug(f"{round(weight, 2)} FN: {chunk_hyp} || {chunks_ref[chunk_idx]}")
                    else:
                        weight = 1.00
                        tn += weight
                        tn_chunks.append((chunk_hyp, chunks_ref[chunk_idx]))
                        # LOGGER.debug(f"{round(weight, 2)} TN: {chunk_hyp}")
            result.append({
                KEY_TP: tp,
                KEY_FP: fp,
                KEY_FN: fn,
                KEY_TN: tn,
                KEY_TP_EDIT: tp_chunks.copy(),
                KEY_FP_EDIT: fp_chunks.copy(),
                KEY_FN_EDIT: fn_chunks.copy(),
                KEY_TN_EDIT: tn_chunks.copy(),
            })
            LOGGER.debug(f"tp={round(tp, 2)}, fp={round(fp, 2)}, fn={round(fn, 2)}, tn={round(tn, 2)}")
        return result


class IndependentChunkMetric(CLEME):
    def evaluate_sample_correction(
            self,
            chunks_hyp: List[Chunk],
            chunks_ref: List[List[Chunk]],
    ) -> List[Dict[str, int]]:
        result = []
        tp, fp, fn, tn = 0, 0, 0, 0
        for chunk_idx, chunk_hyp in enumerate(chunks_hyp):
            cand_chunk_list = [x[chunk_idx] for x in chunks_ref]
            chunk_len = max(len(chunk_hyp.src_tokens), len(chunk_hyp.tgt_tokens))
            if chunk_hyp.type:
                if chunk_hyp in cand_chunk_list:
                    weight = self.weigher_tp(chunk_len)
                    tp += weight
                    LOGGER.debug(f"{round(weight, 2)} TP: {chunk_hyp} || {cand_chunk_list}")
                else:
                    weight = self.weigher_fp(chunk_len)
                    fp += weight
                    LOGGER.debug(f"{round(weight, 2)} FP: {chunk_hyp} || {cand_chunk_list}")
            else:
                if all_correct(cand_chunk_list):
                    weight = self.weigher_fn(chunk_len)
                    fn += weight
                    LOGGER.debug(f"{round(weight, 2)} FN: {chunk_hyp} || {cand_chunk_list}")
                else:
                    weight = 1.00
                    tn += weight
                    # LOGGER.debug(f"{round(weight, 2)} TN: {chunk_hyp}")
        result.append({
            KEY_TP: tp,
            KEY_FP: fp,
            KEY_FN: fn,
            KEY_TN: tn,
        })
        LOGGER.debug(f"tp={round(tp, 2)}, fp={round(fp, 2)}, fn={round(fn, 2)}, tn={round(tn, 2)}")
        return result
