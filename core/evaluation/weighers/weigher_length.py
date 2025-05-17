import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from core.data.objects import Chunk, Dataset
from core.utils import get_logger

from ..schema import BaseChunkMetricResult, SampleMetricResult
from .weigher_base import BaseWeigher

LOGGER = get_logger(__name__, level=logging.INFO)


class LengthWeigher(BaseWeigher, BaseModel):
    """Compute weights for edits/chunks based on their length.

    For true positives (TP), false negatives (FN), and true negatives (TN), longer edits receive
    larger weights. For false positives (FP), longer edits receive smaller weights.

    This is based on the method described in:
    CLEME: De-biasing Multi-reference Evaluation for Grammatical Error Correction [EMNLP 2023]

    Attributes:
        tp_alpha (float): Scale factor for TP edits (default 2.0).
        tp_bias (float): Bias factor for TP edits (default 0.0, later set in setup()).
        tp_min_value (float): Minimum weight for TP edits (default 0.75).
        tp_max_value (float): Maximum weight for TP edits (default 1.25).
        fp_alpha (float): Scale factor for FP edits (default 2.0).
        fp_bias (float): Bias factor for FP edits (default 0.0, later set in setup()).
        fp_min_value (float): Minimum weight for FP edits (default 0.75).
        fp_max_value (float): Maximum weight for FP edits (default 1.25).
        fn_alpha (float): Scale factor for FN edits (default 2.0).
        fn_bias (float): Bias factor for FN edits (default 0.0, later set in setup()).
        fn_min_value (float): Minimum weight for FN edits (default 0.75).
        fn_max_value (float): Maximum weight for FN edits (default 1.25).
        tn_alpha (float): Scale factor for TN edits (default 2.0).
        tn_bias (float): Bias factor for TN edits (default 0.0, later set in setup()).
        tn_min_value (float): Minimum weight for TN edits (default 1.0).
        tn_max_value (float): Maximum weight for TN edits (default 1.0).
    """

    tp_alpha: float = Field(default=2.0, description="Scale factor of tp_edits")
    tp_bias: float = Field(default=0.0, description="Bias factor of tp_edits")
    tp_min_value: float = Field(default=0.75, description="Minimum weight of tp_edits")
    tp_max_value: float = Field(default=1.25, description="Maximum weight of tp_edits")

    fp_alpha: float = Field(default=2.0, description="Scale factor of fp_edits")
    fp_bias: float = Field(default=0.0, description="Bias factor of fp_edits")
    fp_min_value: float = Field(default=0.75, description="Minimum weight of fp_edits")
    fp_max_value: float = Field(default=1.25, description="Maximum weight of fp_edits")

    fn_alpha: float = Field(default=2.0, description="Scale factor of fn_edits")
    fn_bias: float = Field(default=0.0, description="Bias factor of fn_edits")
    fn_min_value: float = Field(default=0.75, description="Minimum weight of fn_edits")
    fn_max_value: float = Field(default=1.25, description="Maximum weight of fn_edits")

    tn_alpha: float = Field(default=2.0, description="Scale factor of tn_edits")
    tn_bias: float = Field(default=0.0, description="Bias factor of tn_edits")
    tn_min_value: float = Field(default=1.0, description="Minimum weight of tn_edits")
    tn_max_value: float = Field(default=1.0, description="Maximum weight of tn_edits")

    def __init__(self, **kwargs: Any) -> None:
        """Inherits initialization from both BaseWeigher and BaseModel."""
        super().__init__()

    def setup(self, dataset_ref: Dataset, **kwargs) -> None:
        """Setup the length weigher by computing average chunk lengths from the reference dataset.

        This method iterates through the reference dataset to compute the average length of correct
        chunks (unchanged) and incorrect chunks (corrected/dummy chunks). These average lengths are then
        used as bias factors for weighting edits.

        Args:
            dataset_ref (Dataset): The reference dataset containing samples.
        """

        # Lists to hold lengths of unchanged (correct) and corrected (incorrect) chunks.
        chunk_len_correct, chunk_len_incorrect = [], []

        # Iterate through samples in the reference dataset.
        for sample in dataset_ref:
            for chunks in sample.chunks[0]:
                for chunk in chunks:
                    if chunk.types:
                        # This chunk is a corrected or dummy chunk.
                        # Use the maximum of source/target token lengths as the chunk length.
                        chunk_len = max(len(chunk.src_tokens), len(chunk.tgt_tokens))
                        # chunk_len = (len(chunk.src_tokens) + len(chunk.tgt_tokens)) / 2
                        chunk_len_incorrect.append(chunk_len)
                    else:
                        # Unchanged chunk.
                        chunk_len_correct.append(len(chunk.src_tokens))
        avg_chunk_len_correct = np.average(chunk_len_correct)
        avg_chunk_len_incorrect = np.average(chunk_len_incorrect)

        LOGGER.info(
            f"avg_chunk_len_correct={round(avg_chunk_len_correct, 2)}, "
            f"avg_chunk_len_incorrect={round(avg_chunk_len_incorrect, 2)}"
        )

        # Set bias factors using the average incorrect chunk length.
        self.tp_bias = avg_chunk_len_incorrect
        self.fp_bias = avg_chunk_len_incorrect
        self.fn_bias = avg_chunk_len_incorrect
        self.tn_bias = avg_chunk_len_incorrect

    def __call__(self, metric_result: SampleMetricResult, **kwargs: Any) -> None:
        """Generate and assign weights to each chunk in the SampleMetricResult based on length.

        Iterates through each reference result (which must be of type BaseChunkMetricResult)
        and assigns a weight computed with the weigh_edit function to each chunk based on its role:
        TP, FP, FN, TN or variants.

        Args:
            metric_result (SampleMetricResult): The metric result for which weights will be computed.
        """
        for ref_result in metric_result.ref_results:
            # Ensure we're handling chunk-level metrics
            assert isinstance(ref_result, BaseChunkMetricResult)

            # Process true positive chunks.
            for tp_chunk in ref_result.tp_chunks:
                tp_chunk.weight = self.weigh_edit(
                    chunk=tp_chunk,
                    alpha=self.tp_alpha,
                    bias=self.tp_bias,
                    min_value=self.tp_min_value,
                    max_value=self.tp_max_value,
                    reverse=False,
                )
                LOGGER.debug(f"TP: {tp_chunk}")

            for fp_chunk in ref_result.fp_chunks:
                fp_chunk.weight = self.weigh_edit(
                    chunk=fp_chunk,
                    alpha=self.fp_alpha,
                    bias=self.fp_bias,
                    min_value=self.fp_min_value,
                    max_value=self.fp_max_value,
                    reverse=True,
                )
                LOGGER.debug(f"FP: {fp_chunk}")

            if ref_result.fp_ne_chunks:
                for fp_ne_chunk in ref_result.fp_ne_chunks:
                    fp_ne_chunk.weight = self.weigh_edit(
                        chunk=fp_ne_chunk,
                        alpha=self.fp_alpha,
                        bias=self.fp_bias,
                        min_value=self.fp_min_value,
                        max_value=self.fp_max_value,
                        reverse=True,
                    )
                    LOGGER.debug(f"FP_NE: {fp_ne_chunk}")

            if ref_result.fp_un_chunks:
                for fp_un_chunk in ref_result.fp_un_chunks:
                    fp_un_chunk.weight = self.weigh_edit(
                        chunk=fp_un_chunk,
                        alpha=self.fp_alpha,
                        bias=self.fp_bias,
                        min_value=self.fp_min_value,
                        max_value=self.fp_max_value,
                        reverse=True,
                    )
                    LOGGER.debug(f"FP_UN: {fp_un_chunk}")

            for fn_chunk in ref_result.fn_chunks:
                fn_chunk.weight = self.weigh_edit(
                    chunk=fn_chunk,
                    alpha=self.fn_alpha,
                    bias=self.fn_bias,
                    min_value=self.fn_min_value,
                    max_value=self.fn_max_value,
                    reverse=False,
                )
                LOGGER.debug(f"FN: {fn_chunk}")

            for tn_chunk in ref_result.tn_chunks:
                tn_chunk.weight = self.weigh_edit(
                    chunk=tn_chunk,
                    alpha=self.tn_alpha,
                    bias=self.tn_bias,
                    min_value=self.tn_min_value,
                    max_value=self.tn_max_value,
                    reverse=False,
                )
                # LOGGER.debug(f"TN: {tn_chunk}")

    @classmethod
    def weigh_edit(
        cls, chunk: Chunk, alpha: float, bias: float, min_value: float, max_value: float, reverse: bool
    ) -> float:
        """Compute the weight of an edit (chunk) based on its token length.

        The weight is computed using an exponential function scaled by alpha and adjusted by bias.
        If reverse is True, the weight is computed in a manner that a longer edit results in a smaller weight,
        otherwise a longer edit increases the weight. Finally, the weight is clipped to the provided min and max values.

        Args:
            chunk (Chunk): The chunk for which to compute the weight.
            alpha (float): The scale factor.
            bias (float): The bias value.
            min_value (float): The minimum allowed weight.
            max_value (float): The maximum allowed weight.
            reverse (bool): If True, computes weight in reverse manner (used for FP edits).

        Returns:
            float: The computed and clipped weight for the chunk.
        """
        # Determine the effective length of the chunk for weighting purposes.
        edit_len = max(len(chunk.src_tokens), len(chunk.tgt_tokens))
        if reverse:
            weight = alpha * (1 / (1 + (alpha - 1) * np.exp(edit_len - bias)))
        else:
            weight = alpha * (1 / (1 + (alpha - 1) * np.exp(-edit_len + bias)))
        # Clip the weight within the specified boundaries and return as a python float.
        return np.clip(weight, min_value, max_value).item()
