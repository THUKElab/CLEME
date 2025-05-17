import copy
from itertools import chain
from typing import List

from bert_score import BERTScorer

from core.data.objects import Chunk, Sample
from core.utils import get_logger

from ..schema import BaseChunkMetricResult, SampleMetricResult
from .weigher_base import BaseWeigher

LOGGER = get_logger(__name__)


class SimilarityWeigher(BaseWeigher):
    """Compute edit weights using BERTScore to measure similarity.

    This weigher uses an instance of BERTScorer to compute similarity scores between sentences.
    We use these similarity scores to determine an edit's weight. For a given edit, the change
    in similarity (difference between anchor sentence score and the pseudo hypothesis sentence)
    acts as the weight.

    Attributes:
        model_name (str): Name or path for the BERT model (default "bert-base-uncased").
        model_layer (int): Specific layer for BERTScore computation.
        batch_size (int): Number of samples to process per batch (default 128).
        device (str): Computation device (e.g., "cpu", "cuda").
        verbose (bool): Whether to output verbose debugging information.
        show_progress (bool): Whether to display progress during processing.
    """

    DEFAULT_MODEL_NAME = "bert-base-uncased"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_layer: int = None,
        batch_size: int = 128,
        device: str = None,
        verbose: bool = False,
        show_progress: bool = False,
    ) -> None:
        """Initialize the SimilarityWeigher.

        Args:
            model_name (str, optional): BERT model name or path.
            model_layer (int, optional): Layer number for BERTScore.
            batch_size (int, optional): Number of samples per batch.
            device (str, optional): Device identifier.
            verbose (bool, optional): Enable verbose debugging output.
            show_progress (bool, optional): Show progress in processing.
        """
        super().__init__()
        self.model_name = model_name
        self.model_layer = model_layer
        self.batch_size = batch_size
        self.verbose = verbose
        self.show_progress = show_progress

        # Instantiate the BERTScorer model using provided configurations.
        self.model = BERTScorer(
            model_type=self.model_name, num_layers=self.model_layer, batch_size=batch_size, device=device
        )
        LOGGER.info(f"Similarity weigher driven by {self.model_name}, Layers: {self.model._num_layers}")

    @DeprecationWarning
    def __call__(
        self, sample_hyp: Sample, sample_ref: Sample, metric_result: SampleMetricResult, for_chunk: bool = True
    ) -> None:
        """Generate weights for a single SampleMetricResult using BERTScore similarity.

        NOTE: This method is deprecated and not recommended for batch processing. Use
        get_weights_batch instead for multiple samples for efficiency.

        Args:
            sample_hyp (Sample): Hypothesis sample.
            sample_ref (Sample): Reference sample.
            metric_result (SampleMetricResult): Metric result to update.
            for_chunk (bool, optional): Must be True (only chunk-level edits are supported).

        Raises:
            ValueError: If for_chunk is not True.
        """
        if not for_chunk:
            raise ValueError("Only support for_chunk = True")

        src = sample_hyp.source[0]
        for ref, ref_result in zip(sample_ref.target, metric_result.ref_results):
            todo_hyps = []
            # Concatenate all chunks that represent modifications.
            assert isinstance(ref_result, BaseChunkMetricResult)
            todo_chunks = (
                ref_result.tp_chunks
                + ref_result.fp_chunks
                + ref_result.fn_chunks
                + ref_result.tn_chunks
                + ref_result.fp_ne_chunks
                + ref_result.fp_un_chunks
            )
            # Build a pseudo hypothesis sentence for each chunk.
            for chunk in todo_chunks:
                todo_hyps.append(src_chunks_to_text(chunks=sample_hyp.chunks[0][0], chunk_index=chunk.chunk_index))
            # Compute weights for the pseudo hypotheses.
            weights = self._get_weights_sample(src=src, ref=ref, todo_hyps=todo_hyps, verbose=self.verbose)
            # Assign computed weights to each chunk.
            for chunk, weight in zip(todo_chunks, weights):
                chunk.weight = weight
            print(todo_chunks)

    def _get_weights_sample(self, src: str, ref: str, todo_hyps: List[str]) -> List[float]:
        """Generate weights for a single sample using BERTScore.

        Args:
            src (str): The source sentence.
            ref (str): The reference sentence.
            todo_hyps (List[str]): List of pseudo hypothesis sentences.

        Returns:
            List[float]: A weight for each pseudo hypothesis edit. The weight is computed as the
                absolute difference between the anchor score (source) and the hypothesis score.
        """
        # Prepend the source sentence to measure the anchor similarity.
        processed_hyps = [src] + todo_hyps
        processed_refs = [ref] * len(todo_hyps)

        # Get the f-score similarities from BERTScorer.
        fscores = self.model.score(cands=processed_hyps, refs=processed_refs, verbose=self.verbose)[-1]
        # The similarity score for the source sentence.
        anchor = fscores[0].item()
        # Return the absolute differences between the anchor and each pseudo hypothesis.
        return [abs(anchor - x.item()) for x in fscores[1:]]

    def get_weights_batch(
        self, samples_hyp: List[Sample], samples_ref: List[Sample], metric_results: List[List[SampleMetricResult]]
    ) -> None:
        """
        Generate weights in batch for multiple SampleMetricResults.

        This method processes multiple samples together. For each reference, it builds a list of pseudo hypotheses,
        computes the similarity scores in batch via BERTScorer, and assigns computed weights to the corresponding chunks.

        Args:
            samples_hyp (List[Sample]): List of hypothesis samples.
            samples_ref (List[Sample]): List of reference samples.
            metric_results (List[List[SampleMetricResult]]): Nested list of metric results.
        """
        srcs: List[str] = []
        refs: List[str] = []
        todo_hyps_list: List[List[str]] = []
        todo_chunks_list: List[List[Chunk]] = []

        # Compute similarity between sources and targets for the provided reference samples.
        self.get_similarity(samples=samples_ref, metric_results=metric_results)

        for sample_hyp, sample_ref, metric_result in zip(samples_hyp, samples_ref, metric_results):
            for ref_idx, ref_result in enumerate(metric_result.ref_results):
                assert isinstance(ref_result, BaseChunkMetricResult)
                # Set true negative chunks weight to 1.0
                for chunk in ref_result.tn_chunks:
                    chunk.weight = 1.0

                todo_hyps = []
                # Accumulate pseudo hypotheses from TP, FP, FP_NE, FP_UN chunks.
                todo_chunks = (
                    ref_result.tp_chunks + ref_result.fp_chunks + ref_result.fp_ne_chunks + ref_result.fp_un_chunks
                )
                for chunk in todo_chunks:
                    todo_hyps.append(src_chunks_to_text(chunks=sample_hyp.chunks[0][0], chunk_index=chunk.chunk_index))

                # For FN chunks, build pseudo hypothesis using the reference chunk.
                for chunk in ref_result.fn_chunks:
                    todo_hyps.append(
                        src_chunks_to_text(chunks=sample_ref.chunks[0][ref_idx], chunk_index=chunk.chunk_index)
                    )

                # Only process samples with edits.
                if todo_hyps:
                    srcs.append(sample_hyp.source[0])
                    refs.append(sample_ref.target[ref_idx])
                    todo_hyps_list.append(todo_hyps)
                    todo_chunks_list.append(todo_chunks + ref_result.fn_chunks)

        # Batch process the pseudo hypotheses to obtain weights.
        weights_list = self._get_weights_batch(srcs=srcs, refs=refs, todo_hyps_list=todo_hyps_list)
        for chunks, weights in zip(todo_chunks_list, weights_list):
            for chunk, weight in zip(chunks, weights):
                chunk.weight = weight

        if self.verbose:
            for sample_hyp, sample_ref, metric_result in zip(samples_hyp, samples_ref, metric_results):
                for ref, ref_result in zip(sample_ref.target, metric_result.ref_results):
                    print(f"SRC: {sample_hyp.source[0]}")
                    print(f"HYP: {sample_hyp.target[0]}")
                    print(f"REF: {ref}")
                    print(f"TP Chunks: {ref_result.tp_chunks}")
                    print(f"FP Chunks: {ref_result.fp_chunks}")
                    print(f"FN Chunks: {ref_result.fn_chunks}")
                    # print(f"TN Chunks: {ref_result.tn_chunks}")
                    print(f"FP_NE Chunks: {ref_result.fp_ne_chunks}")
                    print(f"FP_UN Chunks: {ref_result.fp_un_chunks}")
                    print()

    def get_weights_batch_v2(
        self,
        samples_hyp: List[Sample],
        samples_ref: List[Sample],
        metric_results: List[List[SampleMetricResult]],
    ) -> None:
        """Alternative method to generate batch weights for SampleMetricResults.

        Constructs candidate sentences by applying TP chunks on the source sentence and reversely applying
        others on the reference sentence. The BERTScorer then provides a similarity measure that is used to
        compute the final edit weights.

        Args:
            samples_hyp (List[Sample]): List of hypothesis samples.
            samples_ref (List[Sample]): List of reference samples.
            metric_results (List[List[SampleMetricResult]]): Nested list of metric results.
        """
        cands: List[str] = []
        refs: List[str] = []
        todo_chunks: List[Chunk] = []

        for sample_hyp, sample_ref, metric_result in zip(samples_hyp, samples_ref, metric_results):
            for ref_idx, ref_result in enumerate(metric_result.ref_results):
                assert isinstance(ref_result, BaseChunkMetricResult)
                # Set true negatives to have weight 1.0.
                for chunk in ref_result.tn_chunks:
                    chunk.weight = 1.0

                for chunk in ref_result.tp_chunks:
                    # Apply the chunk on the source sentence.
                    src = sample_hyp.source[0]
                    src_post = src_chunks_to_text(chunks=sample_hyp.chunks[0][0], chunk_index=chunk.chunk_index)
                    cands.append(src)
                    refs.append(src_post)
                    todo_chunks.append(chunk)

                for chunk in ref_result.fn_chunks + ref_result.fp_un_chunks + ref_result.fp_ne_chunks:
                    # Apply the chunk reversely on the reference sentence.
                    cand_chunks = sample_ref.chunks[0][ref_idx]
                    # Replace the chunk at chunk_index with that from the hypothesis.
                    cand_chunks[chunk.chunk_index] = sample_hyp.chunks[0][0][chunk.chunk_index]
                    # Flatten token lists from candidate chunks.
                    cand_tokens = list(chain(*[x.tgt_tokens for x in cand_chunks]))
                    cand_tokens = list(filter(None, cand_tokens))
                    # Join tokens using a space delimiter.
                    cand = " ".join(cand_tokens)
                    cands.append(cand)
                    refs.append(sample_ref.target[ref_idx])
                    todo_chunks.append(chunk)

        # Compute similarities between candidate sentences and references.
        weights = self.model.score(cands=cands, refs=refs, verbose=self.verbose)[-1]
        for chunk, weight in zip(todo_chunks, weights):
            # Inverse the score to assign weight.
            chunk.weight = 1.0 - weight.item()

        # Combine FP_NE and FP_UN to form all FP chunks.
        for metric_result in metric_results:
            for ref_result in metric_result.ref_results:
                ref_result.fp_chunks = copy.deepcopy(ref_result.fp_ne_chunks + ref_result.fp_un_chunks)
                ref_result.fp_chunks.sort(key=lambda x: x.chunk_index)

        if self.verbose:
            for sample_hyp, sample_ref, metric_result in zip(samples_hyp, samples_ref, metric_results):
                for ref, ref_result in zip(sample_ref.target, metric_result.ref_results):
                    print(f"SRC: {sample_hyp.source[0]}")
                    print(f"HYP: {sample_hyp.target[0]}")
                    print(f"REF: {ref}")
                    print(f"TP Chunks: {ref_result.tp_chunks}")
                    print(f"FP Chunks: {ref_result.fp_chunks}")
                    print(f"FN Chunks: {ref_result.fn_chunks}")
                    # print(f"TN Chunks: {ref_result.tn_chunks}")
                    print(f"FP_NE Chunks: {ref_result.fp_ne_chunks}")
                    print(f"FP_UN Chunks: {ref_result.fp_un_chunks}")
                    print()

    def _get_weights_batch(
        self, srcs: List[str], refs: List[str], todo_hyps_list: List[List[str]]
    ) -> List[List[float]]:
        """Generate weights in batch for multiple pseudo hypothesis lists using BERTScorer.

        For each sample in the batch, the method concatenates the source sentence (to serve as an anchor)
        with each pseudo hypothesis. The difference in similarity (absolute difference of f-scores) between
        the anchor and pseudo hypotheses is used as the weight.

        Args:
            srcs (List[str]): List of source sentences.
            refs (List[str]): List of reference sentences.
            todo_hyps_list (List[List[str]]): List where each element is a list of pseudo hypothesis sentences for the sample.

        Returns:
            List[List[float]]: A list containing lists of weights corresponding to each pseudo hypothesis.

        Raises:
            ValueError: If the lengths of srcs, refs, and todo_hyps_list do not match.
        """
        if len(srcs) != len(refs) != len(todo_hyps_list):
            raise ValueError("The input sentences should consist of the same number")

        processed_hyps = []
        processed_refs = []
        # Indices to capture where each anchor (source) is located in the processed list.
        anchor_indices = []
        for src, ref, todo_hyps in zip(srcs, refs, todo_hyps_list):
            # Replace empty hypothesis sentences with the source sentence.
            for idx, hyp in enumerate(todo_hyps):
                if not hyp:
                    LOGGER.warning(f"Empty Sentences for weigher\n" f"SRC:{src}\nREF:{ref}\nHYPs:{todo_hyps}")
                    todo_hyps[idx] = src

            anchor_indices.append(len(processed_hyps))
            processed_hyps.extend([src] + todo_hyps)
            processed_refs.extend([ref] * (1 + len(todo_hyps)))

        # Compute f-scores for all processed hypotheses.
        fscores = self.model.score(cands=processed_hyps, refs=processed_refs, verbose=self.verbose)[-1]

        # Batch compute weights based on differences with anchor scores.
        weights_list = []
        for idx, anchor_idx in enumerate(anchor_indices):
            num_sent = len(todo_hyps_list[idx])
            anchor = fscores[anchor_idx].item()
            # Calculate absolute difference for each pseudo hypothesis.
            weights = [abs(anchor - x.item()) for x in fscores[anchor_idx + 1 : anchor_idx + 1 + num_sent]]
            weights_list.append(weights)
        return weights_list

    def get_similarity(
        self, samples: List[Sample], metric_results: List[List[SampleMetricResult]], verbose: bool = False
    ) -> List[float]:
        """Compute sentence similarities for a list of samples using BERTScorer.

        For each sample, the method computes the similarity between the source and target sentences.
        The similarity for each reference is stored in the corresponding metric result.

        Args:
            samples (List[Sample]): List of samples on which to compute similarities.
            metric_results (List[List[SampleMetricResult]]): Nested list of metric results to update.
            verbose (bool, optional): If True, prints detailed similarity information.

        Returns:
            List[float]: A list of similarity scores (f-scores) for each (source, target) pair.

        Raises:
            ValueError: If the number of target sentences and metric results do not match.
        """
        srcs, tgts = [], []
        for sample in samples:
            src = sample.source[0]
            for tgt in sample.target:
                if not tgt:
                    LOGGER.warning(f"Empty Target: {sample}")
                srcs.append(src)
                tgts.append(tgt)

        # Compute similarity scores using BERTScorer.
        fscores = self.model.score(cands=srcs, refs=tgts, batch_size=self.batch_size)
        fscores = fscores[-1].tolist()

        idx = 0
        for sample, metric_result in zip(samples, metric_results):
            if len(sample.target) != len(metric_result.ref_results):
                raise ValueError("Unequal results")
            for ref_result in metric_result.ref_results:
                ref_result.sim_src_tgt = fscores[idx]
                idx += 1
        if idx != len(fscores):
            raise ValueError()

        if verbose:
            for src, tgt, fscore in zip(srcs, tgts, fscores):
                print(f"Source: {src}")
                print(f"Target: {tgt}")
                print(f"Similarity: {fscore}")
        return fscores


def src_chunks_to_text(chunks: List[Chunk], chunk_index: int, limiter=" ") -> str:
    """Construct a sentence by merging source and target tokens.

    For the specified chunk_index, the target tokens are used;
    for other chunks, the source tokens are concatenated. This is used to
    reconstruct a pseudo hypothesis sentence.

    Args:
        chunks (List[Chunk]): List of Chunk objects.
        chunk_index (int): The index for which target tokens should be used.
        limiter (str, optional): The delimiter to join tokens (default is a space).

    Returns:
        str: A reconstructed sentence from the chunks.
    """
    tokens = []
    for idx, chunk in enumerate(chunks):
        if idx == chunk_index:
            tokens.extend(chunk.tgt_tokens)
        else:
            tokens.extend(chunk.src_tokens)
    tokens = list(filter(None, tokens))
    return limiter.join(tokens)


def tgt_chunks_to_text(chunks: List[Chunk], chunk_index: int, limiter=" ") -> str:
    """Construct a sentence by merging target and source tokens.

    For the specified chunk_index, the source tokens are used;
    for other chunks, the target tokens are concatenated. This is used to
    reconstruct a pseudo hypothesis sentence in reverse manner.

    Args:
        chunks (List[Chunk]): List of Chunk objects.
        chunk_index (int): The index for which source tokens should be used.
        limiter (str, optional): The delimiter to join tokens (default is a space).

    Returns:
        str: A reconstructed sentence from the chunks.
    """
    tokens = []
    for idx, chunk in enumerate(chunks):
        if idx == chunk_index:
            tokens.extend(chunk.src_tokens)
        else:
            tokens.extend(chunk.tgt_tokens)
    tokens = list(filter(None, tokens))
    return limiter.join(tokens)
