from abc import ABC, abstractmethod
from enum import Enum
from itertools import chain, groupby
from typing import Any, List, Sequence, Tuple

from core.data.objects import Edit


class MergeStrategy(str, Enum):
    RULES = "rules"
    SPLIT = "all-split"
    MERGE = "all-merge"
    EQUAL = "all-equal"


class BaseMerger(ABC):
    """A base dummy Merger to derive from.

    Args:
        strategy (str): Merging strategy for automatic alignment (default: rules).
            rules: Use a rule-based merging strategy (default)
            all-split: Merge nothing: MSSDI -> M, S, S, D, I
            all-merge: Merge adjacent non-matches: MSSDI -> M, SSDI
            all-equal: Merge adjacent same-type non-matches: MSSDI -> M, SS, D, I
    """

    @abstractmethod
    def signature(self) -> str:
        """Returns a signature for the tokenizer."""
        raise NotImplementedError()

    def __init__(self, strategy: MergeStrategy = MergeStrategy.RULES) -> None:
        self.strategy = strategy

    def __call__(self, source: Sequence, target: Sequence, align_seq: List[Tuple], tgt_index: int = 0) -> List[Edit]:
        """Process source and target sequences using the specified merge strategy.

        Args:
            source: Source sequence to be processed
            target: Target sequence to be processed
            align_seq: Alignment sequence containing edit operations
            tgt_index: Target index (default: 0)

        Returns:
            List of Edit objects representing the merged edits
        """
        if self.strategy == MergeStrategy.RULES:
            # rules: Rule-based merging
            edits = self.get_rule_edits(source, target, align_seq, tgt_index)
        elif self.strategy == MergeStrategy.SPLIT:
            # all-split: Don't merge anything
            edits = self.get_all_split_edits(source, target, align_seq, tgt_index)
        elif self.strategy == MergeStrategy.MERGE:
            # all-merge: Merge all adjacent non-match ops
            edits = self.get_all_merge_edits(source, target, align_seq, tgt_index)
        elif self.strategy == MergeStrategy.EQUAL:
            # all-equal: Merge all edits of the same operation type
            edits = self.get_all_equal_edits(source, target, align_seq, tgt_index)
        else:  # Unknown
            raise ValueError(f"Unknown merging strategy. Please choose from {MergeStrategy}")
        return edits

    def get_all_split_edits(
        self, source: Sequence, target: Sequence, align_seq: List[Tuple], tgt_index: int = 0
    ) -> List[Edit]:
        """Generate edits without merging any operations.

        Implements the 'all-split' strategy where each non-match operation is treated
        as a separate edit.

        Args:
            source: Source sequence to be processed
            target: Target sequence to be processed
            align_seq: Alignment sequence containing edit operations
            tgt_index: Target index (default: 0)

        Returns:
            List of Edit objects with no merging applied
        """
        edits = []
        for align in align_seq:
            if align[0] != "M":
                src_tokens_tok = source[align[1] : align[2]]
                tgt_tokens_tok = target[align[3] : align[4]]
                edit = Edit(
                    tgt_index=tgt_index,
                    src_interval=[align[1], align[2]],
                    tgt_interval=[align[3], align[4]],
                    src_tokens=([x.text for x in src_tokens_tok] if src_tokens_tok else []),
                    tgt_tokens=([x.text for x in tgt_tokens_tok] if tgt_tokens_tok else []),
                    src_tokens_tok=src_tokens_tok,
                    tgt_tokens_tok=tgt_tokens_tok,
                )
                edits.append(edit)
        return edits

    def get_all_merge_edits(
        self,
        source: Sequence,
        target: Sequence,
        align_seq: List[Tuple],
        tgt_index: int = 0,
    ) -> List[Edit]:
        """Generate edits by merging all adjacent non-match operations.

        Implements the 'all-merge' strategy where adjacent non-match operations are
        merged into a single edit, regardless of their type.

        Args:
            source: Source sequence to be processed
            target: Target sequence to be processed
            align_seq: Alignment sequence containing edit operations
            tgt_index: Target index (default: 0)

        Returns:
            List of Edit objects with adjacent non-matches merged
        """
        edits = []
        for op, group in groupby(align_seq, lambda x: True if x[0] == "M" else False):
            if not op:
                merged = self.merge_edits(list(group))[0]
                src_tokens_tok = source[merged[1] : merged[2]]
                tgt_tokens_tok = target[merged[3] : merged[4]]
                edit = Edit(
                    tgt_index=tgt_index,
                    src_interval=[merged[1], merged[2]],
                    tgt_interval=[merged[3], merged[4]],
                    src_tokens=([x.text for x in src_tokens_tok] if src_tokens_tok else []),
                    tgt_tokens=([x.text for x in tgt_tokens_tok] if tgt_tokens_tok else []),
                    src_tokens_tok=src_tokens_tok,
                    tgt_tokens_tok=tgt_tokens_tok,
                )
                edits.append(edit)
        return edits

    def get_all_equal_edits(
        self,
        source: Sequence,
        target: Sequence,
        align_seq: List[Tuple],
        tgt_index: int = 0,
    ) -> List[Edit]:
        """Generate edits by merging adjacent operations of the same type.

        Implements the 'all-equal' strategy where adjacent operations of the same type
        (e.g., all substitutions) are merged into a single edit.

        Args:
            source: Source sequence to be processed
            target: Target sequence to be processed
            align_seq: Alignment sequence containing edit operations
            tgt_index: Target index (default: 0)

        Returns:
            List of Edit objects with same-type operations merged
        """
        edits = []
        for op, group in groupby(align_seq, lambda x: x[0]):
            if op != "M":
                merged = self.merge_edits(list(group))[0]
                src_tokens_tok = source[merged[1] : merged[2]]
                tgt_tokens_tok = target[merged[3] : merged[4]]
                edit = Edit(
                    tgt_index=tgt_index,
                    src_interval=[merged[1], merged[2]],
                    tgt_interval=[merged[3], merged[4]],
                    src_tokens=([x.text for x in src_tokens_tok] if src_tokens_tok else []),
                    tgt_tokens=([x.text for x in tgt_tokens_tok] if tgt_tokens_tok else []),
                    src_tokens_tok=src_tokens_tok,
                    tgt_tokens_tok=tgt_tokens_tok,
                )
                edits.append(edit)
        return edits

    @staticmethod
    def merge_edits(align_seq: List[Any], tag: str = "X") -> List[Any]:
        """Merge the input alignment sequence to a single edit span.

        Args:
            align_seq (List[Any]): Alignment or edit sequence.
            tag (str): Error type. Defaults to "X".

        Returns:
            List[Any]: Merged alignement or edit sequence.
        """
        if not align_seq:
            return align_seq
        elif isinstance(align_seq[0], Tuple):
            return [(tag, align_seq[0][1], align_seq[-1][2], align_seq[0][3], align_seq[-1][4])]
        elif isinstance(align_seq[0], Edit):
            return [
                Edit(
                    tgt_index=align_seq[0].tgt_index,
                    src_interval=[align_seq[0].src_interval[0], align_seq[-1].src_interval[1]],
                    tgt_interval=[align_seq[0].tgt_interval[0], align_seq[-1].tgt_interval[1]],
                    src_tokens=list(chain(*[x.src_tokens for x in align_seq])),
                    tgt_tokens=list(chain(*[x.tgt_tokens for x in align_seq])),
                    src_tokens_tok=list(chain(*[x.src_tokens_tok for x in align_seq])),
                    tgt_tokens_tok=list(chain(*[x.tgt_tokens_tok for x in align_seq])),
                    types=[tag],
                )
            ]

    @abstractmethod
    def get_rule_edits(
        self, source: Sequence, target: Sequence, align_seq: List[Tuple], tgt_index: int = 0
    ) -> List[Any]:
        """Generate edits using rule-based merging strategy.

        This abstract method must be implemented by subclasses to provide specific rule-based merging logic.

        Args:
            source: Source sequence to be processed
            target: Target sequence to be processed
            align_seq: Alignment sequence containing edit operations
            tgt_index: Target index (default: 0)

        Returns:
            List of Edit objects based on rule-based merging
        """
        raise NotImplementedError()
