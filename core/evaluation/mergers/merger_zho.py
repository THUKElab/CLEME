from itertools import groupby
from typing import Any, List, Tuple

import Levenshtein

from core.data.objects import Edit

from ..constants import PUNCTUATION
from ..langs.zho.config import TOKENIZATION
from .merger_base import BaseMerger, MergeStrategy


class MergerZho(BaseMerger):
    """Merger for Chinese (ZHO).

    This class implements Chinese-specific rules for merging and splitting edit operations.
    It contains specialized logic for handling Chinese language constructs, including
    character-level and word-level processing, and heuristic-based edit merging.
    """

    def signature(self) -> str:
        return "zho"

    def __init__(
        self,
        strategy: MergeStrategy = MergeStrategy.RULES,
        tokenization: TOKENIZATION = TOKENIZATION.CHAR,
        heuristic: bool = True,
        **kwargs: Any,
    ):
        """Initialize the Chinese merger with specific parameters.

        Args:
            strategy: The merging strategy to use (default: MergeStrategy.RULES)
            tokenization: Tokenzation level, either 'char' or 'word' (default: 'char')
            heuristic: Whether to apply heuristic rules for edit refinement (default: True)
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(strategy, **kwargs)
        self.not_merge_token = [punct for punct in PUNCTUATION]
        self.tokenization = tokenization
        self.heuristic = heuristic

    def get_rule_edits(
        self,
        source: List[Tuple],
        target: List[Tuple],
        align_seq: List[Tuple],
        tgt_index: int = 0,
        verbose: bool = False,
    ) -> List[Edit]:
        """Generate edits using Chinese-specific rule-based merging strategy.

        This method processes alignment sequences by grouping them into matches (M),
        transformations (T), and other operations (D, I, S), then applies appropriate
        merging or splitting rules to each group, with additional Chinese-specific
        heuristic refinements.

        Args:
            source: Source sequence of tokens to be processed
            target: Target sequence of tokens to be processed
            align_seq: Alignment sequence containing edit operations
            tgt_index: Target index (default: 0)
            verbose: Whether to print detailed debugging information (default: False)

        Returns:
            List of Edit objects based on Chinese-specific rule-based merging
        """

        def build_edit(align):
            """Helper function to build an Edit object from alignment information.

            Args:
                align: Alignment tuple containing operation type and indices

            Returns:
                Edit: A new Edit object with the specified properties
            """
            src_tokens_tok = source[align[1] : align[2]]
            tgt_tokens_tok = target[align[3] : align[4]]
            return Edit(
                tgt_index=tgt_index,
                src_interval=[align[1], align[2]],
                tgt_interval=[align[3], align[4]],
                src_tokens=[x[0] for x in src_tokens_tok] if src_tokens_tok else [],
                tgt_tokens=[x[0] for x in tgt_tokens_tok] if tgt_tokens_tok else [],
                src_tokens_tok=src_tokens_tok,
                tgt_tokens_tok=tgt_tokens_tok,
                types=[align[0]],
            )

        edits = []
        # Split alignment into groups of M, T and rest. (T has a number after it)
        for op, group in groupby(align_seq, lambda x: x[0][0] if x[0][0] in {"M", "T"} else False):
            group = list(group)
            if op == "T":
                # T is always split
                for align in group:
                    edits.append(build_edit(align))
            else:
                # Process D, I and S subsequence
                processed = self._process_seq(source=source, target=target, align_seq=group)
                # Turn the processed sequence into edits
                for merged in processed:
                    edits.append(build_edit(merged))
        if not self.heuristic:
            return edits

        i, filtered_edits = 0, []
        # src_tokens = [x[0] for x in source]
        # tgt_tokens = [x[0] for x in target]
        while i < len(edits):
            e1_op = edits[i].types[0]
            if i < len(edits) - 2:
                e2_op = edits[i + 1].types[0]
                e3_op = edits[i + 2].types[0]
                # Find "S M S" patterns
                #   S     M     S
                # 冬阴功  对  外国人
                # 外国人  对  冬阴功
                if e1_op == "S" and e2_op == "M" and e3_op == "S":
                    w1 = "".join(edits[i].src_tokens)
                    w2 = "".join(edits[i].tgt_tokens)
                    w3 = "".join(edits[i + 2].src_tokens)
                    w4 = "".join(edits[i + 2].tgt_tokens)
                    if min([len(w1), len(w2), len(w3), len(w4)]) == 1:
                        if w1 == w4 and w2 == w3:
                            group = [edits[i], edits[i + 1], edits[i + 2]]
                            processed = self.merge_edits(
                                group, "T" + str(edits[i + 2].src_interval[1] - edits[i].src_interval[0])
                            )
                            for merged in processed:
                                filtered_edits.append(merged)
                            i += 3
                        else:
                            filtered_edits.append(edits[i])
                            i += 1
                    else:
                        if Levenshtein.distance(w1, w4) <= 1 and Levenshtein.distance(w2, w3) <= 1:
                            # Merge edits with small Levenshtein distance (likely related)
                            group = [edits[i], edits[i + 1], edits[i + 2]]
                            processed = self.merge_edits(
                                group, "T" + str(edits[i + 2].src_interval[1] - edits[i].src_interval[0])
                            )
                            for merged in processed:
                                filtered_edits.append(merged)
                            i += 3
                        else:
                            filtered_edits.append(edits[i])
                            i += 1
                # Find "D M I" or "I M D" patterns
                #   D        M             I
                # 旅游 去   陌生 的   地方
                #     去   陌生 的   地方  旅游
                elif (e1_op == "D" and (e2_op == "M" or e2_op.startswith("T")) and e3_op == "I") or (
                    e1_op == "I" and (e2_op == "M" or e2_op.startswith("T")) and e3_op == "D"
                ):
                    if e1_op == "D":
                        delete_token = edits[i].src_tokens
                        insert_token = edits[i + 2].tgt_tokens
                    else:
                        delete_token = edits[i + 2].src_tokens
                        insert_token = edits[i].tgt_tokens
                    a, b = "".join(delete_token), "".join(insert_token)
                    if len(a) < len(b):
                        a, b = b, a
                    if a not in PUNCTUATION and b not in PUNCTUATION and len(a) - len(b) <= 1:
                        if len(b) == 1:
                            if a == b:
                                # Merge when tokens are identical
                                group = [edits[i], edits[i + 1], edits[i + 2]]
                                processed = self.merge_edits(
                                    group, "T" + str(edits[i + 2].src_interval[1] - edits[i].src_interval[0])
                                )
                                for merged in processed:
                                    filtered_edits.append(merged)
                                i += 3
                            else:
                                filtered_edits.append(edits[i])
                                i += 1
                        else:
                            if Levenshtein.distance(a, b) <= 1 or (len(a) == len(b) and self._check_revolve(a, b)):
                                # Merge when tokens are similar or are rotations of each other
                                group = [edits[i], edits[i + 1], edits[i + 2]]
                                processed = self.merge_edits(
                                    group, "T" + str(edits[i + 2].src_interval[1] - edits[i].src_interval[0])
                                )
                                for merged in processed:
                                    filtered_edits.append(merged)
                                i += 3
                            else:
                                filtered_edits.append(edits[i])
                                i += 1
                    else:
                        filtered_edits.append(edits[i])
                        i += 1
                else:
                    if e1_op != "M":
                        filtered_edits.append(edits[i])
                    i += 1
            else:
                if e1_op != "M":
                    filtered_edits.append(edits[i])
                i += 1
        # In rare cases with word-level tokenization, the following error can occur:
        # M     D   S       M
        # 有    時  住      上層
        # 有        時住    上層
        # Which results in S: 時住 --> 時住
        # We need to filter this case out
        second_filter = []
        for edit in filtered_edits:
            # Avoid mismatches caused by tokenization errors
            span1 = "".join(edit.src_tokens)
            span2 = "".join(edit.tgt_tokens)

            if span1 != span2:
                if edit.types[0] == "S":
                    b = True
                    # In rare cases with word-level tokenization, the following error can occur:
                    # S       I     I       M
                    # 负责任               老师
                    # 负     责任   的     老师
                    # Which results in S: 负责任 --> 负 责任 的
                    # We need to convert this edit to I: --> 的

                    # Check for overlap at the beginning
                    common_str = ""
                    tmp_new_start_1 = edit.src_interval[0]
                    for i in range(edit.src_interval[0], edit.src_interval[1]):
                        if not span2.startswith(common_str + source[i][0]):
                            break
                        common_str += source[i][0]
                        tmp_new_start_1 = i + 1
                    new_start_1, new_start_2 = (edit.src_interval[0], edit.tgt_interval[0])
                    if common_str:
                        tmp_str = ""
                        for i in range(edit.tgt_interval[0], edit.tgt_interval[1]):
                            tmp_str += target[i][0]
                            if tmp_str == common_str:
                                new_start_1, new_start_2 = tmp_new_start_1, i + 1
                                # second_filter.append(("S", new_start_1, edit[2], i + 1, edit[4]))
                                b = False
                                break
                            elif len(tmp_str) > len(common_str):
                                break
                    # Check for overlap at the end
                    common_str = ""
                    new_end_1, new_end_2 = edit.src_interval[1], edit.tgt_interval[1]
                    tmp_new_end_1 = edit.src_interval[1]
                    for i in reversed(range(new_start_1, edit.src_interval[1])):
                        if not span2.endswith(source[i][0] + common_str):
                            break
                        common_str = source[i][0] + common_str
                        tmp_new_end_1 = i
                    if common_str:
                        tmp_str = ""
                        for i in reversed(range(new_start_2, edit.tgt_interval[1])):
                            tmp_str = target[i][0] + tmp_str
                            if tmp_str == common_str:
                                new_end_1, new_end_2 = tmp_new_end_1, i
                                b = False
                                break
                            elif len(tmp_str) > len(common_str):
                                break
                    if b:
                        second_filter.append(edit)
                    else:
                        if new_start_1 == new_end_1:
                            new_edit = ("I", new_start_1, new_end_1, new_start_2, new_end_2)
                        elif new_start_2 == new_end_2:
                            new_edit = ("D", new_start_1, new_end_1, new_start_2, new_end_2)
                        else:
                            new_edit = ("S", new_start_1, new_end_1, new_start_2, new_end_2)
                        second_filter.append(build_edit(new_edit))
                else:
                    second_filter.append(edit)

        if verbose:
            print("========== Parallels ==========")
            print("".join([x[0] for x in source]))
            print("".join([x[0] for x in target]))
            print("========== Results ==========")
            for edit in second_filter:
                op = edit[0]
                s = " ".join(edit.src_tokens)
                t = " ".join(edit.tgt_tokens)
                print(f"{op}:\t{s}\t-->\t{t}")
            print("========== Infos ==========")
            print(str(source))
            print(str(target))
        return second_filter

    @staticmethod
    def _check_revolve(span_a: str, span_b: str) -> bool:
        """Check if span_b is a rotation of span_a.

        This method determines if one string is a rotation of another by
        concatenating span_a with itself and checking if span_b is a substring.

        Args:
            span_a: First string to compare
            span_b: Second string to compare

        Returns:
            bool: True if span_b is a rotation of span_a, False otherwise
        """
        span_a = span_a + span_a
        return span_b in span_a

    def _process_seq(self, source: List[Tuple], target: List[Tuple], align_seq: List[Any]) -> List[Any]:
        """Merge edits by applying Chinese-specific heuristic rules.

        This method implements logic to determine whether adjacent edit operations
        should be merged or split based on linguistic patterns in Chinese.

        Args:
            source: Source sentence tokens
            target: Target sentence tokens
            align_seq: Sequence of alignment operations to process

        Returns:
            List of merged or split alignment operations
        """
        # TODO: If the edit contains punctuation, don't merge it with other edits
        # TODO: If the edit contains missing component labels, don't merge it with other edits
        if len(align_seq) <= 1:
            return align_seq

        # Get the ops for the whole sequence
        ops = [op[0] for op in align_seq]

        # Merge all deletion or insertion operations (95% of human multi-token edits contain substitutions)
        if set(ops) == {"D"} or set(ops) == {"I"}:
            return self.merge_edits(align_seq, set(ops).pop())

        # Do not merge deletion-insertion patterns
        if set(ops) == {"D", "I"} or set(ops) == {"I", "D"}:
            return align_seq

        if set(ops) == {"S"}:
            if self.tokenization == TOKENIZATION.WORD:
                return align_seq
            else:
                return self.merge_edits(align_seq, "S")

        if set(ops) == {"M"}:
            return self.merge_edits(align_seq, "M")

        return self.merge_edits(align_seq, "S")
