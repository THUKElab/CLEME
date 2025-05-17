from itertools import combinations, groupby
from re import sub
from string import punctuation
from typing import Any, List, Sequence, Tuple

import spacy.symbols as POS
from rapidfuzz.distance import Indel

from core.data.objects import Edit

from .merger_base import BaseMerger


class MergerEng(BaseMerger):
    """Merger for English (ENG).

    This class implements English-specific rules for merging and splitting edit operations.
    It contains specialized logic for handling English language constructs such as possessives,
    punctuation, whitespace, hyphenation, and part-of-speech based merging decisions.
    """

    # Set of open-class parts of speech (content words)
    OPEN_POS = {POS.ADJ, POS.AUX, POS.ADV, POS.NOUN, POS.VERB}

    def signature(self):
        return "eng"

    def get_rule_edits(
        self, source: Sequence, target: Sequence, align_seq: List[Tuple], tgt_index: int = 0
    ) -> List[Edit]:
        """Generate edits using English-specific rule-based merging strategy.

        This method processes alignment sequences by grouping them into matches (M),
        transformations (T), and other operations (D, I, S), then applies appropriate
        merging or splitting rules to each group.

        Args:
            source: Source sequence of tokens to be processed
            target: Target sequence of tokens to be processed
            align_seq: Alignment sequence containing edit operations
            tgt_index: Target index (default: 0)

        Returns:
            List of Edit objects based on English-specific rule-based merging
        """
        edits = []
        # Split alignment into groups of M, T and rest. (T has a number after it)
        for op, group in groupby(align_seq, lambda x: x[0][0] if x[0][0] in {"M", "T"} else False):
            group = list(group)
            if op == "M":
                # Ignore M
                continue
            elif op == "T":
                # T is always split (transformations)
                for align in group:
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
            else:
                # Process D, I and S subsequence (deletions, insertions, substitutions)
                processed = self._process_seq(
                    source=source,
                    target=target,
                    align_seq=group,
                )
                # Turn the processed sequence into edits
                for merged in processed:
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

    def _process_seq(self, source: Sequence, target: Sequence, align_seq: List[Tuple]) -> List[Any]:
        """Merge edits by applying English-specific heuristic rules.

        This method implements complex logic to determine whether adjacent edit operations
        should be merged or split based on linguistic patterns in English.

        Args:
            source: Source sentence tokens
            target: Target sentence tokens
            align_seq: Sequence of alignment operations to process

        Returns:
            List of merged or split alignment operations
        """
        # Return single alignments as is
        if len(align_seq) <= 1:
            return align_seq

        # Get the ops for the whole sequence
        ops = [op[0] for op in align_seq]

        # Merge all deletions or all insertions (95% of human multi-token edits contain substitutions)
        if set(ops) == {"D"} or set(ops) == {"I"}:
            return self.merge_edits(align_seq)

        # Flag to track if edit includes a content word
        content = False

        # Get indices of all start-end combinations in the sequence: 012 = 01, 02, 12
        combos = list(combinations(range(0, len(align_seq)), 2))

        # Sort combinations starting with the largest spans first
        combos.sort(key=lambda x: x[1] - x[0], reverse=True)

        # Loop through all possible combinations
        for start, end in combos:
            # Ignore ranges that do NOT contain a substitution.
            if "S" not in ops[start : end + 1]:
                continue

            # Get the tokens in source and target. They will never be empty.
            src = source[align_seq[start][1] : align_seq[end][2]]
            tgt = target[align_seq[start][3] : align_seq[end][4]]

            # Handle first token possessive suffixes
            if start == 0 and (src[0].tag_ == "POS" or tgt[0].tag_ == "POS"):
                return [align_seq[0]] + self._process_seq(source=source, target=target, align_seq=align_seq[1:])

            # Merge possessive suffixes such as [friends -> friend 's]
            if src[-1].tag_ == "POS" or tgt[-1].tag_ == "POS":
                return (
                    self._process_seq(source=source, target=target, align_seq=align_seq[: end - 1])
                    + self.merge_edits(align_seq[end - 1 : end + 1])
                    + self._process_seq(source=source, target=target, align_seq=align_seq[end + 1 :])
                )

            # Handle case changes (same word but different capitalization)
            if src[-1].lower == tgt[-1].lower:
                # Merge first insertion or deletion such as [Cat -> The big cat]
                if start == 0 and (
                    (len(src) == 1 and tgt[0].text[0].isupper()) or (len(tgt) == 1 and src[0].text[0].isupper())
                ):
                    return self.merge_edits(align_seq[start : end + 1]) + self._process_seq(
                        source=source, target=target, align_seq=align_seq[end + 1 :]
                    )

                # Merge with previous punctuation such as [, we -> . We], [we -> . We]
                if (len(src) > 1 and is_punct(src[-2])) or (len(tgt) > 1 and is_punct(tgt[-2])):
                    return (
                        self._process_seq(source=source, target=target, align_seq=align_seq[: end - 1])
                        + self.merge_edits(align_seq[end - 1 : end + 1])
                        + self._process_seq(source=source, target=target, align_seq=align_seq[end + 1 :])
                    )

            # Merge whitespace/hyphens such as [acat -> a cat], [sub - way -> subway]
            # Remove hyphens and apostrophes to check if words are essentially the same
            s_str = sub("['-]", "", "".join([tok.lower_ for tok in src]))
            t_str = sub("['-]", "", "".join([tok.lower_ for tok in tgt]))
            if s_str == t_str:
                return (
                    self._process_seq(source=source, target=target, align_seq=align_seq[:start])
                    + self.merge_edits(align_seq[start : end + 1])
                    + self._process_seq(source=source, target=target, align_seq=align_seq[end + 1 :])
                )

            # Merge same POS or auxiliary/infinitive/phrasal verbs:
            # [to eat -> eating], [watch -> look at]
            pos_set = set([tok.pos for tok in src] + [tok.pos for tok in tgt])
            if len(src) != len(tgt) and (len(pos_set) == 1 or pos_set.issubset({POS.AUX, POS.PART, POS.VERB})):
                return (
                    self._process_seq(source=source, target=target, align_seq=align_seq[:start])
                    + self.merge_edits(align_seq[start : end + 1])
                    + self._process_seq(source=source, target=target, align_seq=align_seq[end + 1 :])
                )

            # Split rules take effect when we get to the smallest chunks
            if end - start < 2:
                # Split adjacent substitutions
                if len(src) == len(tgt) == 2:
                    return self._process_seq(
                        source=source, target=target, align_seq=align_seq[: start + 1]
                    ) + self._process_seq(source=source, target=target, align_seq=align_seq[start + 1 :])
                # Split similar substitutions at sequence boundaries
                if (ops[start] == "S" and char_cost(src[0], tgt[0]) > 0.75) or (
                    ops[end] == "S" and char_cost(src[-1], tgt[-1]) > 0.75
                ):
                    return self._process_seq(
                        source=source, target=target, align_seq=align_seq[: start + 1]
                    ) + self._process_seq(source=source, target=target, align_seq=align_seq[start + 1 :])
                # Split final determiners
                if end == len(align_seq) - 1 and (
                    (ops[-1] in {"D", "S"} and src[-1].pos == POS.DET)
                    or (ops[-1] in {"I", "S"} and tgt[-1].pos == POS.DET)
                ):
                    return self._process_seq(
                        source=source,
                        target=target,
                        align_seq=align_seq[:-1],
                    ) + [align_seq[-1]]

            # Set content word flag if any token is a content word
            if not pos_set.isdisjoint(self.OPEN_POS):
                content = True

        # Merge sequences that contain content words, otherwise return as is
        if content:
            return self.merge_edits(align_seq)
        else:
            return align_seq


def is_punct(token) -> bool:
    """Check whether a token is punctuation.

    Determines if a token is punctuation based on its part of speech tag
    or if it appears in the standard punctuation set.

    Args:
        token: The token to check

    Returns:
        bool: True if the token is punctuation, False otherwise
    """
    return token.pos == POS.PUNCT or token.text in punctuation


def char_cost(a, b) -> float:
    """Calculate the character similarity between two tokens.

    Uses the normalized Indel distance to determine how similar two tokens are
    at the character level. Returns a value between 0 and 1, where higher values
    indicate greater similarity.

    Args:
        a: First token
        b: Second token

    Returns:
        float: Similarity score between 0 and 1
    """
    return 1 - Indel.normalized_distance(a.text, b.text)
