import os
from typing import List, Tuple

from ..constants import PUNCTUATION
from ..langs.zho.config import TOKENIZATION
from .aligner_base import BaseAligner

DEFAULT_DIR_RESOURCE = os.path.join(os.path.dirname(__file__), "../langs/zho")


class AlignerZho(BaseAligner):
    """Chinese text aligner that implements specialized alignment algorithms for Chinese language.

    This class extends BaseAligner with Chinese-specific alignment features including
    semantic, phonetic, and character-based similarity measures. It loads Chinese
    linguistic resources like confusion sets and semantic dictionaries (CiLin).
    """

    def signature(self) -> str:
        return "zho"

    def __init__(
        self,
        del_cost: float = 1.0,
        ins_cost: float = 1.0,
        standard: bool = False,
        brute_force: bool = False,
        verbose: bool = False,
        tokenization: TOKENIZATION = TOKENIZATION.CHAR,
    ) -> None:
        """Initialize the Chinese aligner with specific parameters and resources.

        Loads Chinese-specific resources including confusion dictionary and CiLin semantic
        dictionary for enhanced alignment capabilities.

        Args:
            del_cost (float): Cost for deletion operations. Defaults to 1.0.
            ins_cost (float): Cost for insertion operations. Defaults to 1.0.
            standard (bool): Whether to use standard alignment. Defaults to False.
            brute_force (bool): Whether to use brute force search. Defaults to False.
            verbose (bool): Whether to print verbose information. Defaults to False.
            tokenization (TOKENIZATION): Tokenization method (CHAR or WORD). Defaults to CHAR.
        """

        super().__init__(
            standard=standard,
            del_cost=del_cost,
            ins_cost=ins_cost,
            brute_force=brute_force,
            verbose=verbose,
        )
        self.tokenization = tokenization
        self._open_pos = {}

        # Load resource: Chinese confustion set
        self.confusion_dict = {}
        path_confustion = os.path.join(DEFAULT_DIR_RESOURCE, "confusion_dict.txt")
        with open(path_confustion, "r", encoding="utf-8") as f:
            for line in f:
                li = line.strip().split(" ")
                self.confusion_dict[li[0]] = li[1:]

        # Load resource: Chinese cilin
        self.semantic_dict = {}
        path_cilin = os.path.join(DEFAULT_DIR_RESOURCE, "cilin.txt")
        with open(path_cilin, "r", encoding="gbk") as f:
            for line in f:
                code, *words = line.strip().split(" ")
                for word in words:
                    self.semantic_dict[word] = code

    def get_sub_cost(self, src_token: Tuple, tgt_token: Tuple) -> float:
        """Calculate the cost of Linguistic Damerau-Levenshtein substitution.

        Computes a linguistically-informed substitution cost between source and target
        tokens based on semantic, POS, and character similarities. The approach differs
        based on whether word-level or character-level tokenization is used.

        Args:
            src_token (Tuple): Source token (token, POS, pinyin)
            tgt_token (Tuple): Target token (token, POS, pinyin)

        Returns:
            float: A linguistic cost between 0 < x < 2
        """
        if src_token[0] == tgt_token[0]:
            return 0
        if self.tokenization == TOKENIZATION.WORD:
            # For word-level tokenization, utilize additional POS information
            semantic_cost = self._get_semantic_cost(src_token[0], tgt_token[0]) / 6.0
            pos_cost = self._get_pos_cost(src_token[1], tgt_token[1])
            char_cost = self._get_char_cost(src_token[0], tgt_token[0], src_token[2], tgt_token[2])
            return semantic_cost + pos_cost + char_cost
        else:
            # For character-level tokenization, use character meaning (from CiLin) and visual similarity
            semantic_cost = self._get_semantic_cost(src_token[0], tgt_token[0]) / 6.0
            if src_token[0] in PUNCTUATION and tgt_token[0] in PUNCTUATION:
                pos_cost = 0.0
            elif src_token[0] not in PUNCTUATION and tgt_token[0] not in PUNCTUATION:
                pos_cost = 0.25
            else:
                pos_cost = 0.499
            char_cost = self._get_char_cost(src_token[0], tgt_token[0], src_token[2], tgt_token[2])
            return semantic_cost + char_cost + pos_cost

    def _get_semantic_cost(self, a: str, b: str) -> int:
        """Calculate substitution cost based on semantic information.

        Args:
            a (str): First token
            b (str): Second token

        Returns:
            int: Substitution edit cost based on semantic similarity
        """
        a_class = self._get_semantic_class(a)
        b_class = self._get_semantic_class(b)
        # unknown class, default to 1
        if a_class is None or b_class is None:
            return 4
        elif a_class == b_class:
            return 0
        else:
            return 2 * (3 - self._get_class_diff(a_class, b_class))

    def _get_semantic_class(self, token: str) -> Tuple:
        """Acquire the semantic class of a token from the CiLin dictionary.

        Based on the paper: "Improved-Edit-Distance Kernel for Chinese Relation Extraction"

        Args:
            token (str): The token to look up

        Returns:
            Tuple: A tuple of (high, mid, low) semantic categories, or None if not found
        """
        if token in self.semantic_dict:
            code = self.semantic_dict[token]
            high, mid, low = code[0], code[1], code[2:4]
            return high, mid, low
        return None

    @staticmethod
    def _get_class_diff(a_class: Tuple[str], b_class: Tuple[str]) -> int:
        """Calculate the semantic category difference between two words according to "DaCiLin".

        Args:
            a_class (Tuple[str]): Semantic class of first word
            b_class (Tuple[str]): Semantic class of second word

        Returns:
            int: Difference value where:
                d == 3 for equivalent semantics
                d == 0 for completely different semantics
        """
        d = sum([a == b for a, b in zip(a_class, b_class)])
        return d

    def _get_pos_cost(self, a_pos: str, b_pos: str) -> float:
        """Calculate the edit distance cost based on Part-of-Speech tags.

        Args:
            a_pos (str): POS tag of first word
            b_pos (str): POS tag of second word

        Returns:
            float: Cost based on POS similarity
        """
        if a_pos == b_pos:
            return 0
        elif a_pos in self._open_pos and b_pos in self._open_pos:
            return 0.25
        else:
            return 0.499

    def _get_char_cost(self, a: str, b: str, pinyin_a: List, pinyin_b: List) -> float:
        """Calculate the edit distance cost based on character similarity.

        Args:
            a (str): First character/word
            b (str): Second character/word
            pinyin_a: Pinyin representation of first character/word
            pinyin_b: Pinyin representation of second character/word

        Returns:
            float: Cost based on character similarity
        """
        if not (check_all_chinese(a) and check_all_chinese(b)):
            return 0.5
        if a == b:
            return 0.0

        if len(a) > len(b):
            a, b = b, a
            pinyin_a, pinyin_b = pinyin_b, pinyin_a
        return self._get_spell_cost(a, b, pinyin_a, pinyin_b)

    def _get_spell_cost(self, a: str, b: str, pinyin_a: List, pinyin_b: List) -> float:
        """Calculate the spelling similarity between two words based on glyph and phonetic similarity.

        Args:
            a (str): First token (length less than or equal to b)
            b (str): Second token
            pinyin_a: Pinyin representation of first token
            pinyin_b: Pinyin representation of second token

        Returns:
            float: Substitution cost based on spelling similarity
        """
        count = 0
        for i in range(len(a)):
            for j in range(len(b)):
                if (
                    a[i] == b[j]
                    or (set(pinyin_a) & set(pinyin_b))
                    or (b[j] in self.confusion_dict.keys() and a[i] in self.confusion_dict[b[j]])
                    or (a[i] in self.confusion_dict.keys() and b[j] in self.confusion_dict[a[i]])
                ):
                    count += 1
                    break
        return (len(a) - count) / (len(a) * 2)


def check_all_chinese(word):
    """Check if a word consists entirely of Chinese characters.

    Args:
        word (str): The word to check

    Returns:
        bool: True if all characters are Chinese, False otherwise
    """
    return all(["\u4e00" <= ch <= "\u9fff" for ch in word])
