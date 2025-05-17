from typing import List, Tuple

from pypinyin import Style, pinyin

from core.data.objects import Edit

from ..langs.zho.config import TOKENIZATION
from .classifier_base import BaseClassifier


class ClassifierZho(BaseClassifier):
    """Chinese language classifier for categorizing edit operations in text correction.

    Maps POS tags to error types and classifies edits based on their nature (deletion, insertion, etc.)
    and the linguistic properties of the affected tokens.
    """

    # Suffix words are too few, temporarily classified as "other"
    POS2TYPE = {
        "n": "NOUN",
        "nd": "NOUN",
        "nh": "NOUN-NE",
        "ni": "NOUN-NE",
        "nl": "NOUN-NE",
        "ns": "NOUN-NE",
        "nt": "NOUN-NE",
        "nz": "NOUN-NE",
        "v": "VERB",
        "a": "ADJ",
        "b": "ADJ",
        "c": "CONJ",
        "r": "PRON",
        "d": "ADV",
        "u": "AUX",
        "m": "NUM",
        "p": "PERP",
        "q": "QUAN",
        "wp": "PUNCT",
    }

    def signature(self) -> str:
        return "zho"

    def __init__(self, tokenization: TOKENIZATION = TOKENIZATION.CHAR):
        """Initializes the Chinese classifier with specified tokenization level.

        Args:
            tokenization: Level of analysis - "char" or "word" (word not implemented)
        """
        super().__init__()
        self.tokenization = tokenization

        if self.tokenization == TOKENIZATION.WORD:
            # file_path = os.path.dirname(os.path.abspath(__file__))
            # char_smi = CharFuncs(os.path.join(file_path.replace("modules", ""), "data/char_meta.txt"))
            self.char_smi = None
            raise NotImplementedError

    def __call__(self, source: List[Tuple], target: List[Tuple], edit: Edit) -> Edit:
        """Classifies edit operations into error types.

        Analyzes the source and target text to determine the specific error type
        based on the edit operation and linguistic properties.

        Args:
            source: Information about the source tokens
            target: Information about the target tokens
            edit: Edit operation to be classified

        Returns:
            Edit operation with classified error types
        """
        error_type = edit.types[0]
        src_span = " ".join(edit.src_tokens)
        tgt_span = " ".join(edit.tgt_tokens)

        if error_type[0] == "T":
            edit.types = ["T"]
        elif error_type[0] == "D":
            if self.tokenization == "word":
                if len(src_span) > 1:
                    # Word group redundancy temporarily classified as OTHER
                    edit.types = ["R:OTHER"]
                else:
                    pos = self.POS2TYPE.get(source[edit.src_interval[0]][1])
                    pos = "NOUN" if pos == "NOUN-NE" else pos
                    pos = "MC" if tgt_span == "[缺失成分]" else pos
                    edit.types = ["U:{:s}".format(pos)]
            else:
                edit.types = ["U"]
        elif error_type[0] == "I":
            if self.tokenization == "word":
                if len(tgt_span) > 1:
                    # Word group omission temporarily classified as OTHER
                    edit.types = ["M:OTHER"]
                else:
                    pos = self.POS2TYPE.get(target[edit.tgt_interval[0]][1])
                    pos = "NOUN" if pos == "NOUN-NE" else pos
                    pos = "MC" if tgt_span == "[缺失成分]" else pos
                    edit.types = ["M:{:s}".format(pos)]
            else:
                edit.types = ["M"]
        elif error_type[0] == "S":
            if self.tokenization == "word":
                if check_spell_error(src_span.replace(" ", ""), tgt_span.replace(" ", ""), char_smi=self.char_smi):
                    edit.types = ["S:SPELL"]
                    # Todo 暂且不单独区分命名实体拼写错误
                    # if edit[4] - edit[3] > 1:
                    #     cor = Correction("S:SPELL:COMMON", tgt_span, (edit[1], edit[2]))
                    # else:
                    #     pos = self.get_pos_type(tgt[edit[3]][1])
                    #     if pos == "NOUN-NE":  # 命名实体拼写有误
                    #         cor = Correction("S:SPELL:NE", tgt_span, (edit[1], edit[2]))
                    #     else:  # 普通词语拼写有误
                    #         cor = Correction("S:SPELL:COMMON", tgt_span, (edit[1], edit[2]))
                else:
                    if len(tgt_span) > 1:
                        # Word group replacement temporarily classified as OTHER
                        edit.types = ["S:OTHER"]
                    else:
                        pos = self.POS2TYPE.get(target[edit.tgt_interval[0]][1])
                        pos = "NOUN" if pos == "NOUN-NE" else pos
                        pos = "MC" if tgt_span == "[缺失成分]" else pos
                        edit.types = ["S:{:s}".format(pos)]
            else:
                edit.types = ["S"]
        return edit


def check_spell_error(src_span: str, tgt_span: str, char_smi, threshold: float = 0.8) -> bool:
    """Determines if the difference between source and target spans is a spelling error.

    Compares characters based on shape and pronunciation similarities to identify
    potential spelling mistakes in Chinese text.

    Args:
        src_span: Source text span with potential error
        tgt_span: Target (corrected) text span
        threshold: Similarity threshold for determining spelling errors

    Returns:
        True if the difference is likely a spelling error, False otherwise
    """
    if len(src_span) != len(tgt_span):
        return False
    src_chars = [ch for ch in src_span]
    tgt_chars = [ch for ch in tgt_span]

    if sorted(src_chars) == sorted(tgt_chars):
        # Character transposition within word
        return True

    for src_char, tgt_char in zip(src_chars, tgt_chars):
        if src_char != tgt_char:
            if src_char not in char_smi.data or tgt_char not in char_smi.data:
                return False
            v_sim = char_smi.shape_similarity(src_char, tgt_char)
            p_sim = char_smi.pronunciation_similarity(src_char, tgt_char)
            if v_sim + p_sim < threshold and not (
                set(pinyin(src_char, style=Style.NORMAL, heteronym=True)[0])
                & set(pinyin(tgt_char, style=Style.NORMAL, heteronym=True)[0])
            ):
                return False
    return True
