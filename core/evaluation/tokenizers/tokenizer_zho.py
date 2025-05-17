from functools import lru_cache
from typing import Any, List

from ltp import LTP
from pypinyin import Style, lazy_pinyin, pinyin

from core.utils import get_logger, remove_space

from ..langs.zho.config import TOKENIZATION
from .tokenizer_base import BaseTokenizer

LOGGER = get_logger(__name__)


class TokenizerZho(BaseTokenizer):
    """Chinese language tokenizer implementation.

    Provides tokenization functionality for Chinese text at either character
    or word level, with optional pronunciation information.
    """

    def __init__(self, tokenization: TOKENIZATION = TOKENIZATION.CHAR, **kwargs: Any) -> None:
        """Initialize the Chinese tokenizer.

        Args:
            tokenization (TOKENIZATION): Tokenization level (character or word)
            **kwargs: Additional parameters for the tokenizer
        """
        self._tokenizer = None
        self.tokenization = tokenization

        if tokenization == TOKENIZATION.WORD:
            model_name_or_path = kwargs.get("pretrained_model_name_or_path", "LTP/small")
            self._tokenizer = LTP(pretrained_model_name_or_path=model_name_or_path)
            self._tokenizer.add_words(words=["[缺失成分]"])
            LOGGER.info(f"{self.__class__.__name__} initialize LTP model: {model_name_or_path}")

    def signature(self) -> str:
        return "zho"

    @property
    def delimiter(self) -> str:
        return ""

    @lru_cache(maxsize=2**16)
    def __call__(self, content: str, add_extra_info: bool = True, plain: bool = False):
        """Tokenize Chinese text at character or word level.

        Args:
            content (str): Chinese content to tokenize
            add_extra_info (bool): Whether to include additional information like pinyin
            plain (bool): If True, return only token texts

        Returns:
            Any: Tokenized tokens with optional additional information
        """
        if self.tokenization == TOKENIZATION.CHAR:
            tokens = self._tokenize_char(content, add_extra_info=add_extra_info)
        elif self.tokenization == TOKENIZATION.WORD:
            tokens = self._tokenize_word(content, add_extra_info=add_extra_info)

        if plain:
            return [token[0] for token in tokens]
        return tokens

    def detokenize(self, tokens: List[Any]) -> str:
        """Convert tokens back into a string.

        Args:
            tokens (List[Any]): The tokens to join

        Returns:
            str: The detokenized text
        """
        return "".join([x[0] for x in tokens])

    def _tokenize_char(cls, content: str, add_extra_info: bool = True):
        """Tokenize Chinese content by characters.

        Args:
            content (str): Input Chinese content
            add_extra_info (bool): Whether to acquire extra info like pinyin

        Raises:
            ValueError: If input character has no pinyin

        Returns:
            List[Any]: Tokenized results with character, POS tag, and pinyin
        """
        chars = [x for x in remove_space(content.strip())]
        # "[缺失成分]" (missing component) is a single special token
        chars = " ".join(chars).replace("[ 缺 失 成 分 ]", "[缺失成分]").split()

        if not add_extra_info:
            return chars

        chars_info = []
        for char in chars:
            py = pinyin(char, style=Style.NORMAL, heteronym=True)
            if not len(py):
                # Raise Error if char is empty
                raise ValueError(f"Unknown pinyin ({char}) from {chars}")
            chars_info.append((char, "unk", py[0]))
        return chars_info

    def _tokenize_word(self, content: str, add_extra_info: bool = True) -> List[Any]:
        """Tokenize Chinese content by words.

        Args:
            content (str): Input Chinese text
            add_extra_info (bool): Whether to acquire extra info like POS and pinyin

        Returns:
            List[Any]: Tokenized results with word, POS tag, and pinyin information
        """
        content = remove_space(content.strip())
        if not add_extra_info:
            (words,) = self._tokenizer.pipeline([content], tasks=["cws"]).to_tuple()
            return words[0]
        words, words_pos = self._tokenizer.pipeline([content], tasks=["cws", "pos"]).to_tuple()

        words_info = []
        for word, pos in zip(words, words_pos):
            py = [lazy_pinyin(x) for x in word]
            words_info.append(list(zip(word, pos, py)))
        return words_info
