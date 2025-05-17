from functools import lru_cache
from typing import List, Union

import spacy
from spacy.tokens import Doc

from .tokenizer_base import BaseTokenizer

DEFAULT_SPACY_MODEL = "en_core_web_sm"


class TokenizerEng(BaseTokenizer):
    """English language tokenizer implementation.

    Provides tokenization functionality for English text using either spaCy
    or simple space-based tokenization.
    """

    def __init__(self, enable_spacy: bool = False, spacy_model_name: str = DEFAULT_SPACY_MODEL) -> None:
        """Initialize the English tokenizer.

        Args:
            enable_spacy (bool): Whether to use spaCy for tokenization
            spacy_model_name (str): Name of the spaCy model to use
        """
        self.enable_sapcy = enable_spacy
        self.spacy_model = spacy.load(spacy_model_name, disable=["ner"])

    def signature(self) -> str:
        return "eng"

    @property
    def delimiter(self) -> str:
        """Returns the delimiter between tokens."""
        return " "

    @lru_cache(maxsize=2**16)
    def __call__(self, content: str, plain: bool = False):
        """Tokenize English text using spaCy or space-based tokenization.

        Args:
            content (str): The text to tokenize
            plain (bool): If True, return only token texts

        Returns:
            Union[List[str], Doc]: Tokenized text as a list of strings or spaCy Doc
        """
        if self.enable_sapcy:
            # tokenize by spacy
            text = self.spacy_model(content)
        else:
            # tokenize by spaces
            tokens = content.strip().split()
            text = Doc(self.spacy_model.vocab, tokens)
            # self.spacy_model.tagger(text)
            # self.spacy_model.parser(text)
            self.spacy_model(text)

        if plain:
            return [token.text for token in text]
        return text

    def detokenize(self, tokens: Union[Doc, List[str]]) -> str:
        """Convert tokens back into a string.

        Args:
            tokens (Union[Doc, List[str]]): The tokens to join

        Returns:
            str: The detokenized text
        """
        if isinstance(tokens, List):
            return self.delimiter.join(tokens)
        return tokens.text

    def destroy(self):
        """Clean up resources used by the tokenizer."""
        del self._tokenizer
        self._tokenizer = None
