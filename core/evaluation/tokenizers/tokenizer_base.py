from abc import ABC, abstractmethod
from typing import Sequence


class BaseTokenizer(ABC):
    """Tokenizer provides the functionality of tokenization of text."""

    def signature(self) -> str:
        """Return a signature for the tokenizer."""
        raise NotImplementedError()

    @property
    def delimiter(self) -> str:
        """Returns the delimiter between tokens."""
        return " "

    @abstractmethod
    def __call__(self, content: str, plain: bool = False) -> Sequence:
        """Tokenize an input content with the tokenizer.

        Typically, self.detokenize(self.__call__(line)) == line

        Args:
            content (str): Input content to tokenize.
            plain (bool): Return the result as a list.

        Returns:
            Sequence[Any]: Tokenized tokens.
        """
        raise NotImplementedError()

    @abstractmethod
    def detokenize(self, tokens: Sequence) -> str:
        """Detokenize tokens into a string.

        Args:
            tokens (Sequence): Tokenized input.

        Returns:
            str: Detokenized string.
        """
        raise NotImplementedError()

    def destroy(self) -> None:
        """Clean up resources used by the tokenizer."""
        pass
