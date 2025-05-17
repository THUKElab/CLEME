"""`Scorer` is an abstract class that enforces the implementation of a set
of abstract methods. This way, a correctly implemented metric will work
seamlessly with the rest of the codebase.

Scorer                                        # Abstract Scorer Class
  ├── SystemScorer                            # Corpus-level Scorer
  ├── SentenceScorer                          # Sentence-level Scorer
  ├── SentenceScorerForGLEU                   # Sentence-level Scorer for GLEU
  └── SentenceScorerForGLEU                   # Sentence-level Scorer for GLEU
"""

import sys
from abc import ABC, abstractmethod
from typing import Any, TextIO

from pydantic import BaseModel, Field

from ..schema import BaseScorerResult, OverallScorerResult


class BaseScorer(ABC, BaseModel):
    """Abstract base class for all scorers.

    A scorer is responsible for computing evaluation metrics at either the corpus-level
    or sentence-level. By enforcing implementation of key functions, this class ensures
    that any derived metric scorer is compatible with the rest of the evaluation codebase.

    Args:
        table_print (bool): Flag to indicate whether the result should be printed as a table.
    """

    table_print: bool = Field(default=True)

    def __call__(self, **kwargs: Any) -> OverallScorerResult:
        """Make instances callable to evaluate and score.

        This function simply calls the score() method with the given keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the score() method.

        Returns:
            OverallScorerResult: The aggregated evaluation results.
        """

        return self.score(**kwargs)

    @abstractmethod
    def score(self, **kwargs: Any) -> OverallScorerResult:
        """Compute the overall score given the evaluation inputs.

        This abstract method should be implemented by derived scorer classes.

        Returns:
            OverallScorerResult: The overall scoring result.
        """
        raise NotImplementedError

    @abstractmethod
    def score_corpus(self, **kwargs: Any) -> BaseScorerResult:
        """Compute corpus-level scoring results.

        This function must be implemented by child classes.

        Returns:
            BaseScorerResult: The corpus-level score result.
        """
        raise NotImplementedError

    @abstractmethod
    def score_sentence(self, **kwargs: Any) -> BaseScorerResult:
        """Compute sentence-level scoring results.

        This function must be implemented by child classes.

        Returns:
            BaseScorerResult: The sentence-level score result.
        """

        raise NotImplementedError

    @abstractmethod
    def print_result_table(self, result: OverallScorerResult, sout: TextIO = sys.stdout, **kwargs: Any) -> None:
        """Visualize the overall results in a table format.

        Derived classes should implement a method to print scores in a human‐readable table.

        Args:
            result (OverallScorerResult): The result object with a dictionary of scores.
            sout (TextIO, optional): The output stream to which the table will be written.
        """
        raise NotImplementedError
