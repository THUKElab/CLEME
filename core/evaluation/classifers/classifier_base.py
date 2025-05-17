from abc import ABC, abstractmethod
from typing import Sequence

from core.data.objects import Edit


class BaseClassifier(ABC):
    """A base abstract Classifier to derive from.

    This abstract class defines the interface that all error classifiers
    must implement, providing a foundation for language-specific classifiers.
    """

    @abstractmethod
    def signature(self) -> str:
        """Returns a signature string identifying the classifier."""
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, source: Sequence, target: Sequence, edit: Edit):
        """Classifies grammatical errors based on the source and target text.

        Args:
            source: The original text sequence.
            target: The corrected text sequence.
            edit: An Edit object containing information about the correction.

        Returns:
            Edit: The same Edit object with updated error type information.
        """
        raise NotImplementedError()
