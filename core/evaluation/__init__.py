__version__ = "1.0.0"

__description__ = "Evaluation toolkit for Grammatical Error Correction (GEC)"

from .metrics import GLEU, DependentCLEME, Errant, IndependentCLEME, MaxMatch
from .schema import (
    BaseEditMetricResult,
    BaseScorerResult,
    EditScorerResult,
    HEUOEditScorerResult,
    OverallScorerResult,
    SampleMetricResult,
)
from .scorers import ScorerType, get_scorer
from .weighers import WeigherType

__all__ = [
    GLEU,
    DependentCLEME,
    Errant,
    IndependentCLEME,
    MaxMatch,
    BaseEditMetricResult,
    BaseScorerResult,
    EditScorerResult,
    HEUOEditScorerResult,
    SampleMetricResult,
    OverallScorerResult,
    ScorerType,
    get_scorer,
    WeigherType,
]
