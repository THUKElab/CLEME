from enum import Enum
from typing import Any, Dict, Type

from core.utils import get_logger

from .scorer_base import BaseScorer
from .scorer_gleu import GLEUScorer
from .scorer_heuo import HEUOEditScorer
from .scorer_prf import PRFEditScorer

LOGGER = get_logger(__name__)


class ScorerType(str, Enum):
    PRF = "prf"
    HEUO = "heuo"
    GLEU = "gleu"


SCORER_CLASS: Dict[ScorerType, Type[BaseScorer]] = {
    ScorerType.PRF: PRFEditScorer,
    ScorerType.HEUO: HEUOEditScorer,
    ScorerType.GLEU: GLEUScorer,
}


def get_scorer(scorer_type: ScorerType, **kwargs: Any) -> BaseScorer:
    LOGGER.info(f"Build Scorer: {scorer_type}")
    return SCORER_CLASS[scorer_type](**kwargs)


__all__ = [
    BaseScorer,
    GLEUScorer,
    HEUOEditScorer,
    PRFEditScorer,
    SCORER_CLASS,
    get_scorer,
]
