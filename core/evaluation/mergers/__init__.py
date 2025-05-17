from enum import Enum
from typing import Any, Dict, Type

from core.utils import get_logger

from .merger_base import BaseMerger, MergeStrategy
from .merger_eng import MergerEng
from .merger_zho import MergerZho

LOGGER = get_logger(__name__)


class MergerType(str, Enum):
    ENG = "eng"
    ZHO = "zho"


MERGER_CLASS: Dict[MergerType, Type[BaseMerger]] = {
    MergerType.ENG: MergerEng,
    MergerType.ZHO: MergerZho,
}


def get_merger(merger_type: MergerType, strategy: MergeStrategy = None, **kwargs: Any) -> BaseMerger:
    LOGGER.info(f"Build Merger: {merger_type}")
    strategy = strategy or MergeStrategy.RULES
    return MERGER_CLASS[merger_type](strategy=strategy, **kwargs)


__all__ = [
    BaseMerger,
    MergerEng,
    MergerZho,
    MergerType,
    MERGER_CLASS,
    MergeStrategy,
    get_merger,
]
