from enum import Enum
from typing import Any, Dict, Type

from core.utils import get_logger

from .weigher_base import BaseWeigher
from .weigher_length import LengthWeigher
from .weigher_similarity import SimilarityWeigher

LOGGER = get_logger(__name__)


class WeigherType(str, Enum):
    NONE = "none"
    LENGTH = "length"
    SIMILARITY = "similarity"


WEIGHER_CLASS: Dict[WeigherType, Type[BaseWeigher]] = {
    WeigherType.NONE: BaseWeigher,
    WeigherType.LENGTH: LengthWeigher,
    WeigherType.SIMILARITY: SimilarityWeigher,
}


def get_weigher(weigher_type: WeigherType, **kwargs: Any) -> BaseWeigher:
    weigher = WEIGHER_CLASS[weigher_type](**kwargs)
    LOGGER.info(f"Build Weigher: {weigher_type}: {weigher}")
    return weigher


__all__ = [
    BaseWeigher,
    LengthWeigher,
    SimilarityWeigher,
    WeigherType,
    WEIGHER_CLASS,
    get_weigher,
]
