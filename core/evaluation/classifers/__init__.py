from enum import Enum
from typing import Any, Dict, Type

from core.utils import get_logger

from .classifier_base import BaseClassifier
from .classifier_eng import ClassifierEng
from .classifier_zho import ClassifierZho

LOGGER = get_logger(__name__)


class ClassifierType(str, Enum):
    ENG = "eng"
    ZHO = "zho"


CLASSIFIER_CLASS: Dict[ClassifierType, Type[BaseClassifier]] = {
    ClassifierType.ENG: ClassifierEng,
    ClassifierType.ZHO: ClassifierZho,
}


def get_classifier(classifier_type: ClassifierType, **kwargs: Any) -> BaseClassifier:
    LOGGER.info(f"Build Classifier: {classifier_type}")
    return CLASSIFIER_CLASS[classifier_type](**kwargs)


__all__ = [
    BaseClassifier,
    ClassifierEng,
    ClassifierZho,
    ClassifierType,
    CLASSIFIER_CLASS,
    get_classifier,
]
