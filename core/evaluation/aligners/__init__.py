from enum import Enum
from typing import Any, Dict, Type

from core.utils import get_logger

from .aligner_base import BaseAligner
from .aligner_eng import AlignerEng
from .aligner_zho import AlignerZho

LOGGER = get_logger(__name__)


class AlignerType(str, Enum):
    ENG = "eng"
    ZHO = "zho"


ALIGNER_CLASS: Dict[AlignerType, Type[BaseAligner]] = {
    AlignerType.ENG: AlignerEng,
    AlignerType.ZHO: AlignerZho,
}


def get_aligner(
    aligner_type: AlignerType,
    del_cost: float = 1.0,
    ins_cost: float = 1.0,
    standard: bool = False,
    brute_force: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> BaseAligner:
    LOGGER.info(f"Build Aligner: {aligner_type}")
    return ALIGNER_CLASS[aligner_type](
        del_cost=del_cost, ins_cost=ins_cost, standard=standard, brute_force=brute_force, verbose=verbose, **kwargs
    )


__all__ = [
    BaseAligner,
    AlignerEng,
    AlignerZho,
    AlignerType,
    ALIGNER_CLASS,
    get_aligner,
]
