from enum import Enum
from typing import Any, Dict, Type

from core.utils import get_logger

from .tokenizer_base import BaseTokenizer
from .tokenizer_eng import TokenizerEng
from .tokenizer_zho import TokenizerZho

LOGGER = get_logger(__name__)


class TokenizerType(str, Enum):
    ENG = "eng"
    ZHO = "zho"


TOKENIZER_CLASS: Dict[TokenizerType, Type[BaseTokenizer]] = {
    TokenizerType.ENG: TokenizerEng,
    TokenizerType.ZHO: TokenizerZho,
}


def get_tokenizer(tokenizer_type: TokenizerType, **kwargs: Any) -> BaseTokenizer:
    LOGGER.info(f"Build Merger: {tokenizer_type}")
    return TOKENIZER_CLASS[tokenizer_type](**kwargs)


__all__ = [
    BaseTokenizer,
    TokenizerEng,
    TokenizerZho,
    TokenizerType,
    TOKENIZER_CLASS,
    get_tokenizer,
]
