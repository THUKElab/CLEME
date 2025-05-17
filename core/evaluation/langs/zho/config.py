from enum import Enum

EDIT_TYPE = {
    "M": 0,
    "S": 1,
    "R": 2,
    "W": 3,
}


class TOKENIZATION(str, Enum):
    CHAR = "char"
    WORD = "word"


# TODO
class CharErrorType(str, Enum):
    MISS = "M"
    SUBSTITUE = "S"
