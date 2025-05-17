from .base import BaseMetric
from .cleme.cleme_dependent import DependentCLEME
from .cleme.cleme_independent import IndependentCLEME
from .errant import Errant
from .gleu import GLEU
from .maxmatch import MaxMatch

METRICS = {
    "errant": Errant,
    "gleu": GLEU,
    "maxmatch": MaxMatch,
    "cleme_independent": IndependentCLEME,
    "cleme_dependent": DependentCLEME,
}

__all__ = [BaseMetric, DependentCLEME, IndependentCLEME, Errant, GLEU, MaxMatch]
