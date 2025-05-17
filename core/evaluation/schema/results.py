from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from core.data.objects import Chunk, Edit


class BaseMetricResult(BaseModel):
    pass


class SampleMetricResult(BaseModel):
    ref_results: List[BaseMetricResult] = Field(default_factory=list)


class BaseEditMetricResult(BaseMetricResult):
    tp_edits: List[Edit] = Field(default_factory=list)
    fp_edits: List[Edit] = Field(default_factory=list)
    fn_edits: List[Edit] = Field(default_factory=list)
    tn_edits: List[Edit] = Field(default_factory=list)
    fp_ne_edits: List[Edit] = Field(default_factory=list)
    fp_un_edits: List[Edit] = Field(default_factory=list)


class BaseChunkMetricResult(BaseMetricResult):
    sim_src_tgt: float = Field(default=None)
    tp_chunks: List[Chunk] = Field(default_factory=list)
    fp_chunks: List[Chunk] = Field(default_factory=list)
    fn_chunks: List[Chunk] = Field(default_factory=list)
    tn_chunks: List[Chunk] = Field(default_factory=list)
    fp_ne_chunks: Optional[List[Chunk]] = Field(default=None)
    fp_un_chunks: Optional[List[Chunk]] = Field(default=None)


class BaseScorerResult(BaseModel):
    pass


class OverallScorerResult(BaseModel):
    num_sample: int = Field(default=None)
    scores: Dict[str, Union[BaseScorerResult, float]] = Field(default_factory=dict)


class EditScorerResult(BaseScorerResult):
    tp: float = Field(default=0.0)
    fp: float = Field(default=0.0)
    fn: float = Field(default=0.0)
    tn: float = Field(default=0.0)

    p: float = Field(default=None)
    r: float = Field(default=None)
    f: float = Field(default=None)
    acc: float = Field(default=None)


class HEUOEditScorerResult(BaseScorerResult):
    tp: float = Field(default=0)
    fp: float = Field(default=0)
    fn: float = Field(default=0)
    tn: float = Field(default=0)
    fp_ne: float = Field(default=0)
    fp_un: float = Field(default=0)
    necessary: float = Field(default=0)
    unnecessary: float = Field(default=0)

    com: float = Field(default=None)
    hit: float = Field(default=None)
    err: float = Field(default=None)
    und: float = Field(default=None)
    ove: float = Field(default=None)


class GLEUScorerResult(BaseScorerResult):
    score: float = Field()
    std: Optional[float] = Field(default=None)
    ci: Optional[float] = Field(default=None)
