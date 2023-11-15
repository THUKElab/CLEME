from .scorer_corpus import CorpusScorer, CorpusScorerForGLEU
from .scorer_sentence import SentenceScorer, SentenceScorerForGLEU

SCORERS = {
    "corpus": CorpusScorer,
    "corpus_gleu": CorpusScorerForGLEU,
    "sentence": SentenceScorer,
    "sentence_gleu": SentenceScorerForGLEU,
}
