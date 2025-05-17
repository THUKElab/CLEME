import argparse

from core.evaluation.metrics import GLEU, DependentCLEME, Errant, IndependentCLEME
from core.evaluation.scorers import HEUOEditScorer, ScorerType
from core.evaluation.weighers import LengthWeigher, WeigherType
from core.readers import M2DataReaderWriter

# Default corpus-level length weigher with the dependence assumption
DEFAULT_LENGTH_WEIGHER_CORPUS_DEPENDENT = LengthWeigher(
    tp_alpha=2.0,
    tp_min_value=0.75,
    tp_max_value=1.25,
    fp_alpha=2.0,
    fp_min_value=0.75,
    fp_max_value=1.25,
    fn_alpha=2.0,
    fn_min_value=0.75,
    fn_max_value=1.25,
)
# Default corpus-level length weigher with the independence assumption
DEFAULT_LENGTH_WEIGHER_CORPUS_INDEPENDENT = LengthWeigher(
    tp_alpha=2.0,
    tp_min_value=0.75,
    tp_max_value=1.25,
    fp_alpha=2.0,
    fp_min_value=0.75,
    fp_max_value=1.25,
    fn_alpha=2.0,
    fn_min_value=0.75,
    fn_max_value=1.25,
)
# Default sentence-level length weigher with the dependence assumption
DEFAULT_LENGTH_WEIGHER_SENTENCE_DEPENDENT = LengthWeigher(
    tp_alpha=10.0,
    tp_min_value=1.0,
    tp_max_value=10.0,
    fp_alpha=10.0,
    fp_min_value=0.25,
    fp_max_value=1.0,
    fn_alpha=10.0,
    fn_min_value=1.0,
    fn_max_value=1.0,
)
# Default sentence-level length weigher with the independence assumption
DEFAULT_LENGTH_WEIGHER_SENTENCE_INDEPENDENT = LengthWeigher(
    tp_alpha=10.0,
    tp_min_value=2.5,
    tp_max_value=10.0,
    fp_alpha=10.0,
    fp_min_value=0.25,
    fp_max_value=1.0,
    fn_alpha=10.0,
    fn_min_value=1.0,
    fn_max_value=1.0,
)

# Default scorer for CLEME2.0
DEFAULT_HEUO_SCORER_CORPUS = HEUOEditScorer(
    factor_hit=0.45, factor_err=0.35, factor_und=0.15, factor_ove=0.05, print_table=True
)
# Default scorer for SentCLEME2.0
DEFAULT_HEUO_SCORER_SENTENCE = HEUOEditScorer(
    factor_hit=0.35, factor_err=0.25, factor_und=0.20, factor_ove=0.20, print_table=True
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate grammatical error correction systems using one of the "
            "supported metrics (ERRANT, GLEU, CLEME-v1 or CLEME-v2).\n\n"
            "Typical usage:\n"
            "python %(prog)s \\ \n"
            "  --ref gold.m2 \\ \n"
            "  --hyp system.m2 \\ \n"
            "  --metric cleme \\ \n"
            "  --level corpus \\ \n"
            "  --assumption dependent"
        ),
        usage="%(prog)s [-h] [options] --ref REF --hyp HYP",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--ref", type=str, required=True, help="Path to the reference M2 file.")
    parser.add_argument("--hyp", type=str, required=True, help="Path to the hypothesis M2 file.")
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help="Three-letter ISO-639-3 language code telling the metric which tokenizer to load (default: eng).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["errant", "gleu", "cleme", "cleme2"],
        help=(
            "Choice of evaluation metric:\n"
            "  errant  – F0.5-score from the ERRANT toolkit (Bryant+ 2017).\n"
            "  gleu    – GLEU sentence GEC metric (Napoles+ 2015).\n"
            "  cleme   – CLEME v1 (Jingheng 2023).\n"
            "  cleme2  – CLEME v2 (Jingheng 2025).\n"
        ),
    )
    parser.add_argument(
        "--level",
        type=str,
        required=False,
        default="corpus",
        choices=["corpus", "sentence"],
        help=(
            "Granularity at which the score is aggregated (default: corpus):\n"
            "  corpus    – compute a single F0.5 score over the whole file.\n"
            "  sentence  – compute an F0.5 score per sentence and then average across sentences."
        ),
    )
    parser.add_argument(
        "--assumption",
        default="dependent",
        choices=["dependent", "independent"],
        help=(
            "Assumption used when mapping edits to reference annotations. "
            "Has an effect *only* for CLEME v1:\n"
            "  dependent    – assume that edits may interact; TP/FP/FN are counted under dependence.\n"
            "  independent  – assume edits are independent of each other."
        ),
    )
    args = parser.parse_args()
    print(f"Args: {args}")

    input_ref_dataset = args.ref
    input_hyp_dataset = args.hyp
    lang = args.lang
    metric = args.metric
    level = args.level
    assumption = args.assumption

    # ------------------------------------------------------------------ #
    # 1. Read reference and hypothesis M2 files                          #
    # ------------------------------------------------------------------ #
    reader = M2DataReaderWriter()
    ref_dataset = reader.read(input_ref_dataset)
    hyp_dataset = reader.read(input_hyp_dataset)

    # ------------------------------------------------------------------ #
    # 2. Instantiate the requested metric                                #
    # ------------------------------------------------------------------ #
    if metric == "errant":
        metric = Errant(lang=lang)

    elif metric == "gleu":
        metric = GLEU(lang=lang)

    elif metric == "cleme":
        if level == "corpus" and assumption == "dependent":
            metric = DependentCLEME(
                lang=lang,
                scorer_type=ScorerType.PRF,
                weigher=DEFAULT_LENGTH_WEIGHER_CORPUS_DEPENDENT,
            )
        elif level == "corpus" and assumption == "independent":
            metric = IndependentCLEME(
                lang=lang,
                scorer_type=ScorerType.PRF,
                weigher=DEFAULT_LENGTH_WEIGHER_CORPUS_INDEPENDENT,
            )
        elif level == "sentence" and assumption == "dependent":
            metric = DependentCLEME(
                lang=lang,
                scorer_type=ScorerType.PRF,
                weigher=DEFAULT_LENGTH_WEIGHER_SENTENCE_DEPENDENT,
            )
        elif level == "sentence" and assumption == "independent":
            metric = IndependentCLEME(
                lang=lang,
                scorer_type=ScorerType.PRF,
                weigher=DEFAULT_LENGTH_WEIGHER_SENTENCE_INDEPENDENT,
            )
        else:
            # This can only happen if new options are added incorrectly
            raise ValueError(f"Unsupported combination: level='{level}', assumption='{assumption}'.")

    elif metric == "cleme2":
        if level == "corpus" and assumption == "dependent":
            metric = DependentCLEME(
                lang=lang,
                scorer=DEFAULT_HEUO_SCORER_CORPUS,
                weigher_type=WeigherType.SIMILARITY,
            )
        elif level == "corpus" and assumption == "independent":
            metric = IndependentCLEME(
                lang=lang,
                scorer=DEFAULT_HEUO_SCORER_CORPUS,
                weigher_type=WeigherType.SIMILARITY,
            )
        elif level == "sentence" and assumption == "dependent":
            metric = DependentCLEME(
                lang=lang,
                scorer=DEFAULT_HEUO_SCORER_SENTENCE,
                weigher_type=WeigherType.SIMILARITY,
            )
        elif level == "sentence" and assumption == "independent":
            metric = IndependentCLEME(
                lang=lang,
                scorer=DEFAULT_HEUO_SCORER_SENTENCE,
                weigher_type=WeigherType.SIMILARITY,
            )
        else:
            # This can only happen if new options are added incorrectly
            raise ValueError(f"Unsupported combination: level='{level}', assumption='{assumption}'.")
    else:
        # The argparse `choices=` guarantees we never get here
        raise RuntimeError(f"Unknown metric: {args.metric}")

    # ------------------------------------------------------------------ #
    # 3. Run evaluation                                                  #
    # ------------------------------------------------------------------ #
    score, _ = metric.evaluate(hyp_dataset, ref_dataset)
    print(score)


if __name__ == "__main__":
    main()

"""
Example: CLEME with the dependence assumption

python main.py \
  --ref demo/examples/conll14.errant \
  --hyp demo/examples/conll14-AMU.errant \
  --metric cleme \
  --level corpus \
  --assumption dependent

"""
