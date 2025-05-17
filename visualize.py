import argparse

from core.evaluation.metrics import GLEU, DependentCLEME, Errant, IndependentCLEME
from core.evaluation.scorers import HEUOEditScorer, ScorerType
from core.evaluation.weighers import LengthWeigher, WeigherType
from core.readers import M2DataReaderWriter


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the chunk partition process.",
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
    parser.add_argument("--max_sample", type=int, default=-1, help="Maximum number of samples")
    parser.add_argument("--output_visualize", type=str, required=True, help="The output file of visualization")

    args = parser.parse_args()
    print(f"Args: {args}")

    input_ref_dataset = args.ref
    input_hyp_dataset = args.hyp
    lang = args.lang
    max_sample = args.max_sample
    output_visualize = args.output_visualize

    # ------------------------------------------------------------------ #
    # 1. Read reference and hypothesis M2 files                          #
    # ------------------------------------------------------------------ #
    reader = M2DataReaderWriter()
    ref_dataset = reader.read(input_ref_dataset, max_sample=max_sample)
    hyp_dataset = reader.read(input_hyp_dataset, max_sample=max_sample)

    # ------------------------------------------------------------------ #
    # 2. Visualize the chunk partition process                           #
    # ------------------------------------------------------------------ #
    metric = DependentCLEME(lang=lang, output_visualize=output_visualize)
    metric.prepare_datasets(dataset_hyp=hyp_dataset, dataset_ref=ref_dataset)


if __name__ == "__main__":
    main()

"""
Example:

python visualize.py \
  --ref demo/examples/conll14.errant \
  --hyp demo/examples/conll14-AMU.errant \
  --max_sample 10 \
  --output_visualize visualization.txt
"""
