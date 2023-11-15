"""
CLEME provides unbiased scores for Grammatical Error Correction (GEC) tasks.

References:
[1] CLEME: Debiasing Multi-reference Evaluation for Grammatical Error Correction [EMNLP 2023]
[2] Automatic Annotation and Evaluation of Error Types for Grammatical Error Correction [ACL 2017]

"""
import argparse
import os
import sys

sys.path.append(f"{os.path.dirname(__file__)}/../")
from cleme.cleme import DependentChunkMetric, IndependentChunkMetric
from cleme.data import M2DataReader
from cleme.constants import *


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate correction performance using CLEME.",
        usage="%(prog)s [-h] [options] -ref REF -hyp HYP",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--ref", help="A reference M2 file.", required=True)
    parser.add_argument("--hyp", help="A hypothesis M2 file.", required=True)
    parser.add_argument("--vis", help="Whether visualize evaluate process as tables.", action="store_true")
    parser.add_argument(
        "--level", default="corpus", choices=["corpus", "sentence"],
        help="Evaluation level (default: `corpus`)\n"
             "corpus: compute an F0.5 score over the entire dataset.\n"
             "sentence: compute an average F0.5 score on each sentence."
    )
    parser.add_argument(
        "--scorer", default="dependent", choices=["dependent", "independent"],
        help="Scorer to determine TP, FP, FN, FN edit counts (default: `dependent`)\n"
             "dependent: determine edit counts based on the correction dependence assumption.\n"
             "independent: determine edit counts based on the correction independence assumption."
    )
    args = parser.parse_args()

    # Read M2 file
    reader = M2DataReader()
    dataset_ref = reader.read(args.ref)
    dataset_hyp = reader.read(args.hyp)

    # Define metric
    if args.level == "corpus" and args.scorer == "dependent":
        config = DEFAULT_CONFIG_CORPUS_DEPENDENT
        metric = DependentChunkMetric(scorer=args.level, weigher_config=config)
    elif args.level == "corpus" and args.scorer == "independent":
        config = DEFAULT_CONFIG_CORPUS_INDEPENDENT
        metric = IndependentChunkMetric(scorer=args.level, weigher_config=config)
    elif args.level == "sentence" and args.scorer == "dependent":
        config = DEFAULT_CONFIG_SENTENCE_DEPENDENT
        metric = DependentChunkMetric(scorer=args.level, weigher_config=config)
    elif args.level == "sentence" and args.scorer == "independent":
        config = DEFAULT_CONFIG_SENTENCE_INDEPENDENT
        metric = IndependentChunkMetric(scorer=args.level, weigher_config=config)
    else:
        raise ValueError(f"Undefined configurations for `level` and `scorer`.")

    # Evaluate pipeline
    score, results = metric.evaluate(dataset_hyp, dataset_ref)
    print(score)

    # Visualize
    if args.vis:  metric.visualize(dataset_ref, dataset_hyp)


if __name__ == "__main__":
    main()

"""
Quick start:

######################### Evaluate AMU system #########################
python scripts/evaluate.py --ref tests/examples/conll14.errant  --hyp tests/examples/conll14-AMU.errant

11/15/2023 14:18:57 - INFO - cleme.data -   <class 'cleme.data.M2DataReader'>: Read 1312 samples from conll14.errant.
11/15/2023 14:18:57 - INFO - cleme.data -   <class 'cleme.data.M2DataReader'>: Read 1312 samples from conll14-AMU.errant.
11/15/2023 14:18:57 - WARNING - cleme.cleme -   Ignore empty reference: " || 
11/15/2023 14:18:58 - WARNING - cleme.cleme -   Ignore empty reference: Once his or her family or relatives know that the person does not have the relation with them . || 
11/15/2023 14:18:58 - WARNING - cleme.cleme -   Ignore empty reference: Tears showed nothing . || 
11/15/2023 14:18:58 - INFO - cleme.cleme -   avg_chunk_len_correct=4.16, avg_chunk_len_incorrect=1.79
11/15/2023 14:18:58 - INFO - cleme.cleme -   length_weight_tp: SigmoidWeigher(alpha=2.0, bias=1.7905201847760595, min_value=0.75, max_value=1.25, reverse=False)
11/15/2023 14:18:58 - INFO - cleme.cleme -   length_weight_fp: SigmoidWeigher(alpha=2.0, bias=1.7905201847760595, min_value=0.75, max_value=1.25, reverse=True)
11/15/2023 14:18:58 - INFO - cleme.cleme -   length_weight_fn: SigmoidWeigher(alpha=2.0, bias=1.7905201847760595, min_value=0.75, max_value=1.25, reverse=False)
11/15/2023 14:18:58 - INFO - cleme.cleme -   <class 'cleme.cleme.DependentChunkMetric'> Total samples: 1312, Total time: 0.729 seconds; Preparation time: 0.545
{'num_sample': 1312, 'F': 0.2514, 'Acc': 0.7634, 'P': 0.2645, 'R': 0.2097, 'tp': 313.51, 'fp': 871.8, 'fn': 1181.71, 'tn': 6312.0}

######################### Visualize Demo #########################
python scripts/evaluate.py  --ref tests/examples/demo.errant  --hyp tests/examples/demo-AMU.errant  --vis

11/15/2023 14:24:14 - INFO - cleme.data -   <class 'cleme.data.M2DataReader'>: Read 2 samples from demo.errant.
11/15/2023 14:24:14 - INFO - cleme.data -   <class 'cleme.data.M2DataReader'>: Read 2 samples from demo-AMU.errant.
11/15/2023 14:24:14 - INFO - cleme.cleme -   avg_chunk_len_correct=2.09, avg_chunk_len_incorrect=1.4
11/15/2023 14:24:14 - INFO - cleme.cleme -   length_weight_tp: SigmoidWeigher(alpha=2.0, bias=1.4, min_value=0.75, max_value=1.25, reverse=False)
11/15/2023 14:24:14 - INFO - cleme.cleme -   length_weight_fp: SigmoidWeigher(alpha=2.0, bias=1.4, min_value=0.75, max_value=1.25, reverse=True)
11/15/2023 14:24:14 - INFO - cleme.cleme -   length_weight_fn: SigmoidWeigher(alpha=2.0, bias=1.4, min_value=0.75, max_value=1.25, reverse=False)
11/15/2023 14:24:14 - INFO - cleme.cleme -   <class 'cleme.cleme.DependentChunkMetric'> Total samples: 2, Total time: 0.002 seconds; Preparation time: 0.001
{'num_sample': 2, 'F': 0.2711, 'Acc': 0.8002, 'P': 0.2293, 'R': 1.0, 'tp': 0.8, 'fp': 2.7, 'fn': 0, 'tn': 10.0}
╒════════════╤═══════════════════════════════════════╕
│ sentence   │ chunk-0                               │
╞════════════╪═══════════════════════════════════════╡
│ source     │ Keeping the Secret of Genetic Testing │
├────────────┼───────────────────────────────────────┤
│ target-0   │ Keeping the Secret of Genetic Testing │
├────────────┼───────────────────────────────────────┤
│ target-1   │ Keeping the Secret of Genetic Testing │
├────────────┼───────────────────────────────────────┤
│ target-2   │ Keeping the Secret of Genetic Testing │
╘════════════╧═══════════════════════════════════════╛
╒════════════╤═════════════╤════════════════╤═══════════╤═════════════╤═════════════════╤═════════════╤═══════════╤═════════════╤═════════════╤═════════════╤════════════╤══════════════╤═════════════╕
│ sentence   │ chunk-0     │ chunk-1 *      │ chunk-2   │ chunk-3 *   │ chunk-4         │ chunk-5 *   │ chunk-6   │ chunk-7 *   │ chunk-8     │ chunk-9 *   │ chunk-10   │ chunk-11 *   │ chunk-12    │
╞════════════╪═════════════╪════════════════╪═══════════╪═════════════╪═════════════════╪═════════════╪═══════════╪═════════════╪═════════════╪═════════════╪════════════╪══════════════╪═════════════╡
│ source     │ When we are │ diagonosed out │ with      │             │ certain genetic │ disease     │ , are we  │ suppose     │ to disclose │ this result │ to         │ our          │ relatives ? │
├────────────┼─────────────┼────────────────┼───────────┼─────────────┼─────────────────┼─────────────┼───────────┼─────────────┼─────────────┼─────────────┼────────────┼──────────────┼─────────────┤
│ target-0   │ When we are │ diagnosed      │ with      │             │ certain genetic │ diseases    │ , are we  │ suppose     │ to disclose │ this result │ to         │ our          │ relatives ? │
├────────────┼─────────────┼────────────────┼───────────┼─────────────┼─────────────────┼─────────────┼───────────┼─────────────┼─────────────┼─────────────┼────────────┼──────────────┼─────────────┤
│ target-1   │ When we are │ diagnosed      │ with      │ a           │ certain genetic │ disease     │ , are we  │ supposed    │ to disclose │ this result │ to         │ our          │ relatives ? │
├────────────┼─────────────┼────────────────┼───────────┼─────────────┼─────────────────┼─────────────┼───────────┼─────────────┼─────────────┼─────────────┼────────────┼──────────────┼─────────────┤
│ target-2   │ When we are │ diagnosed out  │ with      │             │ certain genetic │ diseases    │ , are we  │ suppose     │ to disclose │ the results │ to         │ their        │ relatives ? │
╘════════════╧═════════════╧════════════════╧═══════════╧═════════════╧═════════════════╧═════════════╧═══════════╧═════════════╧═════════════╧═════════════╧════════════╧══════════════╧═════════════╛

"""
