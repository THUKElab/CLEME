"""
The toolkit gec-eval provides hassle-free computation of shareable, comparable,
and reproducible scores for Grammatical Error Correction (GEC).

Inspired by errant.
We implement mainstream GEC metrics composed of the following modules:
1) Tokenizer: Tokenize an input line.
2) Aligner: Align parallel source-target sentences.
3) Merger: Merge adjacent token-level edits through linguistic rules.
4) Classifier: Classify grammatical errors through linguistic rules.
5) Metrics: Acquire intermediate evaluation results.
6) Weigher: Weigh an Edit object through a certain strategy.
7) Scorer: Score evaluation results given by metrics.

Currently supported features:
1) CLEME (ENG, ZHO)
2) ERRANT (ENG, ZHO)
3) GLEU (Language-agnostic)
4) Evaluating at both corpus- and sentence-level for all metrics.

TODO:
1) Run gec_evaluate by command line.
2) Evaluating by MaxMatch.
   MaxMatch 代码太落后，并且没有基于面向对象进行实现，所以整合 MaxMatch 的工作将比较艰难。
3) Multiprocess.
4) Hub: Download evaluating dataset online.
5) CLEME 2.0:
   5.1) Weigh Edits by PLMs.
   5.2) Implement new perspective of GEC evaluation.
6) Sensitivity Analysis for accuracy-based or sentence-level metrics.
   yejh 发现 accuracy-based or sentence-level metrics 在某些场合下表现惊人，但大多数情况拉跨。

NOT TO DO: We will not implement these functions in this project.
1) Evaluating I-Measure.
   I-Measure is cost-expensive, and can be replaced with CLEME.
2) Human Evaluation: Maybe implement it in another project.

"""

import argparse
import sys

from data.readers import ListDataReader, M2DataReaderWriter, ParallelDataReaderWriter
from evaluation.metrics import METRICS


def parse_args():
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--source", "-s", type=str, required=True, help="Source file(s) or line(s)"
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        required=True,
        help="Reference file(s) or line(s)",
    )
    parser.add_argument(
        "--reference-format",
        "-rf",
        type=str,
        choices=["text", "m2"],
        default="text",
        help="Format of reference(s):"
        "text -- each line corresponds to a few references, with each reference seperated by '\\t'"
        "m2 -- M2 format",
    )
    parser.add_argument(
        "--prediction",
        "-p",
        type=str,
        required=True,
        help="Prediction file(s) or line(s)",
    )
    parser.add_argument(
        "--prediction-format",
        "-pf",
        type=str,
        choices=["text", "m2"],
        default="text",
        help="Format of prediction(s):"
        "text -- each line represents a prediction sentence"
        "m2 -- M2 format",
    )
    parser.add_argument(
        "--input-from-command",
        action="store_true",
        help="Input from command (True) or file (False)",
    )
    parser.add_argument(
        "--encoding",
        "-e",
        type=str,
        default="utf-8",
        help="Open text files with specified encoding (Default: %(default)s)",
    )

    metrics = list(METRICS.keys())
    parser.add_argument(
        "--metrics",
        "-m",
        type=str,
        nargs="+",
        choices=metrics,
        help="Space-delimited list of metrics to compute",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        choices=["eng", "zho"],
        required=True,
        help="Language (current support: eng & zho)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["char", "word"],
        default="char",
        help="Granularity of tokenizer (only in Chinese)",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        choices=["corpus", "sentence"],
        default="corpus",
        help="Compute metric for each sentence or the whole corpus",
    )
    parser.add_argument(
        "--enable-tqdm", action="store_true", help="Enable/Disable the progress bar"
    )
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--max-sample", type=int, default=-1)
    parser.add_argument("--max-target", type=int, default=-1)
    parser.add_argument("--tokenizer-eng-name", type=str, default="en_core_web_sm")
    parser.add_argument("--tokenizer-zho-name", type=str, default="LTP/legacy")

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    sys.stdin = open(
        sys.stdin.fileno(),
        mode="r",
        encoding=args.encoding,
        buffering=True,
        newline="\n",
    )
    sys.stdout = open(
        sys.stdout.fileno(), mode="w", encoding=args.encoding, buffering=True
    )

    source, reference, prediction = args.source, args.reference, args.prediction
    read_kwargs = dict(
        max_sample=args.max_sample, max_target=args.max_target, encoding=args.encoding
    )
    if args.input_from_command:
        source = [line.strip() for line in source.split("\n")]
        reference = [line.strip().split("\t") for line in reference.split("\n")]
        prediction = [line.strip() for line in prediction.split("\n")]
        reader = ListDataReader()
        reference_dataset = reader.read(zip(source, reference), **read_kwargs)
        prediction_dataset = reader.read(zip(source, prediction), **read_kwargs)
        reference_datasets, prediction_datasets = [reference_dataset], [
            prediction_dataset
        ]
    else:
        source_files = source.strip().split()
        reference_files = reference.strip().split()
        prediction_files = prediction.strip().split()
        reference_datasets, prediction_datasets = [], []
        reference_reader = (
            M2DataReaderWriter()
            if args.reference_format == "m2"
            else ParallelDataReaderWriter()
        )
        prediction_reader = (
            M2DataReaderWriter()
            if args.prediction_format == "m2"
            else ParallelDataReaderWriter()
        )
        for source_file, reference_file, prediction_file in zip(
            source_files, reference_files, prediction_files
        ):
            reference_file_input = (
                reference_file
                if args.reference_format == "m2"
                else [source_file, reference_file]
            )
            prediction_file_input = (
                prediction_file
                if args.prediction_format == "m2"
                else [source_file, prediction_file]
            )
            reference_dataset = reference_reader.read(
                reference_file_input, **read_kwargs
            )
            prediction_dataset = prediction_reader.read(
                prediction_file_input, **read_kwargs
            )
            reference_datasets.append(reference_dataset), prediction_datasets.append(
                prediction_dataset
            )

    metric_kwargs = dict(
        lang=args.language,
        granularity=args.granularity,
        scorer=args.scorer,
        enable_tqdm=args.enable_tqdm,
        tokenizer_eng_name=args.tokenizer_eng_name,
        tokenizer_zho_name=args.tokenizer_zho_name,
    )
    metrics = []
    for metric_name in args.metrics:
        metric = METRICS[metric_name](**metric_kwargs)
        metrics.append(metric)
    for reference_dataset, prediction_dataset in zip(
        reference_datasets, prediction_datasets
    ):
        for metric in metrics:
            score, results = metric.evaluate(
                prediction_dataset,
                reference_dataset,
                num_workers=args.num_workers,
                refresh_tokens=False,
            )
            print(f"{metric.__class__.__name__} | Metrics: {score}")


if __name__ == "__main__":
    main()
