<div align="center">

# CLEME & CLEME2.0: Unbiased and Interpretable Evaluation for Grammatical Error Correction

</div>

This repository hosts the official implementation and resources for our papers on Grammatical Error Correction (GEC) evaluation:

1.  **[CLEME: Debiasing Multi-reference Evaluation for Grammatical Error Correction](https://aclanthology.org/2023.emnlp-main.378/)** (EMNLP 2023 Main Conference)
2.  **[CLEME2.0: Towards More Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction](https://arxiv.org/abs/2407.00934)** (ACL 2025 Main Conference)

**CLEME** and its successor, **CLEME2.0,** are reference-based metrics designed to evaluate GEC systems at the chunk level. CLEME focuses on providing unbiased F-scores in multi-reference settings, while CLEME2.0 enhances this by offering a more interpretable breakdown of system performance.

## 1. Overview

Evaluating Grammatical Error Correction (GEC) systems is inherently challenging due to the subjective nature of natural language and the possibility of multiple reasonable corrections.

Traditional metrics can introduce bias, especially in multi-reference scenarios, and often lack the granularity to provide deep insights into a system's specific strengths and weaknesses.

* **CLEME** addresses the bias in multi-reference evaluation by transforming sentences into chunk sequences with consistent boundaries across the source, hypothesis, and references. This allows for a fairer comparison and a more robust F-score calculation, particularly by leveraging the correction independence assumption at the chunk level.

* **CLEME2.0** builds upon CLEME's chunking foundation to offer enhanced interpretability. It disentangles the GEC system performance into four fundamental aspects:

  1.  **Hit-correction (HC)**: Successfully identified and corrected errors. High HC indicates the system is effective at finding and fixing actual errors.
  2.  **Wrong-correction (WC)**: Errors identified but corrected improperly. High WC suggests the system identifies error locations but applies incorrect fixes. This points to issues with the system's correction generation.
  3.  **Under-correction (UC)**: Errors missed by the system. High UC means the system misses many errors. This indicates problems with error detection.
  4.  **Over-correction (OC)**: Unnecessary edits made to correct text, potentially altering meaning. High OC signifies the system makes too many unnecessary changes, potentially harming fluency or altering meaning. This is a key aspect of faithfulness.

  An ideal GEC system would have a high HC score and low WC, UC, and OC scores. The aggregated score provides a single measure, but analyzing the individual components is crucial for understanding system behavior.

  CLEME2.0 also introduces edit weighting techniques (similarity-based and LLM-based) to better reflect the semantic impact of different corrections.

Together, these metrics aim to provide a more accurate, unbiased, and insightful evaluation framework for the GEC community.

## 2. Key Features

### CLEME

*   **Unbiased Evaluation**: Mitigates bias in multi-reference settings by ensuring consistent edit boundaries through chunk-level processing.
*   **Correction Independence Assumption**: Computes F-scores based on the observation that grammatical error corrections in terms of chunks can be approximately independent.
*   **Visualization**: Supports table-based visualization of the evaluation process for detailed analysis.
*   **Multi-lingual Support**: Currently supports English and Chinese. It is adaptable to other languages with appropriate M2/ERRANT-style annotations.

### CLEME2.0

*   **Interpretable Diagnostics**: Provides scores for four distinct aspects of GEC performance (hit-correction, wrong-correction, under-correction, over-correction), enabling detailed error analysis.
*   **Faithfulness & Grammaticality Assessment**: Helps distinguish between issues related to grammaticality (hit, wrong, under-correction) and faithfulness (over-correction).
*   **Advanced Edit Weighting**: Incorporates similarity-based weighting to capture the varying significance of different edits.
*   **Builds on CLEME**: Leverages the robust chunking mechanism of CLEME for its analysis.

## 3. Requirements and Installation

1. **Python**: Python >= 3.10

2. **ERRANT (or language-specific variants)**: CLEME relies on ERRANT (or its variants) for parsing M2 files and initial edit extraction.
   Install the appropriate version for your target language:

   ```bash
   pip install errant  # For English
   ```

   For other languages, please refer to their specific installation guides:

   | Language                                                     | Link                                                         |
   | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | English                                                      | [ERRANT](https://github.com/chrisjbryant/errant)             |
   | Arabic                                                       | [arabic_error_type_annotation](https://github.com/CAMeL-Lab/arabic_error_type_annotation) |
   | Chinese                                                      | [ChERRANT](https://github.com/HillZhang1999/MuCGEC/blob/main/scorers/ChERRANT) |
   | Czech                                                        | [errant_czech](https://github.com/ufal/errant_czech)         |
   | German                                                       | [ERRANT-German](https://github.com/adrianeboyd/boyd-wnut2018) |
   | Greek                                                        | [ELERRANT](https://github.com/katkorre/elerrant)             |
   | Hindi                                                        | [hindi_grammar_correction](https://github.com/s-ankur/hindi_grammar_correction) |
   | Korean                                                       | [Standard_Korean_GEC](https://github.com/soyoung97/Standard_Korean_GEC) |
   | Russian                                                      | [ERRANT-Russian](https://github.com/Askinkaty/errant)        |
   | Turkish                                                      | [ERRANT-TR](https://github.com/harunuz/erranttr)             |
   | *We recommend using the latest version of ERRANT for potential speed improvements, although ERRANT v2.3.3 was used for the experiments in the CLEME paper.* |                                                              |

3. **Clone this repository and install CLEME**:

   ```bash
   git clone https://github.com/THUKElab/CLEME.git
   
   cd CLEME
   ```

## 4. Usage

### 4.1 Command Line Interface (CLI)

#### CLEME Evaluation

To evaluate a system's hypothesis against references using CLEME:

Example: Corpus-level CLEME with the dependence assumption.

```bash
python main.py \
  --ref demo/examples/conll14.errant \
  --hyp demo/examples/conll14-AMU.errant \
  --metric cleme \
  --level corpus \
  --assumption dependent
```

Example output:

```json
{
  'num_sample': 1312, 
  'F': 0.2514,
  'Acc': 0.7634,
  'P': 0.2645,
  'R': 0.2097,
  'tp': 313.51,
  'fp': 871.8,
  'fn': 1181.71,
  'tn': 6312.0
}
```

To visualize the CLEME evaluation process as tables for detailed comparison:

```bash
python visualize.py \
  --ref demo/examples/conll14.errant \
  --hyp demo/examples/conll14-AMU.errant \
  --max_sample 10 \
  --output_visualize visualization.txt
```

Example output:

![](img\visualization.png)

- Chunks with stars indicate changed text segments.
- Chunks without starts indicate unchanged text segments.

#### CLEME2.0 Evaluation

To evaluate using CLEME2.0 and get disentangled scores:

Example: CLEME2.0 with the dependence assumption.

```bash
python main.py \
  --ref demo/examples/conll14.errant \
  --hyp demo/examples/conll14-AMU.errant \
  --metric cleme2 \
  --level corpus \
  --assumption dependent
```

*Note: Specific output format and options for CLEME2.0 (e.g., for edit weighting) can be found using python main.py --help.*

### 4.2 Python API

#### CLEME Evaluation

```python
from core.evaluation.metrics import GLEU, DependentCLEME, Errant, IndependentCLEME
from core.evaluation.scorers import HEUOEditScorer, ScorerType
from core.evaluation.weighers import LengthWeigher, WeigherType
from core.readers import M2DataReaderWriter

# Initialize reader and metric
reader = M2Reader()

# For CLEME (correction dependence assumption, as in original paper for some comparisons)
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

# Read M2 files
dataset_ref = reader.read("demo/examples/demo.errant")
dataset_hyp = reader.read("demo/examples/demo-AMU.errant")

print(f"Number of reference sentences: {len(dataset_ref)}")
print(f"Number of hypothesis sentences: {len(dataset_hyp)}")

# Build metric
metric = DependentCLEME(
    lang=lang,
    scorer_type=ScorerType.PRF,
    weigher=DEFAULT_LENGTH_WEIGHER_CORPUS_DEPENDENT,
)

# Evaluate
score, _ = metric.evaluate(hyp_dataset, ref_dataset)
print(f"==================== CLEME Evaluation Results ====================")
print(score)
```

#### CLEME2.0 Evaluation

```python
from core.evaluation.metrics import GLEU, DependentCLEME, Errant, IndependentCLEME
from core.evaluation.scorers import HEUOEditScorer, ScorerType
from core.evaluation.weighers import LengthWeigher, WeigherType
from core.readers import M2DataReaderWriter

# Initialize reader and metric
reader = M2Reader()

# Default scorer for CLEME2.0
DEFAULT_HEUO_SCORER_CORPUS = HEUOEditScorer(
    factor_hit=0.45, factor_err=0.35, factor_und=0.15, factor_ove=0.05, print_table=True
)

# Read M2 files
dataset_ref = reader.read("demo/examples/demo.errant")
dataset_hyp = reader.read("demo/examples/demo-AMU.errant")

print(f"Number of reference sentences: {len(dataset_ref)}")
print(f"Number of hypothesis sentences: {len(dataset_hyp)}")

# Build metric
metric = DependentCLEME(
    lang=lang,
    scorer=DEFAULT_HEUO_SCORER_CORPUS,
    weigher_type=WeigherType.SIMILARITY,
)

# Evaluate
score, _ = metric.evaluate(hyp_dataset, ref_dataset)
print(f"==================== CLEME Evaluation Results ====================")
print(score)
```

Refer to `main.py` for more details.

## 5. Adapting to Other Languages

Both CLEME and CLEME2.0 are designed to be language-agnostic at their core, provided that the input data (source, hypothesis, references) is available in the M2 format (or a compatible ERRANT-parsable format).

1. Ensure you have an ERRANT variant capable of parsing your target language's M2 files (see table above).
2. Prepare your data in the M2 format. Each M2 file should correspond to one set of annotations (e.g., one reference set, or one system hypothesis).
3. Use the CLI or API as described, pointing to your language-specific M2 files.

The chunking mechanism itself does not rely on language-specific rules beyond what ERRANT provides for edit extraction.

## 6. Recommended Hyperparameters

The default hyperparameters for CLEME (e.g., for the weigher in DependentChunkMetric) were optimized on the CoNLL-2014 English reference set. These are defined in cleme/constant.py. For CLEME2.0, default weights for combining the four aspects into an aggregated score are also provided, tuned on human judgment datasets like GJG15 and SEEDA. Users may need to tune these parameters if working with significantly different datasets or languages for optimal correlation with human judgment.

## 7. Datasets

The development and evaluation of CLEME and CLEME2.0 were performed on several standard GEC datasets:

- **CLEME**:
  - Human judgment datasets: GJG15 and SEEDA.
  - Reference datasets: CoNLL-2014, BN-10GEC...
- **CLEME2.0**:
  - Human judgment datasets: GJG15 and SEEDA.
  - Reference datasets: CoNLL-2014, BN-10GEC...

We encourage users to test the metrics on diverse datasets to further validate their robustness.

## Citation

If you use CLEME or CLEME2.0 in your research, please cite the respective papers:

```bibtex
@inproceedings{ye-etal-2023-cleme,
    title = "{CLEME}: Debiasing Multi-reference Evaluation for Grammatical Error Correction",
    author = "Ye, Jingheng  and Li, Yinghui  and Zhou, Qingyu  and Li, Yangning and Ma, Shirong and Zheng, Hai-Tao and Shen, Ying",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.378/",
    doi = "10.18653/v1/2023.emnlp-main.378",
    pages = "6174--6189",
}

@article{ye2024cleme2,
  title={{CLEME2.0}: Towards More Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction},
  author={Ye, Jingheng and Xu, Zishan and Li, Yinghui and Cheng, Xuxin and Song, Linlin and Zhou, Qingyu and Zheng, Hai-Tao and Shen, Ying and Su, Xin},
  journal={arXiv preprint arXiv:2407.00934},
  year={2024},
  eprint={2407.00934},
  archiveprefix={arXiv},
  primaryclass={cs.CL}
}
% Note: Update the CLEME2.0 citation once it's officially published in ACL 2025 proceedings.
```

## Contributing

We welcome contributions to improve CLEME and CLEME2.0! Please feel free to:

- Report bugs or issues.
- Suggest new features or enhancements.
- Submit pull requests with bug fixes or new functionalities.

When contributing, please ensure your code adheres to the existing style and includes relevant tests.

## Update Logs

### v1.1.0 (2025.05.17)

- Integration of **CLEME2.0** features.
- CLI and API support for disentangled GEC evaluation (hit, wrong, under, over-correction).
- Implementation of edit weighting mechanisms.
- Updated documentation and examples for CLEME2.0.

### v1.0.0 (2023.11.15)

- Initial release of **CLEME**.
- Support for unbiased multi-reference GEC evaluation.
- CLI and API for F-score calculation.
- A visualization tool for the evaluation process.

## License

This project is licensed under the [Apache License 2.0](https://aigcbest.top/LICENSE).

## Contact & Feedback

For any questions, feedback, or suggestions, please contact Jingheng Ye at yejh22@mails.tsinghua.edu.cn or open an issue on this GitHub repository.
