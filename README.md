<div align="center">

# CLEME: Debiasing Multi-reference Evaluation for Grammatical Error Correction

</div>

The repository contains the codes and data for our EMNLP 2023 Main Paper: [CLEME: Debiasing Multi-reference Evaluation for Grammatical Error Correction](https://arxiv.org/abs/2305.10819).

**CLEME** is a reference-based metric that evaluate Grammatical Error Correction (GEC) systems at the chunk-level, aiming to provide unbiased F$_{0.5}$ scores for GEC multi-reference evaluation.

## Features

- CLEME is **unbiased**, allowing a more objective evaluation pipeline.
- CLEME is able to **visualize** evaluation pipeline as **tables**.
- CLEME supports **English and Chinese** for now. We will extend to other languages in the future.

## Requirements and Installation

- Python version >= 3.7

- ERRANT or variants for other specific languages. We recommend installing the newest version of ERRANT.

  | Language | Link                                                         |
  | -------- | ------------------------------------------------------------ |
  | English  | [ERRANT](https://github.com/chrisjbryant/errant)             |
  | Chinese  | [ChERRANT](https://github.com/HillZhang1999/MuCGEC/blob/main/scorers/ChERRANT) |

  We recommend the newest version of ERRANT for speed gain, although we use ERRANT v2.3.3 in the paper.

- Clone this repository:

```bash
git clone https://github.com/THUKElab/CLEME.git
cd ./CLEME
```

## Usage

### CLI

##### Evaluate AMU system

```bash
python scripts/evaluate.py --ref tests/examples/conll14.errant --hyp tests/examples/conll14-AMU.errant

{'num_sample': 1312, 'F': 0.2514, 'Acc': 0.7634, 'P': 0.2645, 'R': 0.2097, 'tp': 313.51, 'fp': 871.8, 'fn': 1181.71, 'tn': 6312.0}
```

##### Visualize evaluation process as tables

```bash
python scripts/evaluate.py  --ref tests/examples/demo.errant  --hyp tests/examples/demo-AMU.errant  --vis
```

### API

##### Evaluate AMU system using CLEME-dependent

```python
# Read M2 file
dataset_ref = self.reader.read(f"{os.path.dirname(__file__)}/examples/demo.errant")
dataset_hyp = self.reader.read(f"{os.path.dirname(__file__)}/examples/demo-AMU.errant")
print(len(dataset_ref), len(dataset_hyp))
print("Example of reference", dataset_ref[-1])
print("Example of hypothesis", dataset_hyp[-1])

# Evaluate using CLEME_dependent
config_dependent = {
	"tp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
	"fp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": True},
	"fn": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
}
metric_dependent = DependentChunkMetric(weigher_config=config_dependent)
score, results = metric_dependent.evaluate(dataset_hyp, dataset_ref)
print(f"==================== Evaluate Demo ====================")
print(score)

# Visualize
metric_dependent.visualize(dataset_ref, dataset_hyp)
```

Refer to `./tests/test_cleme.py` for more details.

## Adapt to Other Languages

CLEME is language-agnostic, so you can easily employ CLEME for any languages if you have got reference and hypothesis M2 files.

## Recommended Hyper-parameters

We search optimal hyper-parameters on `CoNLL-2014` reference set, which are listed in `.cleme/constant.py`.

## Citation

```bib
@article{ye-et-al-2023-cleme,
  title   = {CLEME: Debiasing Multi-reference Evaluation for Grammatical Error Correction},
  author  = {Ye, Jingheng and Li, Yinghui and Zhou, Qingyu and Li, Yangning and Ma, Shirong and Zheng, Hai-Tao and Shen, Ying},
  journal = {arXiv preprint arXiv:2305.10819},
  year    = {2023}
}
```

## Update Logs

### v1.0 (2023.11.15)

CLEME v1.0 released.

## Contact & Feedback

If you have any questions or feedbacks, please send e-mails to ours: yejh22@mails.tsinghua.edu.cn













