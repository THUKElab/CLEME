from typing import Any, Dict, List

import math
import numpy as np

from .scorer_base import Scorer, ScorerForGLEU, compute_f, compute_acc
from ..constants import *


class SentenceScorer(Scorer):
    def __call__(self, scorer_inputs: List[List[Dict[str, int]]]) -> Dict[str, Any]:
        """ Calculate sentence-level Accuracy Score """
        total_f, total_p, total_r, total_acc = [], [], [], []
        for sample_result in scorer_inputs:
            best_f, best_p, best_r, best_acc = -1.0, -1.0, -1.0, -1.0
            for ref_result in sample_result:
                _tp, _fp, _fn, _tn = ref_result[KEY_TP], ref_result[KEY_FP], ref_result[KEY_FN], ref_result[KEY_TN]
                _p, _r, _f = compute_f(_tp, _fp, _fn)
                _acc = compute_acc(_tp, _fp, _fn, _tn)
                # print(_tp, _fp, _fn, _f)
                if (_f > best_f) or \
                        (_f == best_f and _p > best_p) or \
                        (_f == best_f and _p == best_p and _r > best_r) or \
                        (_f == best_f and _p == best_p and _r == best_r and _acc < best_acc):
                    best_f, best_p, best_r, best_acc = _f, _p, _r, _acc
            total_f.append(best_f)
            total_p.append(best_p)
            total_r.append(best_r)
            total_acc.append(best_acc)

        f, p, r, acc = np.average(total_f), np.average(total_p), np.average(total_r), np.average(total_acc)
        return {
            "num_sample": len(scorer_inputs),
            KEY_F: round(f, 4),
            KEY_P: round(p, 4),
            KEY_R: round(r, 4),
            KEY_ACC: round(acc, 4),
        }


class SentenceScorerForAccuracy(Scorer):
    def __call__(self, scorer_inputs: List[List[Dict[str, int]]]) -> Dict[str, Any]:
        """ Calculate sentence-level Accuracy Score """
        total_f, total_p, total_r, total_acc = [], [], [], []
        for sample_result in scorer_inputs:
            best_acc = -1.0
            for ref_result in sample_result:
                _tp, _fp, _fn, _tn = ref_result[KEY_TP], ref_result[KEY_FP], ref_result[KEY_FN], ref_result[KEY_TN]
                _acc = compute_acc(_tp, _fp, _fn, _tn)
                if _acc > best_acc:
                    best_acc = _acc
            total_acc.append(best_acc)
        acc = np.average(total_acc)
        # return {KEY_F: round(f, 4), KEY_ACC: round(acc, 4), KEY_P: round(p, 4), KEY_R: round(r, 4)}
        return {KEY_ACC: round(acc, 4)}


class SentenceScorerForGLEU(ScorerForGLEU):
    def __call__(self, scorer_inputs: List[List[Dict]]) -> Dict[str, Any]:
        total_scores = []
        for sample_idx, sample_result in enumerate(scorer_inputs):
            sample_score = []
            for ref_idx, ref_result in enumerate(sample_result):
                ref_len = ref_result[KEY_REF_LEN] if ref_result[KEY_REF_LEN] != 0 else 1
                hyp_len = ref_result[KEY_HYP_LEN] if ref_result[KEY_HYP_LEN] != 0 else 1
                log_gleu_prec = 0.0
                for n, precision in ref_result[KEY_NGRAMS].items():
                    numerator = precision[0] if precision[0] != 0 else 1
                    denominator = precision[1] if precision[1] != 0 else 1
                    log_gleu_prec += math.log(float(numerator) / denominator)
                log_gleu_prec /= self.order
                ref_score = math.exp(min([0, 1 - float(ref_len) / hyp_len]) + log_gleu_prec)
                total_scores.append(ref_score)
                sample_score.append(ref_score)
        return {
            "score": np.average(total_scores),
            "std": np.std(total_scores),
            # "ci": scipy.stats.norm.interval(0.95, loc=mean, scale=std)
        }
