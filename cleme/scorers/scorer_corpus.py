import random
from typing import Any, Dict, List

import math

from .scorer_base import Scorer, ScorerForGLEU, compute_f, compute_acc
from ..constants import *


class CorpusScorer(Scorer):
    def __call__(self, scorer_inputs: List[List[Dict]]) -> Dict[str, Any]:
        """ Calculate corpus-level F_beta Score """
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        for sample_result in scorer_inputs:
            best_f, best_tp, best_fp, best_fn, best_tn = -1.0, 0, 0, 0, 0
            for ref_result in sample_result:
                _tp, _fp, _fn, _tn = ref_result[KEY_TP], ref_result[KEY_FP], ref_result[KEY_FN], ref_result[KEY_TN]
                _p, _r, _f = compute_f(total_tp + _tp, total_fp + _fp, total_fn + _fn)
                if (_f > best_f) or \
                        (_f == best_f and _tp > best_tp) or \
                        (_f == best_f and _tp == best_tp and _fp < best_fp) or \
                        (_f == best_f and _tp == best_tp and _fp == best_fp and _fn < best_fn) or \
                        (_f == best_f and _tp == best_tp and _fp == best_fp and _fn == best_fn and _tn > best_tn):
                    best_f, best_tp, best_fp, best_fn, best_tn = _f, _tp, _fp, _fn, _tn
            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
            total_tn += best_tn
        p, r, f = compute_f(total_tp, total_fp, total_fn)
        acc = compute_acc(total_tp, total_fp, total_fn, total_tn)
        # print(f"{total_tp}, {total_fp}, {total_fn}, {total_tn}, {p}, {r}, {f}, {acc}")
        return {
            "num_sample": len(scorer_inputs),
            KEY_F: f, KEY_ACC: acc, KEY_P: p, KEY_R: r,
            KEY_TP: round(total_tp, 2), KEY_FP: round(total_fp, 2),
            KEY_FN: round(total_fn, 2), KEY_TN: round(total_tn, 2),
        }


class CorpusScorerForAccuracy(Scorer):
    def __call__(self, scorer_inputs: List[List[Dict[str, int]]]) -> Dict[str, Any]:
        """ Calculate corpus-level F_beta Score """
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        for sample_result in scorer_inputs:
            best_acc, best_tp, best_fp, best_fn, best_tn = -1.0, 0, 0, 0, 0
            for ref_result in sample_result:
                _tp, _fp, _fn, _tn = ref_result[KEY_TP], ref_result[KEY_FP], ref_result[KEY_FN], ref_result[KEY_TN]
                # _p, _r, _f = compute_f(total_tp + _tp, total_fp + _fp, total_fn + _fn)
                _acc = compute_acc(total_tp + _tp, total_fp + _fp, total_fn + _fn, total_tn + _tn)

                if (_acc > best_acc) or \
                        (_acc == best_acc and _tp > best_tp) or \
                        (_acc == best_acc and _tp == best_tp and _fp < best_fp) or \
                        (_acc == best_acc and _tp == best_tp and _fp == best_fp and _fn < best_fn) or \
                        (_acc == best_acc and _tp == best_tp and _fp == best_fp and _fn == best_fn and _tn > best_tn):
                    best_acc, best_tp, best_fp, best_fn, best_tn = best_acc, _tp, _fp, _fn, _tn
            total_tp += best_tp
            total_fp += best_fp
            total_fn += best_fn
            total_tn += best_tn
        # p, r, f = compute_f(total_tp, total_fp, total_fn)
        acc = compute_acc(total_tp, total_fp, total_fn, total_tn)
        # print(f"{total_tp}, {total_fp}, {total_fn}, {total_tn}, {p}, {r}, {f}, {acc}")
        return {
            # KEY_F: f, KEY_P: p, KEY_R: r,
            KEY_ACC: acc,
            KEY_TP: round(total_tp, 2), KEY_FP: round(total_fp, 2),
            KEY_FN: round(total_fn, 2), KEY_TN: round(total_tn, 2),
        }


class CorpusScorerForGLEU(ScorerForGLEU):
    def __call__(self, scorer_inputs: List[List[Dict]]) -> Dict[str, Any]:
        total_hyp_len, total_ref_len = 0, 0
        total_ngrams = [0] * (len(scorer_inputs[0][0][KEY_NGRAMS]) * 2)
        for sample_result in scorer_inputs:
            for _ in range(self.num_iter):
                ref_idx = random.randint(0, len(sample_result) - 1)
                total_hyp_len += sample_result[ref_idx][KEY_HYP_LEN]
                total_ref_len += sample_result[ref_idx][KEY_REF_LEN]
                for n, precision in sample_result[ref_idx][KEY_NGRAMS].items():
                    assert len(precision) == 2
                    total_ngrams[2 * n - 2] += precision[0]
                    total_ngrams[2 * n - 1] += precision[1]

        # smooth 0 counts for sentence-level scores
        if self.smoothing:
            total_hyp_len = [x if x != 0 else 1 for x in total_ngrams]

        assert len(list(filter(lambda x: x == 0, total_ngrams))) == 0
        log_gleu_prec = sum([math.log(float(x) / y) for x, y in zip(total_ngrams[0::2], total_ngrams[1::2])]) / 4
        score = math.exp(min([0, 1 - float(total_ref_len) / total_hyp_len]) + log_gleu_prec)
        return {
            "score": score,
        }
