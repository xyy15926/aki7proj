#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_data_preprocessing.py
#   Author: xyy15926
#   Created: 2023-07-24 15:58:10
#   Updated: 2024-01-02 10:20:30
#   Description:
# ---------------------------------------------------------

# %%
from pytest import mark
import numpy as np
import pandas as pd
from scipy.stats import entropy, contingency
from scipy.special import kl_div

if __name__ == "__main__":
    from importlib import reload
    from ringbear import metrics
    reload(metrics)

from ringbear.metrics import cal_lifts, cal_woes, cal_ivs


# %%
def test_lifts():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20)
    aa = np.array([3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(60)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2
    assert np.all(cal_lifts(a, b)[2] == [1, 2, 3])
    assert np.all(cal_lifts(aa, b)[2] == [3, 2, 1])
    assert np.all(cal_lifts(a, b, acc_keys=[2, 3])[2] == [2, 3])
    assert np.all(cal_lifts(a, b, acc_keys=[3, 2])[2] == [2, 3])
    assert np.all(cal_lifts(aa, b, acc_keys=[2, 3])[2] == [3, 2])
    assert np.all(cal_lifts(aa, b, acc_keys=[3, 2])[2] == [3, 2])


def test_woe():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20)
    aa = np.array([3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(60)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2
    unis_1, woes_1, ivs_1 = cal_woes(a, b)
    unis_2, woes_2, ivs_2 = cal_woes(aa, b)
    assert np.all(unis_1 == unis_2)
    assert np.all(woes_1 == woes_2[::-1])
    assert np.all(ivs_1 == ivs_2[::-1])


def test_ivs():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20)
    aa = np.array([3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(60)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2
    iv1, iv2 = cal_ivs(np.column_stack((a, aa)), b)
    assert np.isclose(iv1, iv2)

    (f1, f2), ctab = contingency.crosstab(a, b)
    freqr = ctab / ctab.sum(axis=0, keepdims=True)
    iv_ent = (entropy(freqr[:, 0], freqr[:, 1], base=2)
              + entropy(freqr[:, 1], freqr[:, 0], base=2))

    assert np.isclose(iv_ent, iv1)


@mark.filterwarnings("ignore: divide by zero encountered in log")
def test_ivs_with_0():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)
    aa = np.array([3] * 20 + [2] * 20 + [1] * 20 + [4] * 20)
    b = np.zeros(80)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2
    iv1, iv2 = cal_ivs(np.column_stack((a, aa)), b)
    assert np.isclose(iv1, iv2)

    (f1, f2), ctab = contingency.crosstab(a, b)
    freqr = ctab / ctab.sum(axis=0, keepdims=True)
    iv_kl = np.nan_to_num(kl_div(freqr[:, 0], freqr[:, 1])
                          + kl_div(freqr[:, 1], freqr[:, 0]),
                          0, 0 ,0).sum() / np.log(2)

    assert np.isclose(iv1, iv_kl)
