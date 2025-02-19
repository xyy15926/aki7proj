#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_data_preprocessing.py
#   Author: xyy15926
#   Created: 2023-07-24 15:58:10
#   Updated: 2025-02-18 22:06:05
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
    from ubears.ringbear import metrics
    reload(metrics)

from ubears.ringbear.metrics import (cal_lifts_weighted,
                                     cal_woes_weighted,
                                     cal_ivs_weighted)

from ubears.ringbear.metrics import (cal_lifts, cal_woes, cal_ivs)


# %%
def test_lifts():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20)
    aa = np.array([3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(60)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2

    wux, wlifts, wax, wacl, wasc = cal_lifts_weighted(a, b)
    lifts, acl, racl, kcorr, pv = cal_lifts(a, b)
    assert np.all(np.isclose(wlifts, lifts))
    assert np.all(np.isclose(lifts, [2, 0.6, 0.4]))
    assert np.all(np.isclose(wacl, acl))
    assert np.all(np.isclose(acl, [2, 1.3, 1]))
    assert kcorr > 0

    wux, wlifts, wax, wacl, wasc = cal_lifts_weighted(aa, b)
    lifts, acl, racl, kcorr, pv = cal_lifts(aa, b)
    assert np.all(np.isclose(wlifts, lifts))
    assert np.all(np.isclose(lifts, [0.4, 0.6, 2]))
    assert np.all(np.isclose(wacl, racl))
    assert np.all(np.isclose(racl, [2, 1.3, 1]))
    assert kcorr < 0

    wux, wlifts, wax, wacl, wasc = cal_lifts_weighted(a, b, acc_keys=[2, 3])
    lifts, acl, racl, kcorr, pv = cal_lifts(a, b, acc_keys=[2, 3])
    assert np.all(np.isclose(wlifts, lifts))
    assert np.all(np.isclose(lifts, [2, 0.6, 0.4]))
    assert np.all(np.isclose(wacl, acl))
    assert np.all(np.isclose(acl, [0.6, 0.5]))
    assert not wasc
    assert kcorr > 0

    wux, wlifts, wax, wacl, wasc = cal_lifts_weighted(a, b, acc_keys=[3, 2])
    lifts, acl, racl, kcorr, pv = cal_lifts(a, b, acc_keys=[3, 2])
    assert np.all(np.isclose(wlifts, lifts))
    assert np.all(np.isclose(lifts, [2, 0.6, 0.4]))
    assert np.all(np.isclose(wacl, racl))
    assert np.all(np.isclose(racl, [0.6, 0.5]))
    assert wasc
    assert kcorr < 0


# %%
def test_woe():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20)
    aa = np.array([3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(60)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2

    unis_1, woes_1, ivs_1 = cal_woes_weighted(a, b)
    unis_2, woes_2, ivs_2 = cal_woes_weighted(aa, b)

    assert np.all(unis_1 == unis_2)
    assert np.all(woes_1 == woes_2[::-1])
    assert np.all(ivs_1 == ivs_2[::-1])

    woes_n1, ivs_n1 = cal_woes(a, b)
    woes_n2, ivs_n2 = cal_woes(aa, b)

    assert np.all(np.isclose(woes_1, woes_n1))
    assert np.all(np.isclose(woes_2, woes_n2))
    assert np.all(np.isclose(ivs_1, ivs_n1))
    assert np.all(np.isclose(ivs_2, ivs_n2))


def test_ivs():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20)
    aa = np.array([3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(60)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2

    iv1, iv2 = cal_ivs_weighted(np.column_stack((a, aa)), b)
    assert np.isclose(iv1, iv2)

    iv11, iv22 = cal_ivs(np.column_stack((a, aa)), b)
    assert np.isclose(iv11, iv22)
    assert np.isclose(iv1, iv11)

    (f1, f2), ctab = contingency.crosstab(a, b)
    freqr = ctab / ctab.sum(axis=0, keepdims=True)
    iv_ent = (entropy(freqr[:, 0], freqr[:, 1])
              + entropy(freqr[:, 1], freqr[:, 0]))

    assert np.isclose(iv_ent, iv1)


# %%
@mark.filterwarnings("ignore: divide by zero encountered in log")
def test_woe_with_freq_0():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)
    aa = np.array([4] * 20 + [3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(80)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2

    unis_1, woes_1, ivs_1 = cal_woes_weighted(a, b)
    unis_2, woes_2, ivs_2 = cal_woes_weighted(aa, b)

    assert np.all(unis_1 == unis_2)
    assert np.all(woes_1 == woes_2[::-1])
    assert np.all(ivs_1 == ivs_2[::-1])

    woes_n1, ivs_n1 = cal_woes(a, b)
    woes_n2, ivs_n2 = cal_woes(aa, b)

    assert np.all(np.isclose(woes_1, woes_n1))
    assert np.all(np.isclose(woes_2, woes_n2))
    assert np.all(np.isclose(ivs_1, ivs_n1))
    assert np.all(np.isclose(ivs_2, ivs_n2))


@mark.filterwarnings("ignore: divide by zero encountered in log")
def test_ivs_with_freq_0():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)
    aa = np.array([4] * 20 + [3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(80)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2

    iv1, iv2 = cal_ivs_weighted(np.column_stack((a, aa)), b)
    assert np.isclose(iv1, iv2)

    iv11, iv22 = cal_ivs(np.column_stack((a, aa)), b)
    assert np.isclose(iv11, iv22)
    assert np.isclose(iv1, iv11)

    (f1, f2), ctab = contingency.crosstab(a, b)
    freqr = ctab / ctab.sum(axis=0, keepdims=True)
    iv_kl = np.nan_to_num(kl_div(freqr[:, 0], freqr[:, 1])
                          + kl_div(freqr[:, 1], freqr[:, 0]),
                          False, 0, 0, 0).sum()

    assert np.isclose(iv1, iv_kl)

    # Entropy can't handle frequency with 0 while calculating KL-Div.
    iv_ent = (entropy(freqr[:, 0], freqr[:, 1])
              + entropy(freqr[:, 1], freqr[:, 0]))
    assert np.inf == iv_ent
