#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_freqs.py
#   Author: xyy15926
#   Created: 2023-12-06 18:42:42
#   Updated: 2024-01-17 21:05:12
#   Description:
# ---------------------------------------------------------

# %%
from pytest import mark
import numpy as np

if __name__ == "__main__":
    from importlib import reload
    from ringbear import freqs
    reload(freqs)

from sklearn.datasets import load_iris
from scipy.stats import contingency
from ringbear.freqs import (cal_entropy, cal_gini, chi_pairwisely,
                            chi2_only,
                            enhanced_freqs)


# %%
@mark.filterwarnings("ignore: divide by zero")
def test_cal_entropy():
    freqs = np.array([[12   ,12     ,12],
                      [12   ,23     ,122],
                      [12   ,23     ,0],
                      [12   ,23     ,0],
                      [0    ,23     ,0]])
    assert(cal_entropy(freqs).shape[0] == freqs.shape[0])
    assert(cal_entropy(freqs, axis=0).shape[0] == freqs.shape[-1])
    assert(cal_entropy(freqs, axis=1).shape[0] == freqs.shape[0])
    assert(cal_entropy(freqs, axis=None).shape == tuple())
    assert(cal_entropy(freqs[0]).shape == tuple())


def test_gini():
    freqs = np.array([[12   ,12     ,12],
                      [12   ,23     ,122],
                      [12   ,23     ,0],
                      [12   ,23     ,0],
                      [0    ,23     ,0]])
    assert(cal_gini(freqs).shape[0] == freqs.shape[0])
    assert(cal_gini(freqs, axis=0).shape[0] == freqs.shape[-1])
    assert(cal_gini(freqs, axis=1).shape[0] == freqs.shape[0])
    assert(cal_gini(freqs, axis=None).shape == tuple())
    assert(cal_gini(freqs[0]).shape == tuple())


def test_chi_pairwise():
    freqs = np.array([[12   ,12     ,12],
                      [12   ,23     ,122],
                      [12   ,23     ,1],
                      [12   ,23     ,1],
                      [1    ,23     ,1]])
    chis = chi_pairwisely(freqs)
    assert(chis[2, 0] == 0)
    assert(chis.shape[0] == freqs.shape[0] - 1)
    assert(np.all(chi_pairwisely(freqs, axis=0) == chis))
    assert(chi_pairwisely(freqs, axis=1).shape[0] == freqs.shape[1] - 1)


# %%
def test_enhanced_freqs_1D():
    rows = np.concatenate([np.ones(5),
                           np.ones(6) * 2,
                           np.ones(7) * 3])
    others = np.column_stack((rows[::-1], rows))
    ret, unisr = enhanced_freqs(rows, others=others, agg=np.sum)
    assert np.all(ret == [20, 26, 30])
    assert np.all(unisr == [1, 2, 3])
    ret, unisr = enhanced_freqs(rows, others=others, agg=lambda x: np.sum(x, axis=0))
    assert np.all(ret == [[15, 5],
                          [14, 12],
                          [9, 21]])
    assert np.all(unisr == [1, 2, 3])


def test_enhanced_freqs_2D():
    rows = np.concatenate([np.ones(5),
                           np.ones(6) * 2,
                           np.ones(7) * 3])
    cols = rows[::-1] * 6
    others = cols / 2
    ret, unisr, unisc = enhanced_freqs(rows, cols, others=others, agg=np.sum)
    assert np.all(np.nan_to_num(ret, 0) == [[0    , 0     , 45],
                                            [0    , 24    , 18],
                                            [15   , 12    , 0]])
    assert np.all(unisr == [1, 2, 3])
    assert np.all(unisc == [6, 12, 18])


def test_chi2_only():
    X, y = load_iris(return_X_y=True)
    (ux, uy), freqs = contingency.crosstab(X[:, 0], y)
    chi2 = chi2_only(freqs)
    cchi2 = contingency.chi2_contingency(freqs)[0]
    assert chi2 == cchi2


