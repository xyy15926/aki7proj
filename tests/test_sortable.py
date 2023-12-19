#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_data_preprocessing.py
#   Author: xyy15926
#   Created: 2023-07-24 15:58:10
#   Updated: 2023-12-07 16:04:07
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from ringbear import metrics
    reload(metrics)

from ringbear.metrics import lifts, woes, ivs


# %%
def test_lifts():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20)
    aa = np.array([3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(60)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2
    assert np.all(lifts(a, b)[2] == [1, 2, 3])
    assert np.all(lifts(aa, b)[2] == [3, 2, 1])
    assert np.all(lifts(a, b, acc_keys=[2, 3])[2] == [2, 3])
    assert np.all(lifts(a, b, acc_keys=[3, 2])[2] == [2, 3])
    assert np.all(lifts(aa, b, acc_keys=[2, 3])[2] == [3, 2])
    assert np.all(lifts(aa, b, acc_keys=[3, 2])[2] == [3, 2])


def test_woe():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20)
    aa = np.array([3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(60)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2
    unis_1, woes_1, ivs_1 = woes(a, b)
    unis_2, woes_2, ivs_2 = woes(aa, b)
    assert np.all(unis_1 == unis_2)
    assert np.all(woes_1 == woes_2[::-1])
    assert np.all(ivs_1 == ivs_2[::-1])


def test_ivs():
    a = np.array([1] * 20 + [2] * 20 + [3] * 20)
    aa = np.array([3] * 20 + [2] * 20 + [1] * 20)
    b = np.zeros(60)
    b[:10], b[20:23], b[40:42] = [1] * 10, [1] * 3, [1] * 2
    iv1, iv2 = ivs(np.column_stack((a, aa)), b)
    assert iv1 == iv2

