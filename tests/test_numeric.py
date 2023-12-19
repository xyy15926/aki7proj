#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_numeric.py
#   Author: xyy15926
#   Created: 2023-12-07 20:06:03
#   Updated: 2023-12-08 10:04:45
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from ringbear import numeric
    reload(numeric)

from ringbear.numeric import (std_outlier, remove_std_outlier, span_cut,
                              ordinal_encode,
                              POS_FLAG, NEG_FLAG)


# %%
def test_std_outlier():
    arr = np.arange(11)
    arr[-1] = 100
    lowiers, highiers, outlier_map = std_outlier(arr)
    assert lowiers.shape == (0,)
    assert np.all(highiers == [100])
    assert np.all(np.zeros(10, dtype=np.bool_) == outlier_map[:10])
    assert np.all(np.ones(1, dtype=np.bool_) == outlier_map[-1:])

    arr_ = remove_std_outlier(arr)
    assert np.all(arr_ == np.array([*np.arange(10), POS_FLAG]))
    arr_ = remove_std_outlier(arr, max_replace=np.mean)
    # np.mean(np.range(10)) == 4.5, but it will be casted to 4 for dtype.
    assert np.all(arr_ == np.array([*np.arange(10), 4]))

    arr = np.arange(11)
    arr[-1] = 100
    arr[0] = -100
    lowiers, highiers, outlier_map = std_outlier(arr, excluded=True)
    assert np.all(lowiers == [-100])
    assert np.all(highiers == [100])
    assert np.all(np.zeros(9, dtype=np.bool_) == outlier_map[1:10])
    assert np.all(np.ones(1, dtype=np.bool_) == outlier_map[-1:])
    assert np.all(np.ones(1, dtype=np.bool_) == outlier_map[:1])

    arr[-1] = 100
    arr[0] = -100
    lowiers, highiers, outlier_map = std_outlier(arr, excluded=False)
    assert lowiers.shape == (0,)
    assert highiers.shape == (0,)
    assert np.all(np.zeros(11, dtype=np.bool_) == outlier_map)


# %%
def test_span_cut():
    arr = np.concatenate([np.arange(10),
                          np.arange(20, 30),
                          np.arange(50, 70)])
    assert np.all(span_cut(arr, 3) == [0, 14.5, 39.5, 69])
    assert np.all(span_cut(arr, None, 2.5) == [0, 14.5, 39.5, 69])


# %%
def test_ordinal_encode():
    arr = np.concatenate([np.arange(10),
                          np.arange(20, 30),
                          np.arange(50, 70)])
    bin_edges = span_cut(arr, 3)
    ordinal_encode(arr, bin_edges)


