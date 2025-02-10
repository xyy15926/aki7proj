#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_numpy.py
#   Author: xyy15926
#   Created: 2025-02-08 18:36:44
#   Updated: 2025-02-10 22:00:24
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from packaging.version import Version
import numpy as np


# %%
# --------------------------------------------------------------
#               * * * FUTURE WARNING * * *
# --------------------------------------------------------------
# https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
# The behavior of comparison between numpy str and numerics ndarray isn't
# decided yet and the scalar return may be changed.
@pytest.mark.skipif(Version(np.__version__) >= Version("2.0"),
                    reason="Comparison behavior isn't decided yet.")
def test_str_numarr_comparison_lt20():
    # The `ret` is scalar.
    with pytest.warns(FutureWarning):
        ret = np.arange(5).astype(str) == np.arange(5)
    assert np.isscalar(ret)
    with pytest.warns(FutureWarning):
        ret = "a" == np.arange(5)
    assert np.isscalar(ret)

    # This seems not to be ambiguous but FutureWarnings raised anyway.
    with pytest.warns(FutureWarning):
        ret = "a" in np.arange(5)
    assert np.isscalar(ret)


# No more FutureWarning raised.
@pytest.mark.skipif(Version(np.__version__) < Version("2.0"),
                    reason="Comparison behavior has been decided.")
def test_str_numarr_comparison_ge20():
    # The `ret` is not scalar.
    ret = np.arange(5).astype(str) == np.arange(5)
    assert not np.isscalar(ret)
    ret = "a" == np.arange(5)
    assert not np.isscalar(ret)


# %%
# --------------------------------------------------------------
#               * * * FEATURE TESTS * * *
# --------------------------------------------------------------
# `np.isclose` only support numeric dtype.
def test_isclose():
    a = np.array([1, np.nan], dtype=object)
    with pytest.raises(TypeError):
        np.isclose(a, a)
    assert isinstance(a[0], int)
    assert isinstance(a[1], float)


# %%
# --------------------------------------------------------------
#               * * * BUG ???????? * * *
# --------------------------------------------------------------
# `np.unqiue` may return duplicate elements when np.ndarray.dtype is objects
# and `np.nan` exists.
def test_unique():
    a = np.array([1, 1, np.nan], dtype=object)
    # Return `[1, np.nan]`
    ret = np.unique(a)
    assert len(ret) == 2

    a = np.array([np.nan, 1, 1], dtype=object)
    # Return `[1, np.nan]`
    ret = np.unique(a)
    assert len(ret) == 2

    # For np.ndarray with object dtype.
    a = np.array([1, np.nan, 1], dtype=object)
    # Return `[1, np.nan, 1]`
    ret = np.unique(a)
    assert len(ret) == 3
    assert ret[0] == ret[2] == 1
    assert np.isnan(ret[1])

    # For np.ndarray with object dtype.
    # Return `[np.nan, 1]`
    a = np.array([1, np.nan, 1])
    ret = np.unique(a)
    assert a.dtype == "float64"
    assert len(ret) == 2


# %%
# `Non-0 / 0` and `0 / 0` are different in warning messages.
def test_zero_divivde():
    with pytest.warns(RuntimeWarning, match="divide by zero encountered"):
        np.array(1) / np.array(0)
    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
        np.array(0) / np.array(0)
