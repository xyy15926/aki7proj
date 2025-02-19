#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_sklearn.py
#   Author: xyy15926
#   Created: 2025-02-17 16:11:14
#   Updated: 2025-02-19 14:34:25
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from packaging.version import Version
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OrdinalEncoder


# %%
def test_ordinize():
    a = np.array(["11", "12", "13", "11", np.nan, None])
    assert a.dtype == "object"

    # `pd.factorize` treats `None` as `NA`.
    vals, codes = pd.factorize(a)
    assert np.all(vals == (0, 1, 2, 0, -1, -1))

    # OrdinalEncoder only take `np.nan` as `NA`, and `None` will be the 4th.
    vals = (OrdinalEncoder(encoded_missing_value=-1)
            .fit_transform(a.reshape(-1, 1)))
    assert np.all(vals.ravel() == (0, 1, 2, 0, -1, 3))

    # OrdinalEncoder could also take customed ordinal order.
    # And `np.nan` must be added explicitly if `handle_unknown` is `error` and
    # `np.nan` exists in data.
    with pytest.raises(ValueError):
        vals = (OrdinalEncoder(categories=[[None, "11", "12", "13"]],
                               encoded_missing_value=-1)
                .fit_transform(a.reshape(-1, 1)))
    # ATTENTION: `np.nan` should always be put at the last position.
    vals = (OrdinalEncoder(categories=[[None, "11", "12", "13", np.nan]],
                           encoded_missing_value=-1)
            .fit_transform(a.reshape(-1, 1)))
    assert np.all(vals.ravel() == (1, 2, 3, 1, -1, 0))
    # 1. For sklearn >= 1.6, `np.nan` must be the last elements, or ValueError
    #   will be raised.
    if Version(sklearn.__version__) >= Version("1.6"):
        with pytest.raises(ValueError):
            vals = (OrdinalEncoder(categories=[[np.nan, None, "11", "12", "13"]],
                                   encoded_missing_value=-1)
                    .fit_transform(a.reshape(-1, 1)))
    # 2. For sklearn with lower version, `np.nan` could be put at any
    #   position, but it won't be encoded with correspondant value but the
    #   `encoded_missing_value` and the correspondant value will be lost.
    if Version(sklearn.__version__) <= Version("1.1.3"):
        vals = (OrdinalEncoder(categories=[[np.nan, None, "11", "12", "13"]],
                               encoded_missing_value=-1)
                .fit_transform(a.reshape(-1, 1)))
        assert np.all(vals.ravel() == (2, 3, 4, 2, -1, 1))
