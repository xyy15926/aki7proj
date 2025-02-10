#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_scipy.py
#   Author: xyy15926
#   Created: 2025-02-10 19:20:23
#   Updated: 2025-02-10 21:23:38
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
from scipy.stats import contingency
import pandas as pd


# %%
def test_contingency_sortable_only():
    row = [1, 1, 2, 2, 3, 3]
    col = ["a", "a", "a", 1, 1, 1]
    (re, ce), ctab = contingency.crosstab(row, col)
    assert np.issubdtype(re.dtype, int)
    # Col will be casted into `<U21` implicitly.
    assert np.issubdtype(ce.dtype, str)

    row = [1, 1, 2, 2, 3, pd.NA]
    col = [1, 1, 1, 3, 3, 3]
    # `scipy.contingency.crosstab` can only handle sortable array as
    # `ndarray.argsort` is called, namely umcompared mixed dtype are not
    # allowed.
    with pytest.raises(TypeError):
        (re, ce), ctab = contingency.crosstab(row, col)
    # While in the meantime, `pd.crosstab` will ignore NA.
    # But `scipy.contigency.crosstab` is 20 times faster than `pd.crosstab`.
    ret = pd.crosstab(row, col)
    assert len(ret.index) == 3
