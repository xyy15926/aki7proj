#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_azekaban.py
#   Author: xyy15926
#   Created: 2023-10-07 16:46:21
#   Updated: 2023-10-07 16:47:19
#   Description:
# ---------------------------------------------------------

# %%
import pandas as pd
import numpy as np
from azkaban.tag_analysis import pivot_tags


# %%
def test_pivot_tags():
    tags = pd.Series(["a,b,c,c", "b", "a,c", "a,b"])
    assert np.all(pivot_tags(tags).values == [[1, 1, 2], [0, 1, 0],
                                              [1, 0, 1], [1, 1, 0]])
    tags = pd.Series(["a,b,c,c", "", "a,c", "a,b"])
    assert np.all(pivot_tags(tags).values == [[0, 1, 1, 2], [1, 0, 0, 0],
                                              [0, 1, 0, 1], [0, 1, 1, 0]])
    tags = pd.Series(["a,b,c,c", np.nan, "a,c", "a,b"])
    assert np.all(pivot_tags(tags).values == [[0, 1, 1, 2], [1, 0, 0, 0],
                                              [0, 1, 0, 1], [0, 1, 1, 0]])
