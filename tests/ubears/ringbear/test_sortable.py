#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_sortable.py
#   Author: xyy15926
#   Created: 2024-01-17 18:55:12
#   Updated: 2025-01-20 17:49:53
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from ubears.ringbear import sortable
    from ubears.ringbear import numeric
    reload(sortable)
    reload(numeric)

import scipy.stats as ss
from sklearn.datasets import load_iris
from scipy.stats import contingency
from ubears.ringbear.numeric import edge_encode
from ubears.ringbear.sortable import tree_cut, chimerge_cut


# %%
def test_tree_cut():
    X, y = load_iris(return_X_y=True)
    edges, ctab = tree_cut(X[:, 0], y)
    ords = edge_encode(X[:, 0], edges)
    (ux, uy), cctab = contingency.crosstab(ords, y)
    assert np.all(np.isclose(ctab, cctab))


def test_chimerge_cut():
    X, y = load_iris(return_X_y=True)
    edges, ctab = chimerge_cut(X[:, 0], y)
    ords = edge_encode(X[:, 0], edges)
    (ux, uy), cctab = contingency.crosstab(ords, y)
    assert np.all(np.isclose(ctab, cctab))



