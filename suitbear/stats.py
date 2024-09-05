#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: stats.py
#   Author: xyy15926
#   Created: 2024-09-03 21:55:38
#   Updated: 2024-09-04 15:45:12
#   Description:
#   Ref: <http://staff.ustc.edu.cn/~rui/ppt/modeling/modeling_8.html>
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, TypeVar
from collections.abc import Callable, Iterator
from functools import lru_cache
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from IPython.core.debugger import set_trace

import numpy as np
import pandas as pd
import numpy.linalg as linalg

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def AHP_solve(
    hmats: list[list[np.ndarray]],
    domi: bool = True,
) -> np.ndarray:
    """Solve AHP matrix of comparative pair-importance.

    Params:
    ------------------------
    hmats: List among which representes one level of the hierarchy. And the
      matrixs represent comparative pair-importance for the upper level
      features of the whole current features.
    domi: If to rectify the weights of with the number of dominant features.

    Examples:
    ------------------------
    1. Full connections hierarchy.
      Goal:         Goal
      Dims: A       B       C       D
      STGs: S1      S1      S3
      The goal could be assess from dimensions A, B, C, D, and the strategies
      may achive differently for these dimensions. Then the hmats should be:
      [[4*4, ],
       [3*3, 3*3, 3*3, 3*3]]
    2. Non-full connections hierarchy.
      Goal:         Goal
      Dims: A           B               C
      LDim: A1  A2      B1  B2  B3      C1  C2
      The goal could be assess from dimensions A, B, C, and the dimensions
      could be cut into sub-dims. Then the hmats should be:
      [[3*3],
       [7*7, 7*7, 7*7]]
      But the 7*7 matrixs will be filled with 0s for those comparsions like
      A1-B1, A2-C1.

    Return:
    ------------------------
    The final weights of bottom features.
    """
    final_w = None
    for lv, mat_comps in enumerate(hmats):
        comp_ws = []
        CIs = []
        domi_n = []
        for mat in mat_comps:
            fea_n = mat.shape[0]
            eig_vals, eig_vecs = linalg.eig(mat)
            max_idx = np.argmax(eig_vals)
            # Get the maximum eigen value and correspondant eigen vector.
            eig_val, eig_vec = eig_vals[max_idx], eig_vecs[:, max_idx]
            # Normalize the eigen vector.
            eig_vec = eig_vec / eig_vec.sum()

            logger.debug(f"Eigen value: {eig_val}.")
            logger.debug(f"Eigen vector: {eig_vec}.")

            # Rnadom Consistency Index Test.
            CI = (eig_val - fea_n) / (fea_n - 1)
            if CI / AHP_get_RCI()[fea_n - 1] > 0.1:
                logger.warning(f"AHP Random Consistency Index Test failed "
                               f"for {mat}.")

            comp_ws.append(eig_vec)
            CIs.append(CI)
            domi_n.append(fea_n - np.all(mat == 0, axis=1).sum())

        weight = np.vstack(comp_ws)
        if final_w is None:
            final_w = weight
        else:
            # Rectify the weigths with number of dominate features.
            final_w = final_w * domi_n
            final_w /= np.sum(final_w)
            # Rnadom Consistency Index Test.
            if np.sum(np.array(CI) * final_w) > 0.1:
                logger.warning(f"AHP Random Consistency Index Test failed for "
                               f"level-{lv}.")
            final_w = np.dot(final_w, weight)

    return final_w


# %%
@lru_cache
def AHP_get_RCI():
    """Get the pre-caled RCI from 1 to 15.
    """
    RI = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41,
          1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]

    return RI


@lru_cache
def AHP_get_RCI_n(n: int) -> float:
    """Calculate Random Consistency Index.

    Params:
    --------------------
    n: The number of the features.

    Return:
    --------------------
    The RCI for `n`.
    """
    evals = []
    nn = np.arange(1, 10)
    nn = np.concatenate([nn, (1 / nn)[1:]])
    for i in range(1000):
        # Random choice from 1, 10 and their reciprocal.
        a = np.random.choice(nn, (n, n)).astype(np.float_)
        a[np.diag_indices_from(a)] = 1
        a[np.tril_indices_from(a)] = 1 / a.T[np.tril_indices_from(a.T)]
        evs = linalg.eigvals(a)
        evals.append(evs.max())
    evals = np.array(evals)
    rci = np.mean((evals - n) / (n - 1)).real
    return rci
