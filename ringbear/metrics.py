#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: metrics.py
#   Author: xyy15926
#   Created: 2023-04-23 14:32:37
#   Updated: 2023-12-20 19:46:19
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import kendalltau

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def cal_lifts(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    acc_keys: callable | list | None = None,
    ascending: bool | None = None,
    return_counts: bool = False,
    return_cor: bool = False,
) -> tuple:
    """Calculate lifts.

    Calculate lifts, accumulating lift for 1-D `X` and 1-D `y`.
    1. If no specific `acc_keys` are given, `acc_keys` will be all uniques
      factors in `x`.
    2. Accumulating order is determined by sorting `acc_keys` and checking
      Kendall-tau between lifts and sorted `acc_keys` if not specified.

    Params:
    ----------------
    x: 1-D NDA with sortable elements.
    y: 1-D NDA filled with 0 or 1.
    weights: Weights of each record, default 1s.
    acc_keys: List of keys in `x` for caculating accumulating lifts.
      list: Only keys in `acc_keys` will be used to calculate accumulating
        lifts. Attention: calculating accumulating lifts forward or backward
        is determined by `ascending`, instead of this directly.
      None: Unique factors in `x`.
    ascending: If lifts ascends along with given `acc_keys`'s order.
      None: Kendall-tau between `acc_keys`'s lifts and responsible order
        will be use to determine this.
      False: Lifts of `acc_keys` should ascend along with it's order.
      True: Contrary to the above.
    return_counts: If return counts of each uniques in `x`.
    return_cor: IF return Kendall-tau correlations.

    Return:
    ----------------
    unis: Uniques in `x`.
    lifts: Lift of all uniques.
    acc_keys: Keys for calculating accumulating lifts.
    ascending: If lifts ascends along with `acc_keys`.
    corr_ken: Kendall-tau correlation.
    pv: P-value of Kendall-tau.
    counts: Value-counts of uniques in `x`.
    """
    if weights is None:
        arr = np.column_stack((x, y, y))
    else:
        arr = np.column_stack((x, y, weights * y))
    # Sort `arr` according to `x` for splitting, since `x` must be sortable.
    arr = arr[np.argsort(arr[:, 0])]
    unis, indices, counts = np.unique(arr[:, 0], return_counts=True,
                                      return_index=True)
    ones_ratio = np.sum(arr[:, 2]) / arr.shape[0]
    # `splited_weights`: [(1-ratio-per-group, len-per-group), ...]
    splited_weights = np.array([(subarr[:, 2].sum(), subarr.shape[0])
                                for subarr in np.split(arr, indices[1:])])
    lifts = splited_weights[:, 0] / (splited_weights[:, 1] * ones_ratio)

    # Copy all unique factors as `acc_keys` in default.
    if acc_keys is None:
        acc_keys = unis.copy()
        acc_pos = np.arange(len(acc_keys))
    else:
        acc_keys = np.array(acc_keys)
        acc_pos = np.searchsorted(unis, acc_keys)

    # Kendall-tau will be use to determine proper sorting order if
    # `ascending` is not specified.
    corr_ken, pv = kendalltau(lifts[acc_pos], np.arange(len(acc_pos), 0, -1))
    if ascending is None:
        # Attention: `ascending` will be `np.bool_` if not cast explicitly
        # and `is True` is False.
        ascending = bool(corr_ken < 0)
    # Only when `ascending` is True, `sample_weights`, `unis` and `lifts`
    # should be reversed.
    if ascending:
        acc_keys, acc_pos = acc_keys[::-1], acc_pos[::-1]
        logger.info(
            "`acc_keys` passed-in will be reversed to ensure descending lifts.")
    splited_weights = splited_weights[acc_pos]
    acc_lifts = (np.add.accumulate(splited_weights[:, 0])
                 / (np.add.accumulate(splited_weights[:, 1]) * ones_ratio))

    rets = [unis, lifts, acc_keys, acc_lifts, ascending]
    if return_cor:
        rets.extend([corr_ken, pv])
    if return_counts:
        rets.append(counts)

    return rets


# %%
# TODO: Extend WOEs from 2-Classification to MultiClf.
def cal_woes(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple:
    """Calculate WOEs and IVs for each elements.

    Calculate WOEs, IVs of `x` and `y` each elements in sortable `x`.

    Params:
    ----------------
    x: 1-D NDA with sortable elements.
    y: 1-D NDA filled with 0 or 1.
    weights: Weights of each record, default 1s.

    Return:
    ----------------
    unis: Uniques in `x`.
    woes: WOEs of each uniques.
    ivs: IVs of each uniques.
    """
    # Sort `arr` according to `x` for splitting.
    if weights is None:
        arr = np.column_stack((x, y, y, 1 - y))
    else:
        arr = np.column_stack((x, y, weights * y, weights * (1 - y)))
    arr = arr[arr[:, 0].argsort()]
    unis, indices = np.unique(arr[:, 0], return_index=True)

    # `splited_weights` is formatted with
    # [(1-ratio-per-group, len-per-group), ...]
    one_ws, zero_ws = np.sum(arr[:, 2]), np.sum(arr[:, 3])
    one_zero = np.array([
        (subarr[:, 2].sum() / one_ws, subarr[:, 3].sum() / zero_ws,)
        for subarr in np.split(arr, indices[1:])])

    # Calculate woes and ives for each split.
    woes = np.log2(one_zero[:, 0] / one_zero[:, 1])
    ivs = (one_zero[:, 0] - one_zero[:, 1]) * woes
    ivs[~np.isfinite(ivs)] = 0

    return unis, woes, ivs


# %%
def cal_ivs(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None
) -> float:
    """Calculate IVs for each column in 2-D NDA.

    Calculate IVs of each columns in 2-D X or just IV for 1-D x. These IVs
    are different from the IVs returned from `woes_ordered`, which are IVs
    of each uniques.

    Params:
    ----------------
    X: 2-D NDA with sortable elements.
    y: 1-D NDA filled with 0 or 1.
    weights: Weights of each record, default 1s.

    Return:
    ----------------
    1-D NDA of IVs of each columns of X.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.apply_along_axis(
        lambda X: cal_woes(X, y, weights)[2].sum(), 0, X
    )
