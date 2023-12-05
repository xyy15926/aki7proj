#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: biclf.py
#   Author: xyy15926
#   Created: 2023-04-23 14:32:37
#   Updated: 2023-04-25 14:29:15
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import kendalltau
from sklearn.tree import DecisionTreeClassifier

from ringbear.scier import calculate_criterion_from_tree
from ringbear.unifier import unify_shape22

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


FACTOR_MAX = 30
POS_FLAG = 999999
NEG_FLAG = -999999
STRIP_LOOPS_MAX = 100
STRIP_LOOPS_MIN = 5


# %%
def lift_ordered(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    acc_keys: callable | list | None = None,
    ascending: bool | None = None,
    return_counts: bool = False,
    return_cor: bool = False,
) -> tuple:
    """
    Description:
    Calculate lift for `X`, `y`. And accumulatd lift could also be returned
    if ascending if True or False, which determines how to accumulate.

    Params:
    X: sortable
    y: consisting of 0, 1
    weights: weights of each record, default 1s
    acc_keys: keys for caculating accumulating lifts
    ascending: if lifts ascends with given `acc_keys`'s order, which should
        be `False` at most cases.
        None: `acc_keys`'s lifts and responsible order will be use to
            calculate kendell-correlations to determine this.
        False:
        True:
    return_counts:
    return_cor:

    Return:
    unis, lifts, acc_keys, acc_lifts, ascending
    ascending: if lifts ascends with given `acc_keys`'s order
    acc_keys: always be in descending, more likely, order, no matter what
        `ascending` is
    """
    X, y, weights = unify_shape22(X, y, weights)
    arr = np.concatenate((X, y, weights * y), axis=1)

    # Sort `arr` according to `X` for splitting, since `X` must be sortable.
    arr = arr[np.argsort(arr[:, 0])]
    unis, indices, counts = np.unique(
        arr[:, 0], return_counts=True, return_index=True
    )
    ones_ratio = np.sum(arr[:, 2]) / arr.shape[0]
    # `splited_weights` is formatted with
    # [(1-ratio-per-group, len-per-group), ...]
    splited_weights = np.array(
        [
            (subarr[:, 2].sum(), subarr.shape[0])
            for subarr in np.split(arr, indices[1:])
        ]
    )

    # Set `unis` as default `acc_keys`
    if acc_keys is None:
        acc_keys = unis.copy()
        acc_pos = np.arange(len(acc_keys))
    else:
        acc_keys = np.array(acc_keys)
        acc_pos = np.searchsorted(unis, acc_keys)

    # `lifts` should be calculcated under all conditions.
    lifts = splited_weights[:, 0] / (splited_weights[:, 1] * ones_ratio)

    # If `ascending` is not specified, Kendall-corr will be use to determine
    # proper sort order.
    corr_ken, pv = kendalltau(lifts[acc_pos], np.arange(len(acc_pos), 0, -1))
    if ascending is None:
        # Note that `ascending` will be `np.bool_` if it's set by
        # `corr_ken < 0`, thus `ascending is True` will be False.
        ascending = bool(corr_ken < 0)

    # Only when `ascending` is True, `sample_weights`, `unis` and `lifts`
    # should be reversed.
    if ascending:
        acc_keys, acc_pos = acc_keys[::-1], acc_pos[::-1]
        logger.info(
            "Provided `acc_keys` will be reversed to ensure "
            "descending lifts."
        )
    splited_weights = splited_weights[acc_pos]

    # `splited_weights` has been selected and filtered before for
    # accumulating lifts.
    acc_lifts = np.add.accumulate(splited_weights[:, 0]) / (
        np.add.accumulate(splited_weights[:, 1]) * ones_ratio
    )

    rets = [unis, lifts, acc_keys, acc_lifts, ascending]
    if return_cor:
        rets.extend([corr_ken, pv])
    if return_counts:
        rets.append(counts)

    return rets


# %%
def woe_ordered(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> tuple:
    """
    Description:
    Calculate woes, ivs for each elements in given orderable X.

    Params:
    X:
    y:
    weights:

    Return:
    uniques, woes, ivs
    """
    X, y, weights = unify_shape22(X, y, weights)
    arr = np.concatenate((X, y, weights * y, weights * (1 - y)), axis=1)

    # Sort `arr` according to `X` for splitting.
    arr = arr[arr[:, 0].argsort()]
    uniques, indices = np.unique(arr[:, 0], return_index=True)

    # `splited_weights` is formatted with
    # [(1-ratio-per-group, len-per-group), ...]
    ones_weights, zeros_weights = np.sum(arr[:, 2]), np.sum(arr[:, 3])
    one_zero = np.array(
        [
            (
                subarr[:, 2].sum() / ones_weights,
                subarr[:, 3].sum() / zeros_weights,
            )
            for subarr in np.split(arr, indices[1:])
        ]
    )

    # Calculate woes and ives for each split.
    woes = np.log2(one_zero[:, 0] / one_zero[:, 1])
    ivs = (one_zero[:, 0] - one_zero[:, 1]) * woes
    ivs[~np.isfinite(ivs)] = 0

    return uniques, woes, ivs


# %%
def iv_ordered(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """
    Description:
    Calculate iv for 2-D X.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.apply_along_axis(
        lambda X: woe_ordered(X, y, weights)[2].sum(), 0, X
    )


# %%
def select_nodes_with_freqs(
    tree: DecisionTreeClassifier,
    freq_thresh: float = 0.3,
    std_thresh: float = 0.1,
    Xs: list[np.ndarray] = None,
    ys: list[np.ndarray] = None,
) -> list:
    """
    Description:
    Select nodes with frequency meets with following requirements for all
    `Xs` and `ys`:
    1. High frequency
    2. Stability

    Params:
    tree:
    freq_thresh:
    std_thresh:
    Xs:
    ys:

    Return:
    node indices
    """
    # Take tree's original capacities and frequencies into consideration.
    Xs = [None,] if Xs is None else [None, *Xs]
    ys = [None,] if ys is None else [None, *ys]
    # Traverse to get all capacities and frequencies of all the nodes.
    freqss, capss = [], []
    for X, y in zip(Xs, ys):
        freqs, caps = calculate_criterion_from_tree(tree, "freqs", X, y)
        freqss.append(freqs[:, -1])
        capss.append(caps)
    freqss = np.array(freqss)

    # Check frequencies of all the nodes for all `Xs` and `ys`.
    node_map = np.ones(tree.tree_.node_count, dtype=np.bool_)
    # 1. Amount.
    node_map &= np.all(freqss >= freq_thresh, axis=0)
    # 2. Stability.
    node_map &= np.std(freqss, axis=0) < std_thresh

    return np.arange(tree.tree_.node_count)[node_map]
