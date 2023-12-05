#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: npam.py
#   Author: xyy15926
#   Created: 2023-03-28 21:17:46
#   Updated: 2023-04-23 14:55:11
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import numpy as np
from scipy.stats import kendalltau
from collections import defaultdict
import logging
from ringbear.unifier import unify_shape22


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


FACTOR_MAX = 30
POS_FLAG = 999999
NEG_FLAG = -999999
STRIP_LOOPS_MAX = 100
STRIP_LOOPS_MIN = 5


# %%
def get_outliers_with_sigma(
    arr: np.ndarray,
    sigma_n: int = 3,
    loops: int = 1,
    strip_loops: int | bool = True,
) -> tuple:
    """
    Description:
    Get outliers in `arr` according the 3-sigma rule.

    Params:
    arr:
    sigma_n:
    loops: how many times outliers will be checked at most

    Return:
    lowiers, highiers, outlier_map
    """
    outlier_map = np.full_like(arr, False, dtype=np.bool_)
    # Execlude outliers in loops
    for i in range(loops):
        in_arr = arr[~outlier_map]
        mean, std = np.nanmean(in_arr), np.nanstd(in_arr)
        _outlier_map = np.abs(arr - mean) >= (sigma_n * std)
        if np.all(~_outlier_map):
            break
        outlier_map |= _outlier_map

    if strip_loops is True:
        strip_loops = arr.shape[0] < STRIP_LOOPS_MAX and \
            arr.shape[0] > STRIP_LOOPS_MIN
    for i in range(strip_loops):
        for idx in range(arr.shape[0]):
            if outlier_map[idx] or np.isnan(arr[idx]):
                continue
            outlier_map[idx] = True
            in_arr = arr[~outlier_map]
            mean, std = np.nanmean(in_arr), np.nanstd(in_arr)
            outlier_map[idx] = np.abs(arr[idx] - mean) >= (sigma_n * std)

    if (~outlier_map).sum() > 0:
        _max, _min = np.nanmax(arr[~outlier_map]), np.nanmin(arr[~outlier_map])
    else:
        _max, _min = np.nan, np.nan

    # Record outliers before removed
    lowiers = np.unique(arr[arr < _min])
    highiers = np.unique(arr[arr > _max])

    return lowiers, highiers, outlier_map


def remove_outlier_with_sigma(
    arr: np.ndarray,
    sigma_n: int = 3,
    loops: int = 1,
    max_replace: int | float | callable | None = POS_FLAG,
    min_replace: int | float | callable | None = NEG_FLAG,
    return_outliers: bool = False,
    inplace: bool = False,
) -> np.ndarray:
    """
    Description:
    Remove outliers in `arr` according the 3-sigma rule.

    Params:
    arr:
    sigma_n:
    loops: how many times outliers will be checked at most
    max_replace:
    min_replace:

    Return:
    arr, lowiers, highiers
    """
    lowiers, highiers, outlier_map = get_outliers_with_sigma(
            arr, sigma_n, loops)
    _max, _min = arr[~outlier_map].max(), arr[~outlier_map].min()

    # Replace outliers with `max_replace`, `min_replace`
    arr_copy = arr.copy() if not inplace else arr
    if callable(max_replace):
        max_replace = max_replace(arr_copy[~outlier_map])
        arr_copy[arr_copy > _max] = max_replace
    if callable(min_replace):
        min_replace = min_replace(arr_copy[~outlier_map])
        arr_copy[arr_copy < _min] = min_replace

    if return_outliers:
        return arr_copy, lowiers, highiers
    else:
        return arr_copy


# %%
def calculate_criterion(
    caps: list | tuple | np.ndarray,
    criterion: str = "freqs"
) -> float:
    """
    Description:
    Calculate criterion, including frequencies, entropy and gini-index with
    capacities.

    Params:
    caps: ndarray storing the numbers of each events, with shape
        (ROWS, EVENT_N)
    criterion:

    Return:
    ndarray with shape (ROWS, CRITERION_SHAPE)
    criterion_SHAPE: EVENT_N for freqs and 1 for entropy and gini
    """
    freqs = caps / caps.sum(axis=1, keepdims=True)
    if criterion in ("freqs", "freq", "frequency"):
        return freqs
    elif criterion in ("entropy", "ent"):
        ent = -np.sum(freqs * np.log2(freqs), axis=1)
        return ent
    elif criterion == "gini":
        gini = np.sum(freqs * (1 - freqs), axis=1)
        return gini
    else:
        logger.warning("Unsupported criterion %s to calculate with "
                       "capacities.", criterion)
        return None


# # %%
# def build_order_encoder(
#     eles: list,
#     order_key: callable | dict | list | None,
#     default: int = NEG_FLAG
# ) -> dict:
#     """
#     Description:
#     Build encoder mapper with values set according to given `order`, which is
#         used to map `eles` for reordering `eles`.

#     Params:
#     eles: elements that need to be mapped for reordering
#     order_key:
#         list specifying order for `eles`
#         dict or callable specifying mapper
#     default: how to deal with elements in eles that's not specified in
#         `order_key`

#     Return:
#     """
#     mapper = defaultdict(lambda: default)
#     if isinstance(order_key, list):
#         mapper.update({i: order_key.index(i) for i in eles})
#     elif isinstance(order_key, dict):
#         mapper.update({i: order_key.get(i, default) for i in eles})
#     elif callable(order_key):
#         mapper.update({i: order_key(i) for i in eles})

#     return mapper


# def argsort_1d_keyed(
#     a: np.ndarray,
#     order_key: callable | dict | list | None = None,
#     default_key: int | str | None = None,       # todo
#     return_map: bool = False,
#     **kwargs,
# ) -> np.ndarray | tuple:
#     """
#     Description:
#     Argsort `a` with order determined by `key`.
#     Attention:
#     Only 1-d array are tested.

#     Params:
#     a:
#     order_key: o
#     default_key:
#     return_map:
#     kwargs: keyword params for np.argsort
#         axis:
#         kind:
#         order:

#     Return:
#     """
#     if order_key is None:
#         return np.argsort(a, **kwargs)

#     unis, inv = np.unique(a, return_inverse=True)
#     if len(unis) > FACTOR_MAX:
#         logger.warning("To many, %s, factors are used as keys for argsort.",
#                        len(unis))

#     mapper = build_order_encoder(unis, order_key, default_key)
#     a = np.array([mapper[i] for i in unis])[inv].reshape(a.shape)
#     sorted_indices = np.argsort(a, **kwargs)

#     if return_map:
#         return sorted_indices, mapper, a
#     else:
#         return sorted_indices


# %%
# def lift_ordered(
#     X: np.ndarray,
#     y: np.ndarray,
#     weights: np.ndarray | None = None,
#     return_counts: bool = False,
#     ascending: bool | None = None,
#     order_key: callable | dict | list | None = None,
# ) -> tuple:
#     """
#     Description:
#     Calculate lift for `X`, `y`. And accumulatd lift could also be returned
#     if ascending if True or False, which determines how to accumulate.

#     Params:
#     X:
#     y:
#     weights:
#     return_counts:
#     ascending:
#     key:

#     Return:
#     """
#     X, y, weights = unify_shape22(X, y, weights)
#     arr = np.concatenate((X, y, weights * y), axis=1)

#     # Sort `arr` according to `X` for splitting.
#     arr = arr[np.argsort(arr[:, 0])]
#     unis, indices, counts = np.unique(
#         arr[:, 0], return_counts=True, return_index=True)
#     ones_ratio = np.sum(arr[:, 2]) / arr.shape[0]
#     # `splited_weights` is formatted with
#     # [(1-ratio-per-group, len-per-group), ...]
#     splited_weights = np.array(
#         [(subarr[:, 2].sum(), subarr.shape[0])
#          for subarr in np.split(arr, indices[1:])])

#     # For categorical `X`, only return `liftes`
#     # if ascending is None:
#     #     liftes = splited_weights[:, 0] / (splited_weights[:, 1] * ones_ratio)
#     #     if return_counts:
#     #         return unis, liftes, counts
#     #     else:
#     #         return unis, liftes

#     # For ordered `X`, return both `liftes` and `acc_liftes`
#     if order_key is not None:
#         mapper = build_order_encoder(unis, order_key, NEG_FLAG)
#         sort_order = np.argsort(np.array([mapper[i] for i in unis]))
#     else:
#         sort_order = np.argsort(unis)

#     # `lifts` should be calculcated under all conditions.
#     unis = unis[sort_order]
#     splited_weights = splited_weights[sort_order]
#     lifts = splited_weights[:, 0] / (splited_weights[:, 1] * ones_ratio)

#     # If `ascending` is not specified, Kendall-corr will be use to determine
#     # proper sort order.
#     if ascending is None:
#         corr_ken, pv = kendalltau(lifts, unis)
#         ascending = corr_ken < 0

#     # Only when `ascending` is False, `sample_weights`, `unis` and `lifts`
#     # should be reversed.
#     if ascending is False:
#         unis = unis[::-1]
#         splited_weights = splited_weights[::-1]
#         lifts = lifts[::-1]
#         counts = counts[::-1]

#     acc_lifts = np.add.accumulate(splited_weights[:, 0]) / \
#         (np.add.accumulate(splited_weights[:, 1]) * ones_ratio)

#     if return_counts:
#         return unis, lifts, acc_lifts, corr_ken, pv, counts
#     else:
#         return unis, lifts, acc_lifts, corr_ken, pv


