#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: numeric.py
#   Author: xyy15926
#   Created: 2023-03-28 21:17:46
#   Updated: 2023-12-11 14:46:06
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import logging
import numpy as np


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


POS_FLAG = 999999
NEG_FLAG = -999999


# %%
def std_outlier(
    arr: np.ndarray,
    sigma_n: int = 3,
    loops: int = 1,
    excluded: bool = False,
) -> tuple:
    """Get outerlier according to the std-variance.

    Get outliers according to the standard variance.
    1. Mean and std will be calculated to determine the outliers.
    2. Std, mead will change after exluding some outliers, so looping to get
      more precisely outliers is necessary.
    3. Two different pair of mean and std, responsible to whether excluding
      current element(or elements with the same value), could be used to
      check if current element is outliers, which could be choosed by
      `excluded`.

    Params:
    -----------------
    arr: 1-D NDA
      Sortable array.
    sigma_n: int
      The threshold to determining the outliers.
    loops: int
      How many times outliers will be checked at most.
    excluded: bool
      Whether excluding current element out when calculating mean and std.
      This will be much more slow as the data's scale growing.

    Return:
    -----------------
    lowiers: The unique lower outliers.
    highiers: The unique higher outliers.
    outlier_map: Bool array indicating the position of outliers.
    """
    outlier_map = np.full_like(arr, False, dtype=np.bool_)
    # Don't exclude current element when calculating mean and std.
    if not excluded:
        for i in range(loops):
            in_arr = arr[~outlier_map]
            mean, std = np.nanmean(in_arr), np.nanstd(in_arr)
            _outlier_map = np.abs(arr - mean) >= (sigma_n * std)
            # Stop early if not more outliers found.
            if np.all(~_outlier_map):
                break
            outlier_map |= _outlier_map
    # Exclude current element when calculating mean and std.
    else:
        for i in range(loops):
            for idx in range(arr.shape[0]):
                if outlier_map[idx] or np.isnan(arr[idx]):
                    continue
                outlier_map[arr == arr[idx]] = True
                in_arr = arr[~outlier_map]
                mean, std = np.nanmean(in_arr), np.nanstd(in_arr)
                outlier_map[idx] = (np.abs(arr[arr == arr[idx]] - mean)
                                    >= (sigma_n * std))

    # Get the max lowiers and min highiers so to record lower and higher outliers.
    if (~outlier_map).sum() > 0:
        _max, _min = np.nanmax(arr[~outlier_map]), np.nanmin(arr[~outlier_map])
    else:
        _max, _min = np.nan, np.nan
    lowiers = np.unique(arr[arr < _min])
    highiers = np.unique(arr[arr > _max])

    return lowiers, highiers, outlier_map


def remove_std_outlier(
    arr: np.ndarray,
    sigma_n: int = 3,
    loops: int = 1,
    max_replace: int | float | callable = POS_FLAG,
    min_replace: int | float | callable = NEG_FLAG,
    return_outliers: bool = False,
    inplace: bool = False,
) -> np.ndarray:
    """Remove outliers according to the std-variance.

    Get and remove outliers according to the standard variance.
    1. Mean and std will be calculated to determine the outliers by calling
      `get_outliers_with_sigma`.

    Params:
    -----------------
    arr: 1-D NDA
      Sortable array.
    sigma_n: int
      The threshold to determining the outliers.
    loops: int
      How many times outliers will be checked at most.
    max_replace: int | float | callable
      int | float: Value to replace the highiers.
      callable: Accept `arr` without outliers to calculate the `max_replace`.
    min_replace: int | float | callable
      Similar to the above.
    return_outliers: bool
      Whether returning outlier.
    inplace: bool
      Whether modifying `arr` inplace directly.

    Return:
    -----------------
    arr: Array with outlier removed or replaced.
    lowiers: The unique lower outliers.
    highiers: The unique higher outliers.
    """
    lowiers, highiers, outlier_map = std_outlier(arr, sigma_n, loops)
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
def span_cut(
    arr: np.ndarray,
    cuts_n: int | None = 7,
    sigma_n: float | None = 2,
) -> list:
    """Cut numeric at middle point of the largest spans.

    Set the middle point of the largest spans of the adjacent elements.
    1. The number of bins could be specified directly with `cuts_n`.
    2. Span to be segmented could also be determined by specifying the std.
      Significant wide spans will be segmented, which is:
      span - mean(span) > sigma_n * std

    Params:
    -----------------
    arr: 1-D NDA
      Sortable array.
    cuts_n: int
      int: Number of bins.
      None: Ignore this.
    sigma_n: float | None
      Float: Spans' std threshold, which will be overriden by `cuts_n`.
      None: Ignore this.

    Return:
    -----------------
    edges: 1-D NDA
      Bin edges.
    """
    # Copy sort.
    arr = np.sort(arr)
    # Calculate the spans and get the middles of the largest spans.
    spans = arr[1:] - arr[:-1]
    if cuts_n is not None:
        edge_idx = np.argsort(spans)[-cuts_n + 1:]
    else:
        highiers = std_outlier(spans, sigma_n)[1]
        edge_idx = np.where(spans >= highiers.min())[0]
    edges = (arr[edge_idx + 1] + arr[edge_idx]) / 2
    return np.array([arr.min(), *edges, arr.max()])


# %%
def ordinal_encode(
    arr: np.ndarray,
    edges: np.ndarray,
    check_range: bool = False,
) -> np.ndarray:
    """Labelize array ordinally according to given edges.

    Transform sortable array into array of labels, in which elements are
    represented by integer range from 1 to N, A.K.A. binnize.
    1. `np.searchsorted` will be used to determine the label.
    2. Then labels will be dispatched after following rules:
      1: [arr.min()     , edges[0]]
      2: (edges[1]      , edges[1]]
      ...
      N: (edges[N-1]    , arr.max()]

    Params:
    -----------------
    arr: 1-D NDA
      Sortable data array.
    edges: 1-D NDA | list
      Edges to determine the labels.
    check_range: bool
      Whether checking `arr` within the range of `edges`.
      AssertionError will be raised if not satisfied.

    Raise:
    -----------------
    AssertionError

    Return:
    -----------------
    Ordially labeled array range from 1, len(edges).
    """
    if not check_range:
        edges = edges[1: -1]
    else:
        assert np.all(arr <= edges[-1]) and np.all(arr >= edges[0])
    labeled = np.apply_along_axis(lambda x: np.searchsorted(edges, x) + 1, 0, arr)

    return labeled
