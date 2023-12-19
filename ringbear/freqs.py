#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#   Name: freqs.py
#   Author: xyy15926
#   Created: 2023-12-06 18:12:05
#   Updated: 2023-12-11 14:30:40
#   Description: Module with functions handles 2-D NDA of frequencies.
#
# In most cases, the first parameters of the function in this module should be
#   a 1-D or 2-D NDA, among which the elements should be integer representing
#   the frequencies.
# For example:
#     X   1    2    3
#   Y
#   1     5    5    1
#   2     2    3    5
#   Cross table above indicates frequencies of sample(X=1, Y=2) is 2.
# -----------------------------------------------------------------------------

# %%
from __future__ import annotations
import logging
import sys

import numpy as np
from scipy.stats import contingency

MAXSIZE = sys.maxsize

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# TODO: Implement gropuby refering to `enhanced_freqs`.
def gropuby(
    keys: list,
    values: np.ndarray | None = None,
    agg: callable | None = None,
    *,
    pivot: bool = False,
) -> tuple:
    pass


# %%
def enhanced_freqs(
    rows: np.ndarray,
    cols: np.ndarray | None = None,
    *,
    others: np.ndarray | None = None,
    agg: callable | None = None,
) -> tuple:
    """Aggregate for tuple-2 or tuple-1 pairs.

    For different pairs derived from zipping `rows` and `cols`, this function
    will aggregate responsible blocks in `others` determined by the pairs
    seperately. If only `rows` passed, pairs will be the uniques in `rows`.
    1. If neither `others` nor `agg` are passed, this functions behaves just
      like the `scipy.contingency.crosstab` or `value_counts` to calculate
      the frequencies.
    2. The perm-mask ways used here are refered from `np.unique` defined in
      `arraysetops.py`.
    3. `np.lexsort` instead of `np.argsort` is used to get index of sorted
      array when both `rows` and `cols` are passed.
    4. When both `rows` and `cols` passed, `agg` must return a scalar.
      Meanwhile a 1-D NDA is allowed when `cols` is None.

    Params:
    ---------------
    rows: 1-D NDA
      Sortable array as row.
    cols: 1-D NDA
      Sortable array as column.
    others: NDA | None
      Data to be aggragated for pairs derived from rows and columns.
    agg: callable
      Aggregation to be applied on blocks of others.

    Return:
    ---------------
    ret: 2-D NDA
      Array of the aggregation result, default as np.nan if responsible pair
      of `rows` and `cols` not existing.
    unisr: 1-D NDA
      Unique values in `rows`.
    unisc: 1-D NDA
      Unique values in `cols`.
    """
    if cols is None:
        return _enhanced_freqs_1D(rows, others, agg=agg)
    else:
        return _enhanced_freqs_2D(rows, cols, others, agg)


def _enhanced_freqs_1D(
    arr: np.ndarray,
    others: np.ndarray | None = None,
    agg: callable | None = None,
) -> tuple:
    """Aggregate for 1-D array.

    This function will aggregate responsible blocks in `others` determined by
    the uniques in `arr`.
    1. If neither `others` nor `agg` are passed, this functions behaves just
      like the `value_count` to calculate the frequencies.
    2. The perm-mask ways used here are refered from `np.unique` defined in
      `arraysetops.py`.

    Params:
    ---------------
    arr: 1-D NDA
      Sortable array.
    others: NDA | None
      Data to be aggragated for pairs derived from rows and columns.
    agg: callable
      Aggregation to be applied on blocks of others.

    Return:
    ---------------
    ret: 2-D NDA
      Array of the aggregation result.
    unis: 1-D NDA
      Unique values in `arr`.
    """
    if others is None:
        others = np.ones(arr.shape, dtype=np.int_)
    if agg is None:
        agg = np.sum

    # Argsort to get index of sorted array.
    perm = np.argsort(arr)
    aux = arr[perm]

    # Set the mask of uniques.
    mask = np.empty(aux.size, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    unis = aux[mask]

    where = np.concatenate(np.nonzero(mask) + ([mask.size], ))
    auxo = others[perm]
    ret = np.row_stack([agg(auxo[s:e])
                        for s, e in zip(where[:-1], where[1:])])

    return ret.squeeze(), unis


# TODO: Implement for 3D or more general NDA.
# TODO: Move the last part as pivot out of the function. Maybe just like
#   sparse matrix?
def _enhanced_freqs_2D(
    rows: np.ndarray,
    cols: np.ndarray,
    others: np.ndarray | None = None,
    agg: callable | None = None,
) -> tuple:
    """Aggregate for tuple-2 pairs from two 1-D array.

    For different pairs derived from zipping `rows` and `cols`, this function
    will aggregate responsible blocks in `others` determined by the pairs
    seperately.
    1. If neither `others` nor `agg` are passed, this functions behaves just
      like the `scipy.contingency.crosstab` to calculate the frequencies
      but the order of the return.
    2. The perm-mask ways used here are refered from `np.unique` defined in
      `arraysetops.py`.
    3. `np.lexsort` instead of `np.argsort` is used to get index of sorted
      array.

    Params:
    ---------------
    rows: 1-D NDA
      Sortable array as row.
    cols: 1-D NDA
      Sortable array as column.
    others: NDA | None
      Data to be aggragated for pairs derived from rows and columns.
    agg: callable
      Aggregation to be applied on blocks of others.

    Return:
    ---------------
    ret: 2-D NDA
      Array of the aggregation result, default as np.nan if responsible pair
      of `rows` and `cols` not existing.
    unisr: 1-D NDA
      Unique values in `rows`.
    unisc: 1-D NDA
      Unique values in `cols`.
    """
    if others is None:
        others = np.ones(rows.shape, dtype=np.int_)
    if agg is None:
        agg = np.sum

    # Argsort to get index of sorted array.
    perm = np.lexsort((cols, rows))
    arr = np.column_stack((rows, cols))
    aux = arr[perm]
    permc = np.argsort(cols)
    auxr = cols[permc]

    # Set the mask of uniques.
    mask = np.empty(rows.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = auxr[1:] != auxr[:-1]
    unisc = auxr[mask]
    mask[1:] = aux[1:, 0] != aux[:-1, 0]
    unisr = aux[mask, 0]
    mask[1:] = np.any(aux[1:, :] != aux[:-1, :], axis=1)

    # Set the aggregation result.
    auxo = others[perm]
    ret = np.full((unisr.size, unisc.size), np.nan)
    where = np.concatenate(np.nonzero(mask) + ([mask.size], ))
    for start, end in zip(where[:-1], where[1:]):
        r = np.searchsorted(unisr, aux[start, 0])
        c = np.searchsorted(unisc, aux[start, 1])
        ret[r, c] = agg(auxo[start: end])

    return ret, unisr, unisc


# %%
def cal_entropy(
    freqs: np.ndarray,
    axis: int | None = -1,
    keepdims: bool = False,
) -> np.ndarray:
    """Calculate the entropy.

    Calculate the entropy of the given frequencies:
      entropy = -sum(p * log2(p)), [0, +inf)
    1. Bit entropy will be returned as `log2` is used.
    2. Larger entropy, more choas.

    Params:
    ---------------
    freqs: 2-D NDA
      Frequency array.
    axis: int | None
      int: Axis to calculate the entropy along with.
        Default with -1, as entropy will be calculated for each row.
        The axis will be shrinked after calculation.
      None: Calculate the entropy with `freq` as a whole.
    keepdims: bool
      Whether keep the dimension of `freqs` unchanged.

    Return:
    ---------------
    ent: 2-D NDA of shape(-1, 1) or 1-D NDA.
    """
    freqs = freqs / freqs.sum(axis=axis, keepdims=True)
    ent = -np.sum(freqs * np.nan_to_num(np.log2(freqs), False), axis=axis)
    return ent if keepdims else ent.squeeze()


# %%
def cal_gini(
    freqs: np.ndarray,
    axis: int | None = -1,
    keepdims: bool = False,
) -> np.ndarray:
    """Calculate the GINI.

    Calculate the entropy of the given frequencies.
      GINI = sum(p * (1-p)), [0, 1]
    1. Larger GINI, more choas.

    Params:
    ---------------
    freqs: 2-D NDA
      Frequency array.
    axis: int | None
      int: Axis to calculate the GINI along with.
        Default with -1, as GINI will be calculated for each row.
        The axis will be shrinked after calculation.
      None: Calculate GINI with `freq` as a whole.
    keepdims: bool
      Whether keep the dimension of `freqs` unchanged.

    Return:
    ---------------
    gini: 2-D NDA of shape(-1, 1) or 1-D NDA.
    """
    freqs = freqs / freqs.sum(axis=axis, keepdims=True)
    gini = np.sum(freqs * (1 - freqs), axis=axis)
    return gini if keepdims else gini.squeeze()


# %%
def chi_pairwisely(
    freqs: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Calculate the Chis of adjacent bins.

    Calculate the Chis of adjacent rows of columns.
    1. Treat rows or columns as bins and each elements representing the number
      of each category.

    Params:
    ---------------
    freqs: 2-D NDA
      Cross table.
    axis: int
      Axis to calculate the Chis along with.
      Default with 0, as Chis will be calculated for each pair of adjacent
        rows.
      THe axis will remains with decreasing 1 after calculation.

    Return:
    ---------------
    chis: [[chi, p-value], ...]
    """
    if np.any(freqs == 0):
        logger.warning("Invalid frequencies to calculate Chis. "
                       "Columns or rows with zeros may be merged first.")

    # Check how to stagger `freqs`.
    if axis % 2 == 0:
        s_former = np.s_[:-1, :]
        s_latter = np.s_[1: , :]
    else:
        s_former = np.s_[:, :-1]
        s_latter = np.s_[:, 1 :]
    chis = np.apply_along_axis(
        lambda x: contingency.chi2_contingency(x.reshape(2, -1))[0:2],
        axis=(axis + 1) % 2,
        arr=np.concatenate((freqs[s_former], freqs[s_latter]),
                           axis=(axis + 1) % 2)
    )
    # Another implementation with `map`, but much more slower.
    # chis = np.array(list(map(
    #     lambda x: contingency.chi2_contingency(x)[0:2],
    #     zip(freqs[:-1, :], freqs[1:, :]))))
    return chis
