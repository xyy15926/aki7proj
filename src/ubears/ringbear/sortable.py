#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#   Name: sortable.py
#   Author: xyy15926
#   Created: 2023-12-06 21:29:38
#   Updated: 2025-01-20 17:28:45
#   Description: Modules with functions to handle sortable features.
# -----------------------------------------------------------------------------

# %%
from __future__ import annotations
import logging

import numpy as np
import pandas as pd
# from IPython.core.debugger import set_trace

from scipy.stats import contingency
from sklearn.tree import (DecisionTreeClassifier, )

from ubears.ringbear.freqs import chi_pairwisely


# %%
N_BINS = 7
FREQ_MIN = 0.05
PVALUE_MAX = 0.05
CHI_MIN = 34
IMPURITY_DECREASE_MIN = 0.0001


logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# TODO: Test cases
# TODO: Cross table
def tree_cut(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = N_BINS,
    freq_min: float = FREQ_MIN,
    min_impurity_decrease: float = IMPURITY_DECREASE_MIN,
) -> tuple:
    """Cut sortable into bins with DecisionTreeClassifier.

    Cut `x` into `n_bins` by DescisionTreeClassifier with gini as criterion.
    NOTE:
    1. Bins with frequency smaller than `freq_min` will be merged before
      considering Chi-stat.

    Params:
    -----------------
    x: 1-D NDA or 2-D NDA of shape(-1, 1)
      Array to be cut with sortable elements.
    y: 1-D NDA
      Array filled with "categorical".
    n_bins: int
      Number of bins determining when to stop cut.
    freq_min: int
      Minimum frequency of each bin.
    min_impurity_decrease: Float
      Param passed to init tree to restrict the spliting behavior.

    Return:
    -----------------
    threshs: List of bin edges for cutting.
      Note: At first, both lower and upper bound are included. But it's seems
      that `np.histogram` and `pd.cut` behave differently by default.
      `numpy` prefers left-closed interval, `pandas` vice versa. So neither
      lower bound nor higher bound is included but their exteriors.
    ctab: Cross table of final bins.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    # Fit tree classifer.
    tree = DecisionTreeClassifier(
        min_samples_leaf=freq_min,
        max_leaf_nodes=n_bins,
        min_impurity_decrease=min_impurity_decrease,
    ).fit(x, y)

    cl = tree.tree_.children_left
    cr = tree.tree_.children_right
    value = tree.tree_.value[:, 0, :]

    # Select leaf node by compare its children node with -1.
    leaf_node_map = ((cl == -1) & (cr == -1))
    # Get thresholds directly.
    threshs = np.concatenate([[np.min(x) - 1e-6],
                              np.sort(tree.tree_.threshold[~leaf_node_map]),
                              [np.max(x) + 1e-6]])

    def leaf_post_order(cl, cr):
        ctab = []
        st = []
        cur = 0
        last = None
        while st or cur is not None:
            while cur != -1 and cur is not None:
                st.append(cur)
                cur = cl[cur]
            cur = st[-1]
            if cr[cur] == last or cr[cur] == -1:
                if cr[cur] == -1:
                    ctab.append(value[cur])
                last = st.pop()
                cur = None
            else:
                cur = cr[cur]
        return np.array(ctab)

    ctab = leaf_post_order(cl, cr)

    return threshs, ctab


# %%
# TODO: Test cases
# TODO: Docs
def chimerge_cut(
    x: np.ndarray | list | pd.Series,
    y: np.ndarray | list | pd.Series,
    n_bins: int = N_BINS,
    pvalue_max: float = PVALUE_MAX,
) -> tuple:
    """Cut sortable into bins according to Chi-stat.

    Cut `x` into `n_bins` with infomation from `y` bottom-up, in which
    process Chi of adjacent bins will calculated and adjacent bins with
    minimum Chi will be merged together until distributions of `y` in
    different bins differ obviously.

    NOTE:
    1. Adjacent bins with straight zeros for one type  will be merged before
      considering Chi-stat. And no minimum frequency will be applied to bins
      as we just try to merge bins instead of chisquare-test, so the
      robustness of each bin should not be considered.
    2. Cross table should be 2-D NDA, each elements of which represents the
      frequency of the samples with responsible X and Y in cross table.
      For example:
        X   1    2    3
      Y
      1     5    5    1
      2     2    3    5
      Cross table above indicates frequencies of sample(X=1, Y=2) is 2.

    Params:
    -----------------
    x: 1-D NDA
      Array to be cut with sortable elements.
    y: 1-D NDA
      Array filled with "categorical".
    n_bins: int
      Number of bins determining when to stop cut.

    freq_min: int
      Minimum frequency of each bin.
      ----- Deprecated. -------

    pvalue_max: float
      P-value for restrict Chi
      NOTE: All Chi-stat share the same DOF(degree of freedom), so Chi's
      minimum and p-value's maximum are gotten at the same position. And
      p-value will be used to determine when to stop merge bins instead
      of Chi-stats for p-value's being more intuitive.

    Return:
    -----------------
    bins_edges: List of bin edges for cutting.
    ctab: Cross table of final bins.
    """
    assert(x.ndim == 1 and y.ndim == 1)
    (unis_x, unis_y), ctab = contingency.crosstab(x, y)
    bin_edges = np.array([*unis_x, unis_x[-1]])

    # Merge adjacent zeros in cross table or expected frequencies can't be
    # calculated.
    idx = 0
    while idx < ctab.shape[0] - 1:
        if not np.all(np.any(ctab[idx:idx + 2], axis=0)):
            ctab, bin_edges = _merge_bins(ctab, bin_edges, idx)
        else:
            idx += 1

    # # Loop until all the frequences of bins are larger than `freq_min`.
    # while ctab.shape[0] > 1:
    #     # Caculate freqs for each `x`.
    #     ctab_freqs = ctab.sum(axis=1) / x.shape[0]
    #     freq_mindx = np.argmin(ctab_freqs)
    #     if ctab_freqs[freq_mindx] >= freq_min:
    #         break

    #     # Compare chi with the chis of left and right interval to determine
    #     # to merge to left or merge to right.
    #     if freq_mindx == 0:
    #         merge_idx = 1
    #     elif freq_mindx == ctab.shape[0] - 1:
    #         merge_idx = ctab.shape[0] - 1
    #     else:
    #         chis_ = chi_pairwisely(ctab)
    #         merge_idx = (freq_mindx + 1
    #                      if chis_[freq_mindx] > chis_[freq_mindx - 1]
    #                      else freq_mindx)

    #     # Update `bin_edges`, crosstab and frequencies.
    #     ctab, bin_edges = _merge_bins(ctab, bin_edges, merge_idx)

    # Loop until statisfying both `n_bins` and `pvalue_max`.
    while ctab.shape[0] > 1:
        # Calculate Chi-stat pairwisely.
        chis_ = chi_pairwisely(ctab)
        # Merge utils achieving `n_bins` and `chi_max`.
        # NOTE: All Chi-stat share the same DOF(degree of freedom), so Chi's
        # minimum and p-value's maximum are gotten at the same position. And
        # p-value will be used to determine when to stop merge bins instead
        # of Chi-stats for p-value's being more intuitive.
        chi_mindx = np.argmax(chis_[:, 0])
        merge_idx = chi_mindx + 1
        if not (ctab.shape[0] > n_bins or chis_[chi_mindx, 1] > pvalue_max):
            break

        # Update `bin_edges` and crosstab.
        ctab, bin_edges = _merge_bins(ctab, bin_edges, merge_idx)

    bin_edges[1:-1] -= .000001
    return bin_edges, ctab


def _merge_bins(
    ctab: np.ndarray,
    bin_edges: np.ndarray,
    merge_idx: int
) -> tuple:
    """Merge adjacent bins in crosstab.

    Merge given bins according to `merge_idx`
    Merge bins[merge_idx] and bins[merge_idx+1].  in cross table.
    1. For bins edges, drop the bins directly.
    2. Cross table should be 2-D NDA, each elements of which represents the
      frequency of the samples with responsible X and Y in cross table.
      For example:
        X   1    2    3
      Y
      1     5    5    1
      2     2    3    5
      Cross table above indicates frequencies of sample(X=1, Y=2) is 2.

    Params:
    -----------------
    ctab: 2-D NDA
      Cross table with frequencies for each condition.
    bin_edges: 1-D NDA
      List of bin edges.
    merge_idx: int
      Merge bins[merge_idx] and bins[merge_idx+1].

    Return:
    -----------------
    ctab: Cross table after bin merged.
    bin_edges: List of bin edges after bin merged.
    """
    bin_edges = np.concatenate(
        (bin_edges[:merge_idx + 1], bin_edges[merge_idx + 2:]), axis=0)
    ctab = np.concatenate((
        ctab[:merge_idx],
        ctab[merge_idx:merge_idx + 2].sum(axis=0, keepdims=True),
        ctab[merge_idx + 2:]), axis=0)
    return ctab, bin_edges
