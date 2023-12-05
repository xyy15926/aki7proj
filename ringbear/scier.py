#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: sciproc.py
#   Author: xyy15926
#   Created: 2023-01-04 12:36:20
#   Updated: 2023-04-28 17:52:30
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import os
import sys
from collections import deque, defaultdict
from collections.abc import Sequence
from functools import partial
import numpy as np
import pandas as pd
import logging
from scipy.stats import contingency
from sklearn.tree import (DecisionTreeClassifier, )
from sklearn.ensemble import (RandomForestClassifier, )
from sklearn.preprocessing import (FunctionTransformer, )
import ringbear.unifier
import ringbear.executor
import ringbear.npam
# from importlib import reload
# reload(ringbear.unifier)
# reload(ringbear.executor)
# reload(ringbear.npam)
from ringbear.unifier import (unify_shape22, )
from ringbear.executor import (aggregate_with_key, )
from ringbear.npam import (calculate_criterion, )


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


# %% ------------------------------------------------------------------------
#                   * * * * Transformer Modification * * * *
#
#   Some transformers will be inherited and modified here to fit in the whole
#   process.
# ---------------------------------------------------------------------------
class OneToOneFunctionTransformer(FunctionTransformer):
    """
    Description:
    This class inherits from FunctionTrasnformer with only attributes
    `faeture_names_out` and `n_features_in` add, which endues the class
    the feature `get_feature_names_out`.

    Attention:
    This implement is based on skikit-learn 1.1.2, this may need to be changed
    with the releases.
    """
    def fit(self, X, y=None):
        self.feature_names_out = "one-to-one"
        self.n_features_in_ = X.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
        return super().fit(X, y)


# %% ------------------------------------------------------------------------
#                   * * * * DecisionTree Utils * * * *
#
#   Utils for decision tree in sklearn.
# ---------------------------------------------------------------------------
def build_parent_from_children(*children: np.ndarray) -> np.ndarray:
    """
    Description:
    Construct the array of parent indices from the arrays of children node
    indices:
    child_array_N[i]: indicate the index of chlid N of node i

    Params:
    children: arrays of children node indices.

    Return:
    ndarray
    """
    length = children[0].shape[0]
    parent = np.zeros(length, dtype=np.int8)
    parent[0] = -1
    # Traverse all the arrays of children node indices, for each array:
    # 1. `child[child>0]`: fetch valid children node indices.
    # 2. `np.arange[child>0]`: fetch coresponding parent node indices for
    #   each child node.
    for child in children:
        parent[child[child > 0]] = np.arange(length, dtype=np.int8)[child > 0]
    return parent


def extract_paths_from_tree(
    tree: DecisionTreeClassifier,
    node_indices: list | tuple | np.ndarary | None = None,
) -> tuple:
    """
    Description:
    Extract paths from root to node determined in `node_indices`, along with
    recording capacity, criterion-values and path-choice.

    Params:
    tree:
    node_indices:
    capcacities:
    criterion_vals:

    Return:
    [(node_index, index_val, node_capacity, path), ...]
    index_val: array with shape (NODE_N_IN_PATH, INDEX_LENGTH)
    node_capcity: array with shape (NODE_N_IN_PATH, KIND_N)
    path: deque[(feature_indices, left(<=False) | right(>True), threshold),...]
    """
    # Construct parent indices.
    parent = build_parent_from_children(tree.tree_.children_left,
                                        tree.tree_.children_right)
    node_indices = range(tree.tree_.node_count) if node_indices is None \
        else node_indices
    valid_paths = []
    # Travese all node to check if satisfying the conditions.
    for node_index in node_indices:
        cur_index, path_steps = node_index, deque()
        # Fetch steps in path from leaves to root with `parent` storing
        # indices of parent.
        while cur_index > 0:
            par_index = parent[cur_index]
            path_steps.appendleft((
                tree.tree_.feature[par_index],
                tree.tree_.children_right[par_index] == cur_index,
                tree.tree_.threshold[par_index]))
            cur_index = par_index
        valid_paths.append(path_steps)

    return valid_paths


# %%
def calculate_criterion_from_tree(
    tree: DecisionTreeClassifier,
    criterion_type: str = "freq",
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    weights: np.ndarray | None = None
) -> tuple:
    """
    Description:
    Calculate criterion of each node in tree.
    Return `tree_.impurity`, or cast it to given criterion, directly if
    no `X` is provided. Else, capacities of each node will be caculated with
    `decision_path`.
    Note that even if `X` is the whole trained-set, capacities stored in
    `tree.tree_.value` may differ with `decision_path(X)` in bootstrapping
    cases.

    Params:
    tree:
    criterion_type: only frequency, entropy and gini-index supported
    X:
    y:
    weights:

    Return:
    criteria, capacities
    """
    # Call `decision_path` to calculate the capacity of each node.
    if X is not None:
        X, y, weights = unify_shape22(X, y, weights)
        # Call `decsion_path` to get the decision paths of `X`, with shape
        # (SAMPLE_N, TREE_NODE_N)
        paths = tree.decision_path(X).toarray()
        arr = np.concatenate([paths, y, weights], axis=1)
        # 1. Sort and split array acorrding to `y`.
        # 2. `sum` `weights * arr[i]` to compute the capacities in each
        #   section.
        arr = arr[arr[:, -2].argsort()]
        uniques, indices = np.unique(arr[:, -2], return_index=True)
        capacities = np.array([np.sum(subarr[:, :-2] * subarr[:, -1:], axis=0)
                               for subarr in np.split(arr, indices[1:])]).T
    # Fetch `tree_.value` as capacities directly.
    else:
        capacities = tree.tree_.value[:, 0, :]

    return calculate_criterion(capacities, criterion_type), capacities


# %%
def tree_cut_ordered(
    x: np.ndarray | list | pd.Series,
    y: np.ndarray | list | pd.Series,
    n_bins: int = N_BINS,
    freq_min: float = FREQ_MIN,
    min_impurity_decrease: float = IMPURITY_DECREASE_MIN,
) -> tuple:
    """
    Description:
    Cut `x` into `n_bins` by DescisionTreeClassifier with gini as criterion.

    Params:
    x:
    y:
    n_bins:
    freq_min:
    min_impurity_decrease: param passed to init tree to restrict the split
        behavior.

    Return:
    bins_edges, crosstab of final bins
    """
    x, y = unify_shape22(x, y)
    # Fit tree classifer.
    tree = DecisionTreeClassifier(
        min_samples_leaf=freq_min,
        max_leaf_nodes=n_bins,
        min_impurity_decrease=min_impurity_decrease,
    ).fit(x, y)

    # Select leaf node by compare its children node with -1.
    leaf_node_map = (tree.tree_.children_left == -1) & \
        (tree.tree_.children_right == -1)

    # Get thresholds directly.
    threshs = tree.tree_.threshold[~leaf_node_map]
    threshs = np.array([np.min(x), *threshs, np.max(x)])

    ctab = tree.tree_.value[:, 0, :]
    return threshs, ctab[leaf_node_map]


# %%
def chi_pairwise(
    ctab: np.ndarray
) -> np.ndarray:
    """
    Decription:
    Caculate chi for adjecent bins.

    Params:
    ctab: cross table

    Return:
    [[chi, p-value], ]
    """
    # `chis_` stores `[[chi_, pvalue], ...]`
    chis_ = np.apply_along_axis(
        lambda x: contingency.chi2_contingency(x.reshape(2, -1))[0:2],
        axis=1,
        arr=np.concatenate((ctab[:-1, :], ctab[1:, :]), axis=1)
    )
    return chis_


def chi_pairwise_itl(
    ctab: np.ndarray
) -> np.ndarray:
    """
    Decription:
    Caculate chi for adjecent bins.

    Params:
    ctab: cross table

    Return:
    [[chi, p-value], ]
    """
    chis_ = np.array(list(
        map(lambda x: contingency.chi2_contingency(x)[0:2],
            zip(ctab[:-1, :], ctab[1:, :]))))
    return chis_


def _merge_bins(
    ctab: np.ndarray,
    bin_edges: np.ndarray,
    merge_idx: int
) -> tuple:
    """
    Description:
    Merge cross table and bin edges according to `merge_idx`.

    Params:
    ctab: cross table with frequencies for each condition
    bin_edges: bin edges
    merge_idx: bin edge to be dropped

    Return:
    cross table, bin edges
    """
    bin_edges = np.concatenate(
        (bin_edges[:merge_idx], bin_edges[merge_idx+1:]), axis=0)
    ctab = np.concatenate((
        ctab[:merge_idx-1 if merge_idx > 0 else 0],
        ctab[merge_idx-1:merge_idx] + ctab[merge_idx:merge_idx+1],
        ctab[merge_idx+1:]), axis=0)
    return ctab, bin_edges


def chimerge_cut_ordered(
    x: np.ndarray | list | pd.Series,
    y: np.ndarray | list | pd.Series,
    n_bins: int = N_BINS,
    freq_min: float = FREQ_MIN,
    pvalue_max: float = PVALUE_MAX,
) -> tuple:
    """
    Decription:
    Cut `x` into `n_bins` with infomation from `y` bottom-up, in which process
    chi of adjacent bins will calculated and adjacent bins with minimum chi
    will be merged together until distributions of `y` in different bins differ
    obviously.
    Note that bins with frequency smaller than `freq_min` will be merged early.

    Params:
    x:
    y:
    n_bins:
    freq_min:
    pvalue_max: p-value for restrict chi
        Note that all chi_ share the same dof, so chi_'s minimum and pvalue's
        maximum are at the same position. And `pvalue` will be used to
        determine when to stop merge bins instead of chi-stats for its
        intuition.

    Return:
    bins_edges, crosstab of final bins
    """
    assert x.ndim == 1 and y.ndim == 1
    (unis_x, unis_y), ctab = contingency.crosstab(x, y)
    bin_edges = np.array([*unis_x, unis_x[-1]])

    # Merge adjacent zeros in cross table or expected frequencies can't be
    # calculated.
    # Get all adject zeros in cross table.
    adj_zeros = np.any(
        (ctab == 0) & (np.subtract.accumulate(ctab, axis=0) == 0),
        axis=1)[1:].nonzero()[0] + 1
    # Move on after merge bins.
    adj_zeros = adj_zeros - np.arange(adj_zeros.shape[0])
    for i in adj_zeros:
        ctab, bin_edges = _merge_bins(ctab, bin_edges, i)

    # Loop until all the frequences of bins are larger than `freq_min`.
    while ctab.shape[0] > 1:
        # Caculate freqs for each `x`.
        ctab_freqs = ctab.sum(axis=1) / x.shape[0]
        freq_mindx = np.argmin(ctab_freqs)
        if ctab_freqs[freq_mindx] >= freq_min:
            break

        # Compare chi with the chis of left and right interval to determine
        # to merge to left or merge to right.
        if freq_mindx == 0:
            merge_idx = 1
        elif freq_mindx == ctab.shape[0] - 1:
            merge_idx = ctab.shape[0] - 1
        else:
            chis_ = chi_pairwise(ctab)
            merge_idx = freq_mindx + 1 \
                if chis_[freq_mindx, 1] > chis_[freq_mindx-1, 1] \
                else freq_mindx

        # Update `bin_edges`, crosstab and frequencies.
        ctab, bin_edges = _merge_bins(ctab, bin_edges, merge_idx)

    # Loop until statisfying both `n_bins` and `pvalue_max`.
    while ctab.shape[0] > 1:
        # Calculate chi_ pairwisely.
        chis_ = chi_pairwise(ctab)
        # Merge utils achieving `n_bins` and `chi_max`.
        # Note that all chi_ share the same dof, so chi_'s minimum and
        # pvalue's maximum are at the same position. And `pvalue` will
        # be used to determine when to stop merge bins instead of
        # chi-stats for its intuition.
        chi_mindx = np.argmax(chis_[:, 1])
        merge_idx = chi_mindx + 1
        if not (ctab.shape[0] > n_bins or chis_[chi_mindx, 1] > pvalue_max):
            break

        # Update `bin_edges` and crosstab.
        ctab, bin_edges = _merge_bins(ctab, bin_edges, merge_idx)

    return bin_edges, ctab
