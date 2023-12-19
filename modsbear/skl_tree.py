#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: trees.py
#   Author: xyy15926
#   Created: 2023-12-06 09:18:58
#   Updated: 2023-12-11 11:52:45
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import logging
from collections import deque

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ringbear.freqs import (enhanced_freqs,
                            cal_entropy, cal_gini)

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def biclf_select_nodes(
    tree: DecisionTreeClassifier,
    Xs: list[np.ndarray] | None = None,
    ys: list[np.ndarray] | None = None,
    lift_thresh: float = 2
) -> list:
    """Select nodes in DTC according to their capacities.

    Select nodes according to lifts of y=1 samples in DecisionTreeClassifier.
    1. Other (X, y) pairs could be passed to DTC instead of only default
      training data used to calculate lifts for robustness.

    Params:
    -------------
    tree: DescisionTreeClassifier
      Criterion of the nodes in the tree will be calculcated.
    Xs: List of 2-D NDA.
    ys: List of 1-D NDA.
    lift_thresh: float
      Lift threshold for tree nodes to be selected.

    Return:
    1-D NDA storing node indices.
    """
    # Take tree's original capacities and frequencies into consideration.
    Xs = [None,] if Xs is None else [None, *Xs]
    ys = [None,] if ys is None else [None, *ys]

    node_map = np.ones(tree.tree_.node_count, dtype=np.bool_)
    # Traverse to get all capacities and frequencies of all the nodes.
    rfreqs, freqs = [], []
    for X, y in zip(Xs, ys):
        rfreqs, freqs = tree_node_metric(tree, X, y, "freq")
        lift_41 = rfreqs[:, -1] / rfreqs[0, -1]
        node_map &= (lift_41 > lift_thresh)

    return np.arange(tree.tree_.node_count)[node_map]


# %%
def extract_paths_from_tree(
    tree: DecisionTreeClassifier,
    node_indices: list | tuple | np.ndarary | None = None,
) -> tuple:
    """Extract nodes path from tree.

    Extract node paths from root to the nodes in `node_indices` from tree.
    1. Each node in path should contain the feature, threshold and splitting
      direction.
    2. Refering <https://scikit-learn.org/1.2/auto_examples/tree/plot_unveil_tree_structure.html>
      for `tree.tree_` attributes description in detail.

    Params:
    -------------
    tree: DescisionTreeClassifier
      Criterion of the nodes in the tree will be calculcated.
    node_indices: Sequence
      Storing the nodes indicating the path to be extracted from tree.

    Return:
    -------------
    node_path: deque([(node, feature, left-or-right, threshold), ...])
      node: Integer representing the node.
      feature: Feature used for splitting the node.
      threshold: Threshold when splitting the node.
      left-or-right: Bool indicating the next is left or right child
        True: Right child, > `threshold`
        False: Left child, <= `threshold`
    """
    tree_ = tree.tree_
    # Construct parent indices.
    parent = build_parent_from_children(tree_.children_left,
                                        tree_.children_right)
    node_indices = (range(tree_.node_count) if node_indices is None
                    else node_indices)
    valid_paths = []
    for node_index in node_indices:
        cur_index = parent[node_index]
        path_steps = deque()
        while cur_index >= 0:
            path_steps.appendleft((
                cur_index,
                tree_.feature[cur_index],
                tree_.threshold[cur_index],
                tree_.children_right[cur_index] == cur_index,))
            cur_index = parent[cur_index]
        valid_paths.append(path_steps)

    return valid_paths


def build_parent_from_children(*children: np.ndarray) -> np.ndarray:
    """Build parent pointers with children pointers.

    Construct the array of parent pointer with array of children pointers
    of which each element represent the child node with the index in the
    array.

    Params:
    --------------
    children: np.ndarray
      >= 0: Chlid node index.
      -1: Null child node.

    Return:
    --------------
    parent: np.ndarray of parent pointer.
    """
    length = children[0].shape[0]
    parent = np.zeros(length, dtype=np.int8)
    parent[0] = -1
    # Traverse all the arrays of children node indices, for each array:
    # 1. `child[child>0]`: Fetch valid children node indices.
    # 2. `np.arange[child>0]`: Fetch coresponding parent node indices for
    #   each child node.
    for child in children:
        parent[child[child > 0]] = np.arange(length, dtype=np.int8)[child > 0]
    return parent


# %%
def tree_node_metric(
    tree: DecisionTreeClassifier,
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
    metric: str = "freq",
    weights: np.ndarray | None = None
) -> tuple:
    """Calculate given criterion for each node in the tree.

    Calculate criterion of each node in tree.
    1. `Tree._tree.impurity` stores the criterion value in training process.
      But the capacities of each node instead of `Tree._tree.impurity` will
      be used at any time for the following reasons:
      1. The criterion used for training may not be the criterion wanted.
      2. The values are calculated with training data, which may be
        overscoring.
      3. In bootstrapping models such RandomForest, training data are chosen
        randomly from the whole, which can't be repeated anymore.
    2. `Tree.decision_path()` will be called to get the prediction paths of
      all samples.

    Return `tree_.impurity`, or cast it to given criterion, directly if
    no `X` is provided. Else, capacities of each node will be caculated with
    `decision_path`.
    Note that even if `X` is the whole trained-set, capacities stored in
    `tree.tree_.value` may differ with `decision_path(X)` in bootstrapping
    cases.

    Params:
    --------------
    tree: DescisionTreeClassifier
      Criterion of the nodes in the tree will be calculcated.
    metric: str
      Only frequency, entropy and gini-index are supported currently.
      freqs: frequency
      entropy: `cal_entropy`
      gini: GINI index by `cal_gini`
    X: None | 2-D NDA of the shape at tree training stage
      NDA: Data array passed to DecisionTreeClassifier to get the predicted
        labels. So its shape must be the same as the input when tree fitting.
      None: Used the capacities set at the traning stage storing in
        `tree.tree_.value`.
    y: None | 1-D NDA
      NDA: Actual label array, which will be used to calculated the criterion
        with the predicted labels.
      None: Like the above.
    weights: 1-D NDA
      Weights of samples.

    Return:
    --------------
    cri: 1-N for entropy for GINI or 2-D NDA for relative frequencies.
    freqs: 2-D NDA of shape (TREE_NODE_N, SORTED_LABEL_N)
      Storing frequencies of labels in nodes.
    """
    # Call `decision_path` to calculate the capacity of each node.
    if X is not None:
        # `paths` if 2-D NDA of shape (SAMPLE_N, TREE_NODE_N) filled with 0, 1.
        #   Its elements represent whether the sample in the responsible node.
        paths = tree.decision_path(X).toarray()
        if weights is not None:
            paths *= weights.reshape(-1, 1)
        freqs = enhanced_freqs(y, others=paths, agg=lambda x: x.sum(axis=0))[0].T
    # Fetch `tree.tree_.value` as freqs directly.
    else:
        freqs = tree.tree_.value[:, 0, :]

    if metric == "freq":
        cri = freqs / freqs.sum(axis=1, keepdims=True)
    elif metric == "entropy":
        cri = cal_entropy(freqs)
    elif metric == "gini":
        cri = cal_gini(freqs)
    else:
        raise ValueError(f"Invalid arguments {metric} for `metric`.")
    return cri, freqs
