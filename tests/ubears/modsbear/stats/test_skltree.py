#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_skltree.py
#   Author: xyy15926
#   Created: 2023-12-08 17:36:14
#   Updated: 2024-05-26 19:11:16
#   Description:
# ---------------------------------------------------------

# %%
from pytest import mark

if __name__ == "__main__":
    from importlib import reload
    from ubears.modsbear.stats import skltree
    from ubears.ringbear import freqs
    reload(skltree)
    reload(freqs)

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ubears.modsbear.stats.skltree import (
    tree_node_metric,
    extract_paths_from_tree,
    build_parent_from_children,
    biclf_select_nodes
)


# %%
def make_data(length: int = 20, *,
              add_nan: bool = False,
              seed: int = 7777) -> tuple:
    np.random.seed(seed)
    X = np.column_stack([
        np.random.randint(1, 5, length),
        np.random.randint(1, 100, length),
        np.random.randint(1, 100, length),
        np.random.randint(1, 100, length),
    ])
    y = np.random.choice([0, 1], length)
    if add_nan:
        X.iloc[np.random.choice(range(length), 2), 0] = np.nan
        X.iloc[np.random.choice(range(length), 2), 5] = np.nan
        X.iloc[np.random.choice(range(length), 1)] = np.nan
    return X, y


# %%
@mark.filterwarnings("ignore: divide by zero")
def test_tree_node_metric():
    X, y = make_data()
    tree = DecisionTreeClassifier().fit(X, y)
    rfreqs_none, freqs_none = tree_node_metric(tree)
    rfreqs_xy, freqs_xy = tree_node_metric(tree, X, y)
    assert np.all(freqs_none == freqs_xy)
    assert np.all(rfreqs_none == rfreqs_xy)
    gini_xy, freqs_xy = tree_node_metric(tree, X, y, "gini")
    ent_xy, freqs_xy = tree_node_metric(tree, X, y, "entropy")
    assert np.all(ent_xy >= gini_xy)
    assert np.all(freqs_none == freqs_xy)


# %%
def test_extract_paths_from_tree():
    X, y = make_data(seed=7777777)
    tree = DecisionTreeClassifier().fit(X, y)
    node_idxs = biclf_select_nodes(tree)
    assert node_idxs.ndim == 1
    paths = extract_paths_from_tree(tree, node_idxs)
    assert len(paths) == node_idxs.size
