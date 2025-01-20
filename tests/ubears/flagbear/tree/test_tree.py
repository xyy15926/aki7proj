#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_tree.py
#   Author: xyy15926
#   Created: 2023-12-15 16:13:36
#   Updated: 2024-11-11 14:12:32
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.tree import tree
    reload(tree)

from ubears.flagbear.tree.tree import BiTNode, GeTNode


# %%
def test_GenericTree():
    nodes = [GeTNode(i) for i in range(10)]
    nodes[0].chs = nodes[1:4]
    nodes[1].chs = nodes[4:5]
    nodes[2].chs = nodes[5:10]

    st = nodes[0].post_order()
    vals = [node.val for node in st]
    assert vals == [4, 1, 5, 6, 7, 8, 9, 2, 3, 0]

    st = []
    nodes[0].post_order(st.append)
    vals = [node.val for node in st]
    assert vals == [4, 1, 5, 6, 7, 8, 9, 2, 3, 0]
