#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_graph.py
#   Author: xyy15926
#   Created: 2023-12-13 15:53:12
#   Updated: 2023-12-14 14:19:23
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.llp import graph
    reload(graph)
from ubears.flagbear.llp.graph import backward_update_traverse, backward_update_digraph


# %%
def gen_digraph():
    nodes = list(range(10))
    links = ((0, (1, )),
             (1, (2, 3)),
             (2, (3, 4)),
             (3, (5, 7, 8)),
             (4, (6, )),
             (6, (1, 2)),
             (7, (1, 8)),
             (8, (9, )))
    return nodes, links


# %%
def test_backward_update_traverse():
    nodes, links = gen_digraph()
    links = {i: j for i,j in links}
    initials = [{i} for i in nodes]
    node_ST = []
    mark_D = dict.fromkeys(nodes, 0)
    result_D = {}
    # Error
    backward_update_traverse(0, nodes, node_ST, mark_D,
                             lambda x: initials[x],
                             lambda x: links.get(x, []),
                             lambda x, y: x.update(y),
                             result_D)
    assert result_D[0] == set(range(10))
    assert result_D[1] == set(range(1, 10))
    assert result_D[5] == {5}
    assert result_D[8] == {8, 9}
    assert result_D[9] == {9}
    assert result_D[1] is result_D[2]
    assert result_D[2] is result_D[3]
    assert result_D[3] is result_D[4]
    assert result_D[4] is result_D[6]
    assert result_D[6] is result_D[7]


def test_backward_update_traverse_max():
    nodes, links = gen_digraph()
    links = {i: j for i,j in links}
    initials = [[i] for i in nodes]
    node_ST = []
    mark_D = dict.fromkeys(nodes, 0)
    result_D = {}

    def update_F(x, y):
        x[0] = max(x[0], y[0])

    backward_update_traverse(0, nodes, node_ST, mark_D,
                             lambda x: initials[x],
                             lambda x: links.get(x, []),
                             update_F,
                             result_D)

    assert result_D[0] == [9]
    assert result_D[1] == [9]
    assert result_D[5] == [5]
    assert result_D[8] == [9]
    assert result_D[9] == [9]
    assert result_D[1] is result_D[2]
    assert result_D[2] is result_D[3]
    assert result_D[3] is result_D[4]
    assert result_D[4] is result_D[6]
    assert result_D[6] is result_D[7]
    assert result_D[0] is not result_D[1]
    assert result_D[0] is not result_D[8]
    assert result_D[0] is not result_D[9]


def test_backward_update_digraph():
    nodes, links = gen_digraph()
    links = {i: j for i,j in links}
    initials = [{i} for i in nodes]
    result_D = backward_update_digraph(nodes,
                                       lambda x: initials[x],
                                       lambda x: links.get(x, []),
                                       lambda x, y: x.update(y))
    assert result_D[0] == set(range(10))
    assert result_D[1] == set(range(1, 10))
    assert result_D[5] == {5}
    assert result_D[8] == {8, 9}
    assert result_D[9] == {9}
    assert result_D[1] is result_D[2]
    assert result_D[2] is result_D[3]
    assert result_D[3] is result_D[4]
    assert result_D[4] is result_D[6]
    assert result_D[6] is result_D[7]




