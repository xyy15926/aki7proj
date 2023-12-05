#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: graph.py
#   Author: xyy15926
#   Created: 2023-12-02 21:10:25
#   Updated: 2023-12-03 16:00:13
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from sys import maxsize as MAXINT
from typing import Any


# %%
def backward_update_digraph(
    nodes: list, rel_F: callable, cal_F: callable, update_F: callable,
) -> dict:
    """
    Description:
    Update and return status of nodes with traverse the digraph only once.
    1. Status of each node should be and only be determined by the pointing
    nodes directly, UNION for example.
    2. Status must be represented with a sharable container, SET for example. Or
    status won't be updated "automatically" for all nodes in a circle.

    Notions:
    1. Another function, refered as `key_F`, may be help if the nodes can't be used
    as the key for mapper, A.K.A. node can't be hashed, A.K.A. `__hash__` isn't
    implemented for node.

    Params:
    nodes: list of nodes
    rel_F: callable to get pointing nodes
    cal_F: callable to get initial result for each node
    update_F: callable to update result for each node with its pointing nodes

    Result:
    result-container for each node.
    """
    mark_D = dict.fromkeys(nodes, 0)
    result_D = {}
    node_ST = []

    for node in nodes:
        if mark_D[node] == 0:
            backward_update_traverse(
                node, nodes, node_ST, mark_D, rel_F, cal_F, update_F, result_D
            )
    return result_D


def backward_update_traverse(
    node: Any,
    nodes: list,
    node_ST: list,
    mark_D: dict,
    rel_F: callable,
    cal_F: callable,
    update_F: callable,
    result_D: dict,
) -> None:
    """
    Description:
    Update and return status of nodes with traverse the digraph only once.
    1. Status of each node should be and only be determined by the pointing
    nodes directly, UNION for example.
    2. Status must be represented with a sharable container, SET for example. Or
    status won't be updated "automatically" for all nodes in a circle.

    Notions:
    1. Another function, refered as `key_F`, may be help if the nodes can't be used
    as the key for mapper, A.K.A. node can't be hashed, A.K.A. `__hash__` isn't
    implemented for node.

    Params:
    node: start node
    nodes: list of nodes
    node_ST: stack storing the nodes in processing
    mark_D: dict remark if nodes has been precessed
    rel_F: callable to get pointing nodes
    cal_F: callable to get initial result for each node
    update_F: callable to update result for each node with its pointing nodes
    result_D: dict storing the result of each node

    Result:
    result-container for each node.
    """
    node_ST.append(node)
    mark_D[node] = node_pos = len(node_ST)
    result_D[node] = cal_F(node)

    # Travese pointing nodes.
    for nn in rel_F(node):
        if mark_D[nn] == 0:
            backward_update_traverse(
                nn, nodes, node_ST, mark_D, rel_F, cal_F, update_F, result_D
            )
        # Set the nodes' index in the `node_ST` with the minimum among all the
        # nodes of a cycle when destacking backwards. So to determine how to
        # pop the nodes out.
        mark_D[node] = min(mark_D[node], mark_D[nn])
        update_F(result_D[node], result_D[nn])

    # Reach the leaf node or the start node of the cycle.
    if mark_D[node] == node_pos:
        ele = node_ST.pop()
        mark_D[ele] = MAXINT
        result_D[ele] = result_D[node]
        # Pop all the other nodes of the cycle out.
        while node != ele:
            ele = node_ST.pop()
            mark_D[ele] = MAXINT
            # `result` must store sharable container instead of copiable value.
            # Or the `result` can't be set properly if some node belongs to
            # two or more cycles.
            # TODO: weakref
            result_D[ele] = result_D[node]
