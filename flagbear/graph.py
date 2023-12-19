#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: graph.py
#   Author: xyy15926
#   Created: 2023-12-02 21:10:25
#   Updated: 2023-12-14 14:22:18
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from sys import maxsize as MAXINT
from typing import TypeVar, Any
from collections.abc import Mapping, Sequence, Callable
# from IPython.core.debugger import set_trace

Node = TypeVar("node")
Container = TypeVar("container")


# %%
# TODO: Finish forward updatation.
def forward_update_traverse(
    node: Node,
    nodes: list[Node],
    node_ST: list[Node],
    mark_D: dict[Node, int],
    cal_F: Callable[[Node], Container],
    rel_F: Callable[[Node], list[Node]],
    update_F: Callable[[Container, Container], None],
    result_D: dict[Node, Container],
) -> None:
    pass


# %%
def backward_update_digraph(
    nodes: list[Node],
    cal_F: Callable[[Node], Container],
    rel_F: Callable[[Node], list[Node]],
    update_F: Callable[[Container, Container], None],
) -> dict[Node, Container]:
    """Compute stable status for nodes in digraph.

    Update and return status of nodes with traversing the digraph only once.
    1. Status of each node should be and only be determined by the pointing
      nodes **directly**, UNION for example.
    2. Nodes in a circle share one status after traversing.
    3. So, status must be represented with a sharable container, SET for
      example. Or status won't be updated "automatically" for all nodes in a
      circle.

    Notions:
    1. Another function, refered as `key_F`, may be help if the nodes can't be
      used as the key for mapper, namely node can't be hashed.

    Params:
    ---------------------------
    nodes: Node should be hashable so as to be marked if processed during
      traversing.
    cal_F: Get initial status for each node.
    rel_F: Get the pointing nodes.
    update_F: Update status for each node with the status of the nodes pointing
      to the it. Accept status of two node and update the former one with the
      latter one.

    Result:
    ---------------------------
    result_D: Dict of node and final result.
    """
    mark_D = dict.fromkeys(nodes, 0)
    result_D = {}
    node_ST = []

    for node in nodes:
        if mark_D[node] == 0:
            backward_update_traverse(
                node, nodes, node_ST, mark_D, cal_F, rel_F, update_F, result_D
            )
    return result_D


def backward_update_traverse(
    node: Node,
    nodes: list[Node],
    node_ST: list[Node],
    mark_D: dict[Node, int],
    cal_F: Callable[[Node], Container],
    rel_F: Callable[[Node], list[Node]],
    update_F: Callable[[Container, Container], None],
    result_D: dict[Node, Container],
) -> None:
    """Compute stable status for nodes in digraph.

    Update and return status of nodes with traverse the digraph only once.
    1. Status of each node should be and only be determined by the pointing
      nodes directly, UNION for example.
    2. Nodes in a circle share one status after traversing.
    3. So, status must be represented with a sharable container, SET for
      example. Or status won't be updated "automatically" for all nodes in a
      circle.

    Notions:
    1. Another function, refered as `key_F`, may be help if the nodes can't be
      used as the key for mapper, namely node can't be hashed.
    2. Every node in nodes must be processed by this so to get the final status
      of all nodes in digraph, which is be recorded in `mark_D`.

    Example:
    A -> B & C
    B -> C
    status[A] Should be updated by status[B] and status[C], namely "backward".

    Params:
    ---------------------------
    node:  Start Node.
    nodes: Node should be hashable so as to be marked if processed during
      traversing.
    node_ST: Stores the nodes in processing.
    mark_D: Markes if nodes has been precessed.
    cal_F: Get initial status for each node.
    rel_F: Get the pointing nodes.
    update_F: Update status for each node with the status of the nodes pointing
      to the it. Accept status of two node and update the former one with the
      latter one.
    result_D: Stores the result of each node

    Result:
    ----------------------------
    None.
    Final status is stored in `result_D`.
    """
    node_ST.append(node)
    mark_D[node] = node_pos = len(node_ST)
    result_D[node] = cal_F(node)

    # Travese pointing nodes.
    nns = rel_F(node)
    if nns is None:
        return
    for nn in nns:
        # set_trace()
        if mark_D[nn] == 0:
            backward_update_traverse(
                nn, nodes, node_ST, mark_D, cal_F, rel_F, update_F, result_D
            )
        # Set the nodes' index in the `node_ST` with the minimum among all the
        # nodes of a cycle when destacking backwards. So to determine how to
        # pop the nodes out.
        mark_D[node] = min(mark_D[node], mark_D[nn])
        # Update state of `node` with status of `node`.
        update_F(result_D[node], result_D[nn])

    # Reach the leaf node or the start node of the cycle.
    if mark_D[node] == node_pos:
        # set_trace()
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
            result_D[ele] = result_D[node]
