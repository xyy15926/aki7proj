#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: networkx.py
#   Author: xyy15926
#   Created: 2024-07-29 10:08:59
#   Updated: 2024-07-29 17:13:05
#   Description:
# ---------------------------------------------------------

# %%
import logging
from typing import List, Tuple

import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from suitbear.finer import get_tmp_path

logging.basicConfig(
    # format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    format="%(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# ref: <https://www.osgeo.cn/networkx/tutorial.htm>
# ref: <https://pyvis.readthedocs.io/en/latest/tutorial.html>
def graph_manipulation():
    # Manipulate Undirected Graph.
    G = nx.Graph([("s", "e", {"etype": "init"}), ])
    G.add_node(1, etype="node")
    G.add_nodes_from([
        (2, {"etype": "list"}),
        (3, {"etype" : "list"}),
    ])
    G.add_edge(4, 5, etype="edge")
    G.add_edges_from([
        (4, 5),
        (4, 2),
    ], etype="list")
    G.add_weighted_edges_from([
        (3, 1, 0.5),
    ])
    G.add_edge(5, 2, weight=0.7)

    # Check the nodes, edges and degrees view of Graph.
    assert len(G.nodes) == 7
    assert len(G.edges) == 5
    assert len(G.adj) == 7
    assert list(G.edges([1, 2, 4])) == [(1, 3), (2, 4), (2, 5), (4, 5)]
    assert list(dict(G.degree([1, 2, 4])).values()) == [1, 2, 2]

    # Remove nodes or edges in Graph.
    G.remove_node(1)
    assert len(G.nodes) == 6
    G.remove_edges_from([(2, 4), ])
    assert len(G.edges) == 3

    # Check and manipulate edges.
    assert G[4][5]["etype"] == "list"
    G[4][5]["color"] = "blue"
    assert G.edges[4, 5]["color"] == "blue"

    # Factorize nodes in G with labels of int, including nodes of named with
    # integers.
    G_int_label = nx.convert_node_labels_to_integers(G, label_attribute=True)
    assert "s" not in G_int_label.nodes
    assert "s" in G.nodes

    # Init Graph from exsiting graph.
    DiG = nx.DiGraph(G, gtype="directed")
    assert len(DiG.edges) == 6
    assert list(DiG.out_edges) == list(DiG.edges)
    assert list(DiG.in_edges) != list(DiG.edges)
    assert DiG.in_edges == DiG.edges
    # `.neighbors` equals to `.sucessors` all the time, namely both of them
    # take only successors into consideration, while `.degree` takes both
    # in-degrees and out-degrees.
    assert DiG.degree[5] == len(list(DiG.neighbors(5))) * 2
    assert DiG.degree[5] == len(list(DiG.successors(5))) * 2


def draw_graph():
    G = nx.petersen_graph()

    options = {
        'node_color': 'black',
        'node_size': 100,
        'width': 3,
    }
    plt.subplot(131)
    # Default spring layout
    nx.draw(G, with_labels=True, **options)
    plt.subplot(132)
    # Shell layout
    nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True)
    ax3 = plt.subplot(133)
    # Spectral layout
    nx.draw_spectral(G, ax=ax3, with_labels=True)
    plt.savefig(get_tmp_path() / "networkx_draw.png")
    plt.close()

    net = Network()
    net.from_nx(G)
    net.show(get_tmp_path() / "pyvis_network_show.html")
    # `display` may not be supported in some front-end.
    display(HTML(get_tmp_path() / "pyvis_network_show.html"))


