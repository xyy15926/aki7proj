#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: display.py
#   Author: xyy15926
#   Created: 2024-11-04 16:41:52
#   Updated: 2024-11-08 16:27:43
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING
import logging

import os
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Graph

from suitbear.finer import tmp_file
from suitbear.kgraph.kgenum import NodeType

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def save_as_html(
    rel_df: pd.DataFrame,
    node_df: pd.DataFrame = None,
    fname: str = "kgraph/graph.html"
) -> Graph:
    if node_df is not None:
        # Display only the nodes in the relations.
        node_df = node_df[node_df["nid"].isin(rel_df["source"])
                          | node_df["nid"].isin(rel_df["target"])]
        # Set node categories.
        categories = [{"name": nt.name} for nt in NodeType]
        cat_map = {nt.value: idx for idx, nt in enumerate(NodeType)}
        # Echcarts graph html with duplicated nodes won't render correctly.
        node_df = (pd.DataFrame({"name": node_df["nid"],
                                 "category": node_df["ntype"].map(cat_map)})
                   .drop_duplicates(subset="name"))
    else:
        node_df = (pd.concat([rel_df["source"], rel_df["target"]])
                   .drop_duplicates()
                   .to_frame("name"))
        node_df["category"] = 0
        categories = [{"name": "undefined"}]

    # Prepare nodes and links.
    nodes = node_df.to_dict("records")

    links = (rel_df[["source", "target"]]
             .to_dict("records"))

    g = (
        Graph(
            init_opts=opts.InitOpts(
                width="1400px",
                height="720px",
                animation_opts=opts.AnimationOpts(
                    animation=False,
                    animation_duration=1000,
                ),
            )
        ).add(
            "",
            nodes,
            links,
            categories=categories,
            is_focusnode=True,
            is_roam=True,
            is_draggable=False,
            is_rotate_label=False,
            layout="force",
            symbol="triangle",
            symbol_size=[10, 10],
            gravity=0.2,
            friction=0.6,
            is_layout_animation=True,
            repulsion=50,
            edge_symbol=[None, "arrow"],
        ).set_global_opts(
            legend_opts=opts.LegendOpts(
                is_show=True
            ),
            title_opts=opts.TitleOpts(
                title=None,
            )
        )
    )
    ghtml = g.render(tmp_file(fname, incr=0).with_suffix(".html"))
    logger.info(f"Graph saved at {ghtml}.")

    return g
