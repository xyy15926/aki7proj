#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: graphagg.py
#   Author: xyy15926
#   Created: 2024-10-15 09:13:32
#   Updated: 2024-11-01 10:54:11
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar
import logging
from itertools import product

from suitbear.crosconf import cross_graph_aggs_and_filters
from suitbear.kgraph.kgenum import NodeType
from suitbear.kgraph import afrels, pbocrels
from suitbear.kgraph.gxgine import GRAPH_REL, GRAPH_NODE


# %%
def last_mois(field: str, desc: str = ""):
    mois = [1, 2, 3, 6, 9, 12, 18, 24, 36, 48]
    reprs = [(f"last_{moi}m",
              f"({field} >= -{moi}) & ({field} <= 0)",
              f"近{moi}月{desc}") for moi in mois]
    reprs += [(None, None, None)]
    return reprs


def last_dois(field: str = "day_itvl(update, today)", desc: str = ""):
    dois = [1, 2, 3, 5, 10, 15, 20, 30, 60, 90, 180, 270, 365, 730, 1460]
    reprs = [(f"last_{doi}d",
              f"({field} >= -{doi}) & ({field} <= 0)",
              f"近{doi}日{desc}") for doi in dois]
    reprs += [(None, None, None),]
    return reprs


def none_of_all():
    reprs = [(None, None, None), ]
    return reprs


# %%
# TODO
GREL_CERTNO_DP1 = {
    "part": "certno_links_dp1",
    "desc": "身份证1度关联",
    "from_": GRAPH_REL,
    "prikey": "nid",
    "depth": 1,
    "ntype": [NodeType.CERTNO, ],
    "key_fmt": "{cond}_{agg}",
    "cond": {
        "rel_type_src": (afrels.node_role_reprs(NodeType.CERTNO, "source")
                         + pbocrels.node_role_reprs(NodeType.CERTNO, "source")),
        "rel_type_tgt": (afrels.node_role_reprs(NodeType.CERTNO, "target")
                         + pbocrels.node_role_reprs(NodeType.CERTNO, "target")),
        "dois": last_dois("day_itvl(update, today)"),
        "None": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        {
            "aggs": ["cnt",],
            "filters": [
                {
                    "source": (["rel_type_src", "dois"], ["None"]),
                }
            ]
        },{
            "aggs": ["cnt",],
            "filters": [
                {
                    "target": (["rel_type_tgt", "dois"], ["None"]),
                }
            ]
        },
    ]
}


# %%
GREL_TEL_DP1 = {
    "part": "tel_links_dp1",
    "desc": "手机号1度关联",
    "from_": GRAPH_REL,
    "prikey": "nid",
    "depth": 1,
    "ntype": [NodeType.TEL, ],
    "key_fmt": "{cond}_{agg}",
    "cond": {
        "rel_type_src": (afrels.node_role_reprs(NodeType.TEL, "source")
                         + pbocrels.node_role_reprs(NodeType.TEL, "source")),
        "rel_type_tgt": (afrels.node_role_reprs(NodeType.TEL, "target")
                         + pbocrels.node_role_reprs(NodeType.TEL, "target")),
        "dois": last_dois("day_itvl(update, today)"),
        "None": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        {
            "aggs": ["cnt",],
            "filters": [
                {
                    "source": (["rel_type_src", "dois"], ["None"]),
                }
            ]
        },{
            "aggs": ["cnt",],
            "filters": [
                {
                    "target": (["rel_type_tgt", "dois"], ["None"]),
                }
            ]
        },
    ]
}


# %%
ORGNO_LINK_DP1 = {
    "part": "orgno_links_dp1",
    "desc": "统一代码1度关联",
    "from_": ["GRAPH_REL"],
    "prikey": ["nid"],
    "depth": 1,
    "ntype": [NodeType.ORGNO],
    "key_fmt": "{cond}_{agg}",
    "cond": {
        "rel_type": (afrels.node_role_reprs(NodeType.ORGNO, "both")),
        "dois": last_dois("day_itvl(update, today)"),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        {
            "agg": ["cnt",],
            "filter": [
                {
                    "both": (["rel_type", "dois"], ["None"]),
                }
            ]
        },
    ]
}


# %%
GRAPH_LINKS_CONF = {
    GREL_CERTNO_DP1["part"]     : GREL_CERTNO_DP1,
    GREL_TEL_DP1["part"]        : GREL_TEL_DP1,
}


# %%
def df_graph_agg_confs(confs: dict = None):
    import pandas as pd
    if confs is None:
        confs = {**GRAPH_LINKS_CONF}

    pconfs = []
    aconfs = {}
    for pname, pconf in confs.items():
        pname = pconf["part"]
        pconfs.append((pconf["part"],
                       pconf["depth"],
                       pconf["prikey"],
                       pconf["from_"],
                       pconf.get("ntype")))
        aconf = cross_graph_aggs_and_filters(
            pconf["cros"], pconf["agg"], pconf["cond"], pconf["key_fmt"])
        aconfs[pname] = pd.DataFrame.from_records(
            aconf, columns=["key", "cond", "agg", "cmt"])

    # Concat the confs.
    pconfs = pd.DataFrame.from_records(
        pconfs, columns=["part", "depth", "prikey", "from_", "ntype"])
    aconfs = pd.concat(aconfs.values(), keys=aconfs.keys()).droplevel(level=1)
    aconfs.index.set_names("part", inplace=True)
    aconfs = aconfs.reset_index()

    return pconfs, aconfs
