#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: graphagg.py
#   Author: xyy15926
#   Created: 2024-10-15 09:13:32
#   Updated: 2024-11-07 17:27:28
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar
import logging
from itertools import product

from ubears.suitbear.dirt.crosconf import cross_graph_aggs_and_filters
from ubears.suitbear.kgraph.kgenum import NodeType
from ubears.suitbear.kgraph import afrels, pbocrels
from ubears.suitbear.kgraph.gxgine import GRAPH_REL, GRAPH_NODE


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
GREL_CERTNO_DP1 = {
    "part": "grel_certno_dp1",
    "desc": "身份证1度关联",
    "from_": GRAPH_REL,
    "prikey": ["nid"],
    "depth": 1,
    "ntype": [NodeType.CERTNO, ],
    "key_fmt": "certno_{cond}_{agg}",
    "cond": {
        "link_type": (afrels.link_type_reprs(NodeType.CERTNO, "both")
                      + pbocrels.link_type_reprs(NodeType.CERTNO, "both")),
        "dois": last_dois("day_itvl(update, today)"),
        "NONE": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        {
            "aggs": ["cnt",],
            "conds": [
                {
                    "both": (["link_type", "dois"], ["NONE"]),
                }
            ]
        }
    ]
}


# %%
GREL_TEL_DP1 = {
    "part": "grel_tel_dp1",
    "desc": "手机号1度关联",
    "from_": GRAPH_REL,
    "prikey": ["nid"],
    "depth": 1,
    "ntype": [NodeType.TEL, ],
    "key_fmt": "tel_{cond}_{agg}",
    "cond": {
        "link_type": (afrels.link_type_reprs(NodeType.TEL, "both")
                      + pbocrels.link_type_reprs(NodeType.TEL, "both")),
        "dois": last_dois("day_itvl(update, today)"),
        "NONE": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        {
            "aggs": ["cnt",],
            "conds": [
                {
                    "both": (["link_type", "dois"], ["NONE"]),
                }
            ]
        }
    ]
}


# %%
GREL_VIN_DP1 = {
    "part": "grel_vin_dp1",
    "desc": "车架号1度关联",
    "from_": ["GRAPH_REL"],
    "prikey": ["nid"],
    "depth": 1,
    "ntype": [NodeType.VIN],
    "key_fmt": "vin_{cond}_{agg}",
    "cond": {
        "link_type": (afrels.link_type_reprs(NodeType.VIN, "both")
                      + pbocrels.link_type_reprs(NodeType.VIN, "both")),
        "dois": last_dois("day_itvl(update, today)"),
        "NONE": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        {
            "aggs": ["cnt",],
            "conds": [
                {
                    "both": (["link_type", "dois"], ["NONE"]),
                }
            ]
        },
    ]
}


# %%
GREL_ORGNAME_DP1 = {
    "part": "grel_orgname_dp1",
    "desc": "机构名称1度关联",
    "from_": ["GRAPH_REL"],
    "prikey": ["nid"],
    "depth": 1,
    "ntype": [NodeType.ORGNAME],
    "key_fmt": "orgname_{cond}_{agg}",
    "cond": {
        "link_type": (afrels.link_type_reprs(NodeType.ORGNAME, "both")
                      + pbocrels.link_type_reprs(NodeType.ORGNAME, "both")),
        "dois": last_dois("day_itvl(update, today)"),
        "NONE": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        {
            "aggs": ["cnt",],
            "conds": [
                {
                    "both": (["link_type", "dois"], ["NONE"]),
                }
            ]
        },
    ]
}


# %%
GREL_ADDR_DP1 = {
    "part": "grel_addr_dp1",
    "desc": "地址1度关联",
    "from_": ["GRAPH_REL"],
    "prikey": ["nid"],
    "depth": 1,
    "ntype": [NodeType.ADDR, ],
    "key_fmt": "addr_{cond}_{agg}",
    "cond": {
        "rel_type": (afrels.link_type_reprs(NodeType.ADDR, "both")
                     + pbocrels.link_type_reprs(NodeType.ADDR, "both")),
        "dois": last_dois("day_itvl(update, today)"),
        "NONE": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        {
            "aggs": ["cnt",],
            "conds": [
                {
                    "both": (["rel_type", "dois"], ["NONE"]),
                }
            ]
        },
    ]
}


# %%
GREL_VIN_DP2 = {
    "part": "grel_vin_dp2",
    "desc": "测试多层次-车架号1度关联",
    "from_": ["GRAPH_REL"],
    "prikey": ["nid"],
    "depth": 2,
    "ntype": [NodeType.VIN],
    "key_fmt": "{cond}_{agg}",
    "cond": {
        "vin_rel_type_src": afrels.node_role_reprs(NodeType.VIN, "source"),
        "vin_rel_type_tgt": afrels.node_role_reprs(NodeType.VIN, "target"),
        "certno_ntypes": afrels.ntype_reprs([NodeType.CERTNO]),
        "certno_rel_type": afrels.node_role_reprs(NodeType.CERTNO, "both"),
        "dois": last_dois("day_itvl(update, today)"),
        "NONE": none_of_all(),
    },
    "agg": {
        "cnt": ("cnt", "count(_)", "数量"),
    },
    "cros": [
        {
            "aggs": ["cnt",],
            "conds": [
                {
                    "source": (["vin_rel_type_src", "dois"], ["certno_ntypes"]),
                    "target": (["vin_rel_type_tgt", "dois"], ["certno_ntypes"]),
                },{
                    "both": (["certno_rel_type", ], ["NONE"]),
                }
            ]
        },
    ]
}


# %%
GRAPH_LINKS_CONF = {
    GREL_CERTNO_DP1["part"]     : GREL_CERTNO_DP1,
    GREL_TEL_DP1["part"]        : GREL_TEL_DP1,
    GREL_VIN_DP1["part"]        : GREL_VIN_DP1,
    GREL_ORGNAME_DP1["part"]    : GREL_ORGNAME_DP1,
    GREL_ADDR_DP1["part"]       : GREL_ADDR_DP1,
    GREL_VIN_DP2["part"]        : GREL_VIN_DP2,
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
