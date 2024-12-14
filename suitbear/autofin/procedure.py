#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: procedure.py
#   Author: xyy15926
#   Created: 2024-10-06 15:02:13
#   Updated: 2024-12-14 19:17:44
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple
from collections.abc import Mapping
import logging

import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from modsbear.dflater import ex2df, ex4df, exenv
    from suitbear.dirt import crosconf, exdf
    from suitbear.autofin import confflat, conftrans, confagg
    from suitbear.pboc import confflat as pboc_confflat
    from suitbear.kgraph import kgenum, afrels, pbocrels, display, gxgine
    from suitbear.autofin import graphagg
    reload(ex2df)
    reload(ex4df)
    reload(exenv)
    reload(crosconf)
    reload(exdf)
    reload(confflat)
    reload(conftrans)
    reload(confagg)
    reload(pboc_confflat)
    reload(kgenum)
    reload(afrels)
    reload(pbocrels)
    reload(gxgine)
    reload(graphagg)

import os
from collections import ChainMap
import sqlalchemy as sa
from tqdm import tqdm
from IPython.core.debugger import set_trace

from flagbear.llp.parser import EnvParser
from flagbear.slp.finer import get_tmp_path, get_assets_path
from flagbear.slp.pdsl import save_with_excel
from modsbear.dflater.exenv import EXGINE_ENV
from suitbear.kgraph.kgenum import NodeType, df_enum_confs
from suitbear.kgraph.display import save_as_html
from suitbear.dirt.exdf import (flat_ft_dfs, trans_from_dfs,
                                agg_from_dfs, agg_from_graphdf)
from suitbear.autofin.confflat import df_flat_confs as af_flat_confs
from suitbear.pboc.confflat import df_flat_confs as pboc_flat_confs
from suitbear.autofin.conftrans import (df_trans_confs, merge_certno_perday,
                                        MAPPERS, MAPPERS_CODE)
from suitbear.autofin.confagg import df_agg_confs, PERSONAL_CONF, MASS_CONF
from suitbear.autofin.graphagg import df_graph_agg_confs, GRAPH_REL, GRAPH_NODE

AUTOFIN_AGG_CONF = {**PERSONAL_CONF, **MASS_CONF}

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def write_autofin_confs(conf_file: str):
    """Write AutoFin parsing confs to Excel.
    """
    dfs = {}
    flat_pconfs, flat_fconfs = af_flat_confs()
    dfs["autofin_flat_parts"] = flat_pconfs
    dfs["autofin_flat_fields"] = flat_fconfs

    trans_pconf, trans_fconf = df_trans_confs()
    dfs["autofin_trans_parts"] = trans_pconf
    dfs["autofin_trans_fields"] = trans_fconf
    mappers = pd.concat([pd.DataFrame(v).T for v in MAPPERS.values()], axis=0,
                        keys=MAPPERS.keys())
    dfs["autofin_trans_maps"] = mappers

    agg_pconfs, agg_fconfs = df_agg_confs(AUTOFIN_AGG_CONF)
    dfs["autofin_agg_parts"] = agg_pconfs
    dfs["autofin_agg_fields"] = agg_fconfs

    dfs["graph_enum_types"] = df_enum_confs()

    pboc_pconfs, pboc_node_confs, pboc_rel_confs = pbocrels.df_graph_confs()
    dfs["pboc_graph_parts"] = pboc_pconfs
    dfs["pboc_graph_nodes"] = pboc_node_confs
    dfs["pboc_graph_rels"] = pboc_rel_confs

    af_gconfs, af_node_confs, af_rel_confs = afrels.df_graph_confs()
    dfs["autofin_graph_parts"] = af_gconfs
    dfs["autofin_graph_nodes"] = af_node_confs
    dfs["autofin_graph_rels"] = af_rel_confs

    graph_agg_pconfs, graph_agg_fconfs = df_graph_agg_confs()
    dfs["graph_var_parts"] = graph_agg_pconfs
    dfs["graph_var_aggs"] = graph_agg_fconfs

    return save_with_excel(dfs, conf_file)


# %%
def autofin_vars(
    dfs: dict[str, pd.DataFrame],
    agg_key_mark: pd.Series | set | list = None,
    env: Mapping = MAPPERS_CODE,
    envp: EnvParser = None,
) -> dict[str, pd.DataFrame]:
    """Calculate index from PBOC records.

    Params:
    ---------------------------
    dfs: Series of PBOC records.
    agg_key_mark: The necessary aggregation to be calculated.

    Return:
    ---------------------------
    Dict[part-name, aggregation result]
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    # 1. Apply transformation on DataFrames in `dfs`.
    # Transfrom with `auto` or `inplace` will be all fine, as the `from_`s in
    # `trans_pconfs` are all len-1 list.
    tpconfs, tfconfs = df_trans_confs()
    trans_from_dfs(dfs, tpconfs, tfconfs, how="auto", envp=envp)

    # 2. Construct and apply aggregation config.
    agg_rets = {}
    agg_pconfs, agg_fconfs = df_agg_confs(AUTOFIN_AGG_CONF)
    if agg_key_mark is not None:
        agg_fconfs = agg_fconfs[agg_fconfs["key"].isin(agg_key_mark)]
    # Only transformed parts will be registered in `trans_ret`, so the original
    # `dfs` with transformed parts will be used.
    agg_from_dfs(dfs, agg_pconfs, agg_fconfs, envp=envp, agg_rets=agg_rets)

    # 3. Construct and apply graph aggregations.
    gpconfs, gfconfs = df_graph_agg_confs()
    if agg_key_mark is not None:
        gfconfs = gfconfs[gfconfs["key"].isin(agg_key_mark)]
    agg_from_graphdf(dfs, gpconfs, gfconfs, envp=envp, agg_rets=agg_rets)

    return dfs, agg_rets


# %%
if __name__ == "__main__":
    from flagbear.slp.pdsl import save_with_pickle, load_from_pickle

    write_autofin_confs("autofin/autofin_conf.xlsx")

    # Prepare mock data.
    mock_file = get_assets_path() / "autofin/autofin_mock_20241101.xlsx"
    af_flat_pconfs, af_flat_fconfs = af_flat_confs()
    pboc_flat_pconfs, pboc_flat_fconfs = pboc_flat_confs()
    flat_pconfs = pd.concat([af_flat_pconfs, pboc_flat_pconfs], axis=0)
    flat_fconfs = pd.concat([af_flat_fconfs, pboc_flat_fconfs], axis=0)
    xlr = pd.ExcelFile(mock_file)
    dfs = {}
    for shname in xlr.sheet_names:
        df = pd.read_excel(xlr, sheet_name=shname)
        dfs[shname] = df
    flat_rets = flat_ft_dfs(dfs, flat_pconfs, flat_fconfs)

    # Build graph DF.
    afrel_df, afnode_df = afrels.build_graph_df(dfs)
    pbrel_df, pbnode_df = pbocrels.build_graph_df(dfs)
    rel_df = (pd.concat([afrel_df, pbrel_df])
              .sort_values("update")
              .drop_duplicates(subset=["source", "target"], keep="last"))
    node_df = (pd.concat([afnode_df, pbnode_df])
               .drop_duplicates())
    dfs[GRAPH_REL] = rel_df
    dfs[GRAPH_NODE] = node_df
    for ele in NodeType:
        dfs[ele.value] = node_df

    # Merge records first.
    dfs["autofin_pretrial"] = merge_certno_perday(dfs["autofin_pretrial"])
    dfs["autofin_sectrial"] = merge_certno_perday(dfs["autofin_sectrial"])

    # Set `today` with the date in the `mock_file`.
    MAPPERS_CODE["today"] = pd.Timestamp("20241101")
    dfs, agg_rets = autofin_vars(dfs, None, env=MAPPERS_CODE)
    # save_with_pickle(agg_rets, "autofin/agg_dfs")
    # old_agg_rets = load_from_pickle("autofin/agg_dfs")
    # for key in agg_rets:
    #     ndf = agg_rets[key]
    #     odf = old_agg_rets[key]
    #     assert np.all(np.isclose(ndf.values, odf.values, equal_nan=True))

    save_with_excel(agg_rets, "autofin/autofin_aggs.xlsx")
    save_with_excel(dfs, "autofin/autofin_flats.xlsx")
    save_as_html(rel_df, node_df, "autofin/autofin_graph.xlsx")
