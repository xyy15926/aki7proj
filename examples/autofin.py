#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: procedure.py
#   Author: xyy15926
#   Created: 2024-10-06 15:02:13
#   Updated: 2025-02-25 18:12:12
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
    from ubears.flagbear.const import callables
    from ubears.flagbear.llp import parser
    from ubears.modsbear.dflater import ex2df, ex4df
    from ubears.suitbear.dirt import crosconf, exdf
    from ubears.suitbear.autofin import confflat, conftrans, confagg
    from ubears.suitbear.pboc import confflat as pboc_confflat
    from ubears.suitbear.kgraph import kgenum, afrels, pbocrels, display, gxgine
    from ubears.suitbear.autofin import graphagg
    reload(callables)
    reload(parser)
    reload(ex2df)
    reload(ex4df)
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
from IPython.core.debugger import set_trace

from ubears.flagbear.llp.parser import EnvParser
from ubears.flagbear.slp.finer import get_tmp_path, get_assets_path, tmp_file
from ubears.flagbear.slp.pdsl import (save_with_excel,
                                      save_with_pickle,
                                      load_from_pickle)
from ubears.suitbear.kgraph.kgenum import NodeType, df_enum_confs
from ubears.suitbear.kgraph.display import draw_rels
from ubears.suitbear.dirt.exdf import (flat_ft_dfs, trans_from_dfs,
                                       agg_from_dfs, agg_from_graphdf)
from ubears.suitbear.autofin.confflat import df_flat_confs as af_flat_confs
from ubears.suitbear.pboc.confflat import df_flat_confs as pboc_flat_confs
from ubears.suitbear.autofin.conftrans import (df_trans_confs,
                                               MAPPERS, MAPPERS_CODE)
from ubears.suitbear.autofin.confagg import df_agg_confs, PERSONAL_CONF, MASS_CONF
from ubears.suitbear.autofin.graphagg import df_graph_agg_confs, GRAPH_REL, GRAPH_NODE

AUTOFIN_AGG_CONF = {**PERSONAL_CONF, **MASS_CONF}
INFOCODE_MAPPER_FILE = get_assets_path() / "autofin/infocode_mapper.xlsx"


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def merge_certno_perday(df: pd.DataFrame):
    """Merge duplicated records of the same biz with the same certno per-day.
    """
    def drop_and_merge(sub_df: pd.DataFrame):
        # sub_df = subdf.sort_values("apply_date", ascending=False)
        ser = sub_df.iloc[0]
        # ATTENTION: this takes effects just because the precendece of
        # `accept`, `reject` and `validation` are the same with their
        # alphabetic order.
        ser["approval_result"] = sub_df["approval_result"].sort_values().iloc[0]
        ser["approval_codes"] = ",".join(
            set(sub_df["approval_codes"].dropna().values))
        return ser

    df = (df.groupby(["biztype", "channel_code", "certno", "apply_date"])
          .apply(drop_and_merge)
          .reset_index(drop=True))

    return df


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
    envp = EnvParser(env) if envp is None else envp

    # 1. Apply transformation on DataFrames in `dfs`.
    # Transfrom with `auto` or `inplace` will be all fine, as the `from_`s in
    # `trans_pconfs` are all len-1 list.
    tpconfs, tfconfs = df_trans_confs()
    trans_ret = trans_from_dfs(dfs, tpconfs, tfconfs, how="auto", envp=envp)

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

    return dfs, trans_ret, agg_rets


# %%
if __name__ == "__main__":
    write_autofin_confs("autofin/autofin_conf.xlsx")

    # Prepare mock data.
    mock_file = get_assets_path() / "autofin/autofin_mock_20241101.xlsx"
    xlr = pd.ExcelFile(mock_file)
    dfs = {}
    for shname in xlr.sheet_names:
        df = pd.read_excel(xlr, sheet_name=shname)
        dfs[shname] = df

    # Get mapper from customed infocode to standard approval code cats.
    immap = dict(pd.read_excel(INFOCODE_MAPPER_FILE).dropna(subset="code_cat")
                 [["code", "code_cat"]].values)
    MAPPERS_CODE["appr_cats_mapper_lv21"] = immap
    MAPPERS_CODE["today"] = pd.Timestamp("20241101")
    envp = EnvParser(MAPPERS_CODE)

    # Flat the data.
    af_flat_pconfs, af_flat_fconfs = af_flat_confs()
    pboc_flat_pconfs, pboc_flat_fconfs = pboc_flat_confs()
    flat_pconfs = pd.concat([af_flat_pconfs, pboc_flat_pconfs], axis=0)
    flat_fconfs = pd.concat([af_flat_fconfs, pboc_flat_fconfs], axis=0)
    flat_rets = flat_ft_dfs(dfs, flat_pconfs, flat_fconfs, envp=envp)

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
    dfs, trans_ret, agg_rets = autofin_vars(dfs, None, envp=envp)

    # old_dfs = load_from_pickle("autofin/flat_dfs")
    # for key in dfs:
    #     ndf = dfs[key]
    #     odf = old_dfs[key]
    #     eqs = ((ndf == odf) | (pd.isna(ndf) & pd.isna(odf))
    #            | (ndf.applymap(type) == tuple))
    #     assert np.all(eqs)
    # save_with_pickle(dfs, "autofin/flat_dfs")

    # old_agg_rets = load_from_pickle("autofin/agg_dfs")
    # for key in agg_rets:
    #     ndf = agg_rets[key]
    #     odf = old_agg_rets[key]
    #     assert np.all(np.isclose(ndf.values, odf.values, equal_nan=True))
    # save_with_pickle(agg_rets, "autofin/agg_dfs")

    save_with_excel(agg_rets, "autofin/autofin_aggs.xlsx")
    save_with_excel(dfs, "autofin/autofin_flats.xlsx")
    draw_rels(rel_df, node_df, tmp_file("autofin/autofin_graph.html"))
