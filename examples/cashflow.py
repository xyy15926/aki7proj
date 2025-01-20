#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: procedure.py
#   Author: xyy15926
#   Created: 2024-12-17 08:55:39
#   Updated: 2024-12-17 15:31:08
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import List, Tuple
from collections.abc import Mapping, Sequence
import logging

import numpy as np
import pandas as pd

if __name__ == "__main__":
    from importlib import reload
    from ubears.modsbear.dflater import ex2df, ex4df, exenv, exoptim
    from ubears.modsbear.spanner import manidf
    from ubears.suitbear.dirt import exdf
    from ubears.suitbear.cashflow import confflat, confagg, conftrans
    reload(ex2df)
    reload(ex4df)
    reload(exenv)
    reload(exoptim)
    reload(manidf)
    reload(exdf)
    reload(confflat)
    reload(conftrans)
    reload(confagg)

import os
from collections import ChainMap
import sqlalchemy as sa
from tqdm import tqdm
from IPython.core.debugger import set_trace

from ubears.flagbear.llp.parser import EnvParser
from ubears.flagbear.str2.fliper import extract_field
from ubears.flagbear.slp.finer import get_assets_path, get_tmp_path
from ubears.flagbear.slp.pdsl import save_with_excel
from ubears.modsbear.docer.pdframe import extract_tables, format_table
from ubears.modsbear.dflater.exenv import EXGINE_ENV
from ubears.suitbear.dirt.exdf import (flat_ft_dfs, trans_from_dfs,
                                agg_from_dfs, dep_from_fconfs)
from ubears.suitbear.cashflow.confflat import df_flat_confs
from ubears.suitbear.cashflow.conftrans import df_trans_confs, TRANS_ENV
from ubears.suitbear.cashflow.confagg import df_agg_confs


# %%
def write_cashflow_confs(conf_file: str = "cashflow/cashflow_conf.xlsx"):
    """Write POBC confs to Excel.
    """
    dfs = {}

    flat_pconfs, flat_fconfs = df_flat_confs()
    dfs["cashflow_flat_parts"] = flat_pconfs
    dfs["cashflow_flat_fields"] = flat_fconfs

    trans_pconfs, trans_fconfs = df_trans_confs()
    dfs["cashflow_trans_parts"] = trans_pconfs
    dfs["cashflow_trans_fields"] = trans_fconfs

    agg_pconfs, agg_fconfs = df_agg_confs()
    dfs["cashflow_agg_parts"] = agg_pconfs
    dfs["cashflow_agg_fields"] = agg_fconfs

    # Compile to get only necessary fields for `mark_fconfs`.
    cpconfs = pd.concat([trans_pconfs, agg_pconfs], axis=0)
    cfconfs = pd.concat([trans_fconfs, agg_fconfs], axis=0)
    dfs["pboc_all_parts"] = cpconfs
    dfs["pboc_all_fields"] = cfconfs

    save_with_excel(dfs, conf_file)


# %%
def cashflow_vars(
    dfs: dict[str, pd.DataFrame],
    env: Mapping = TRANS_ENV,
    envp: EnvParser = None,
) -> dict[str, pd.DataFrame]:
    """Calculate indicators from PBOC records.

    Params:
    ---------------------------
    dfs: Series of PBOC records.
    agg_key_mark: The necessary aggregation to be calculated.

    Return:
    ---------------------------
    dfs: Dict[part-name, DF updated by transformation result]
    trans_ret: Dict[part-name, DF with transformation result]
    agg_ret: Dict[part-name, DF of aggregation result]
    mark_ret: Dict[part-name, DF of mark]
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    trans_pconfs, trans_fconfs = df_trans_confs()
    agg_pconfs, agg_fconfs = df_agg_confs()

    trans_ret = trans_from_dfs(dfs, trans_pconfs, trans_fconfs,
                               how="auto", envp=envp)
    agg_ret = agg_from_dfs(trans_ret, agg_pconfs, agg_fconfs, envp=envp)

    return dfs, trans_ret, agg_ret


# %%
if __name__ == "__main__":
    write_cashflow_confs("cashflow/cashflow_conf.xlsx")

    # Extract cashflow from pdfs.
    flat_pconfs, flat_fconfs = df_flat_confs()
    files = list((get_assets_path() / "cashflow").iterdir())
    dfs = {}
    for idx, file in enumerate(files):
        if file.suffix.lower() != ".pdf":
            continue
        ftype = "flat_" + file.name[:6]
        table = extract_tables(file)[0]
        # Get colnames, dtypes from `fconfs`.
        fconfs = flat_fconfs[flat_fconfs["part"] == ftype]
        cols = fconfs.iloc[2:, -1].tolist()
        dtypes = dict(fconfs[["desc", "dtype"]].iloc[2:].values)
        colnames = dict(fconfs[["desc", "key"]].iloc[2:].values)
        # Extract table.
        ftable, desc = format_table(table, cols,
                                    drop_chars="\n\t",
                                    dtypes=dtypes)
        ftable.rename(colnames, axis=1, inplace=True)
        ftable["rid"] = f"rid{idx:04}"
        ftable["certno"] = f"certno{idx:04}"
        df = dfs.setdefault(ftype, [])
        df.append(ftable)
    for k, v in dfs.items():
        dfs[k] = pd.concat(v, axis=0)

    dfs, trans_ret, agg_ret = cashflow_vars(dfs)
    save_with_excel(dfs, "cashflow/cashflow_flats.xlsx")
    save_with_excel(agg_ret, "cashflow/cashflow_aggs.xlsx")
