#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: procedure.py
#   Author: xyy15926
#   Created: 2024-04-22 10:13:57
#   Updated: 2024-09-19 10:39:07
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
    from flagbear import patterns, parser, fliper
    from modsbear import exgine
    from suitbear.pboc import confflat, confagg, conftrans
    reload(patterns)
    reload(parser)
    reload(fliper)
    reload(exgine)
    reload(confflat)
    reload(conftrans)
    reload(confagg)

import os
from collections import ChainMap
import sqlalchemy as sa
from tqdm import tqdm
from IPython.core.debugger import set_trace

from flagbear.patterns import REGEX_TOKEN_SPECS
from flagbear.parser import EnvParser
from flagbear.fliper import extract_field
from modsbear.exgine import (agg_on_df,
                             compress_hierarchy,
                             flat_records,
                             trans_on_df,
                             EXGINE_ENV)
from suitbear.finer import get_assets_path, get_tmp_path
from suitbear.pboc.confflat import PBOC_PARTS, df_flat_confs
from suitbear.pboc.conftrans import MAPPERS, MAPPERS_CODE, TRANS_CONF, df_trans_confs
from suitbear.pboc.confagg import (df_agg_confs,
                                   LV2_AGG_CONF,
                                   LV20_AGG_CONF,
                                   LV1_AGG_CONF)

MAPPERS_CODE["today"] = pd.Timestamp.today()
PBOC_AGG_CONF = {**LV2_AGG_CONF, **LV20_AGG_CONF,
                 **LV1_AGG_CONF}
# agg_pconfs, agg_aconfs = agg_confs()
# flat_parts, flat_fields = flat_confs()


# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def flat_fields(
    src: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Extract fields from PBOC records.

    Params:
    ---------------------------
    src: Series of PBOC records.

    Return:
    ---------------------------
    Dict[part-name, DataFrame of values of parts]
    """
    # Read fields extraction config and addup some default settings.

    # Extract fields.
    dfs = {}
    for pconf in PBOC_PARTS.values():
        psrc = compress_hierarchy(src, pconf["steps"])
        fconf = []
        for key, step, dtype, desc in pconf["fields"]:
            # Set the default values to get the proper dtype for DataFrame
            # constructed in `flat_records` automatically.
            if dtype in REGEX_TOKEN_SPECS:
                fconf.append((key, None, step, dtype,
                              REGEX_TOKEN_SPECS[dtype][-1]))
            else:
                fconf.append((key, step))
        ret = flat_records(psrc, fconf)
        dfs[pconf["part"]] = ret

    return dfs


# %%
def agg_from_dfs(
    dfs: dict[str, pd.DataFrame],
    agg_pconfs: pd.DataFrame,
    agg_fconfs: pd.DataFrame,
    env: Mapping = MAPPERS_CODE,
    envp: EnvParser = None,
    *,
    agg_rets: dict[str, pd.DataFrame] = None,
    engine: sa.engine.Engine = None,
    table_prefix: str = "",
) -> dict[str, pd.DataFrame]:
    """Apply aggregations on DataFrames.

    1. `agg_pconfs` will be treated as the index to determine the order of the
      transformations and aggregations.
    2. As the aggregations are defined on the DataFrames step by step, the
      group keys could be pre-determined at table-granularity in `agg_pconfs`
      and should be the primary key and join key for aggregations result.

    Params:
    ------------------------
    dfs: Dict of DataFrame.
    agg_pconfs: Part-conf for aggregations.
    agg_fconfs: Field-conf for aggregations.
    env: Mapping to provide extra searching space for EnvParser.
    envp: EnvParser to execute string.
      ATTENTION: `env` will be ignored if `envp` is passed.
    agg_rets:
    engine:
    table_prefix:

    Return:
    ------------------------
    Dict[part, DataFrame of aggregation]
    """
    # Init EnvParser.
    if envp is None:
        if env is None:
            envp = EnvParser(EXGINE_ENV)
        else:
            envp = EnvParser(ChainMap(env, EXGINE_ENV))

    agg_rets = {} if agg_rets is None else agg_rets
    # Both the source DataFrames and results in process will be used as the
    # searching space.
    df_space = ChainMap(agg_rets, dfs)
    # 2. Aggregate in `level`-decending order, namely bottom-up.
    for idx, pconf in agg_pconfs.sort_values(by="level",
                                             ascending=False).iterrows():
        part_name = pconf["part"]
        from_ = pconf["from_"]
        join_key = pconf.get("joinkey")
        if not isinstance(join_key, (tuple, list)):
            join_key = [None]
        prikey = pconf["prikey"]

        # Skip if the result already exists
        # Or no aggregation should be applied as the empty `from_` indicates.
        if (part_name in agg_rets) or not isinstance(join_key, (tuple, list)):
            continue

        # 2.1 Prepare DataFrame for aggregation.
        joined_df = df_space[from_[0]]
        ljoin_key = join_key[0]
        if joined_df.empty:
            agg_rets[part_name] = pd.DataFrame()
            continue

        # Join the DataFrames indicated by `from_` one-by-one.
        for rpart_name, rjoin_key in zip(from_[1:], join_key[1:]):
            joined_df = joined_df.merge(df_space[rpart_name],
                                        how="left",
                                        left_on=ljoin_key,
                                        right_on=rjoin_key)

        # 2.2 Aggregate.
        agg_rules = agg_fconfs.loc[agg_fconfs["part"] == part_name,
                                   ["key", "cond", "agg"]].values
        tqdm.pandas(desc=part_name)
        agg_ret = (joined_df.groupby(prikey)
                   .progress_apply(agg_on_df, rules=agg_rules, envp=envp))
        agg_rets[part_name] = agg_ret

        if engine is not None:
            today = np.datetime64("today").astype(str).replace("-","")
            try:
                # 1. Unique index, namely Index.unique is True, will be set as
                #   key automatically, which is not compatiable with `TEXT`
                #   dtype, which is the default dtype for object.
                # 2. Value in DataFrame may not be compatiable with MYSQL's
                #   dtype.
                agg_ret.reset_index().to_sql(
                    name=f"{table_prefix}{part_name}_{today}",
                    con=engine,
                    index=False,
                    if_exists="fail")
            except Exception as e:
                logger.warning(e)

    return agg_rets


# %%
def write_pboc_confs(conf_file: str):
    """Write POBC parsing confs to Excel.
    """
    xlw = pd.ExcelWriter(conf_file)

    trans_confs = df_trans_confs()
    trans_confs.to_excel(xlw, sheet_name="pboc_vars_trans")
    mappers = pd.concat([pd.DataFrame(v).T for v in MAPPERS.values()],
                        axis=0,
                        keys=MAPPERS.keys())
    mappers.to_excel(xlw, sheet_name="pboc_vars_maps")

    agg_pconfs, agg_fconfs = df_agg_confs(PBOC_AGG_CONF)
    agg_pconfs.to_excel(xlw, sheet_name="pboc_vars_parts")
    agg_fconfs.to_excel(xlw, sheet_name="pboc_vars_aggs")

    xlw.close()


# %%
def pboc_vars(
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
    for part_name, conf in TRANS_CONF.items():
        trans_rules = [(key, cond, trans) for key, trans, cond, desc
                       in conf["trans"]]
        df = dfs[part_name]
        if not df.empty:
            dfs[part_name] = trans_on_df(df, trans_rules, envp=envp)

    # 2. Construct and apply aggregation config.
    agg_pconfs, agg_fconfs = df_agg_confs(PBOC_AGG_CONF)
    if agg_key_mark is not None:
        agg_fconfs = agg_fconfs[agg_fconfs["key"].isin(agg_key_mark)]
    ret = agg_from_dfs(dfs, agg_pconfs, agg_fconfs, envp=envp)

    return dfs, ret


# %%
def write2file(
    dfs: dict[pd.DataFrame],
    filename: str
) -> None:
    """Write dict of DataFrame into Excel.
    """
    xlw = pd.ExcelWriter(filename)
    for part, var_df in dfs.items():
        if not var_df.empty:
            stop = 0
            while stop * 10000 < var_df.shape[1]:
                var_df.iloc[:, stop * 10000: (stop + 1) * 10000].to_excel(
                    xlw, sheet_name=f"{part}_part{stop}")
                stop += 1
    xlw.close()


# %%
def pboc_from_files(
    files: List,
    today: str = "report",
    fields_file: str = "pboc_fields.xlsx",
    vars_file: str = "pboc_aggvars.xlsx",
    agg_keys: str = None,
) -> pd.DataFrame:
    """Extract PBOC fields from files and apply aggregations.

    Params:
    -----------------------
    files: List of PBOC reports filenames with JSON format.
    today: How to set the report parsing day.
    fields_file: Filenames for writing the PBOC fields back.
    vars_file: Filenames for writing the PBOC aggregation results back.
    agg_keys: Files storing the aggregation keys for calculation. All
      aggregations will be calculated if not provied.
    """
    # files = list((get_assets_path() / "pboc_reports").iterdir())
    # today = "report"
    # fields_file = "pboc_fields.xlsx"
    # vars_file = "pboc_vars.xlsx"
    # agg_keys = None

    report_recs = {}
    for file in files:
        report = open(file, "r").read()
        report_id = extract_field(report, "PRH:PA01:PA01A:PA01AI01")
        report_recs[report_id] = report

    dfs = flat_fields(pd.Series(report_recs))

    if today == "report":
        basic_info = dfs["pboc_basic_info"]
        report_dates = basic_info.set_index("PA01AI01")["PA01AR01"].rename("today")
        for part_name, df in dfs.items():
            if part_name == "pboc_basic_info" or df.empty:
                continue
            df = pd.merge(df, report_dates, how="left",
                          left_on="rid", right_index=True)
            dfs[part_name] = df

    if isinstance(agg_keys, (str, os.PathLike)) and os.path.isfile(agg_keys):
        agg_keys = pd.read_excel(agg_keys)["key"]

    dfs, agg_rets = pboc_vars(dfs, agg_keys)
    write2file(dfs, fields_file)
    write2file(agg_rets, vars_file)

    return dfs, agg_rets


# %%
if __name__ == "__main__":
    # PBOC_JSON = os.path.join(ASSETS, "pboc_utf8.json")
    aggconf_file = "pboc_aggconf.xlsx"
    write_pboc_confs(aggconf_file)

    files = list((get_assets_path() / "pboc_reports").iterdir())
    today = "report"
    fields_file = "pboc_fields.xlsx"
    vars_file = "pboc_vars.xlsx"
    # agg_keys_file = get_assets_path() / "pboc_aggconf_mark.xlsx"
    fields, aggs = pboc_from_files(files, today,
                                   fields_file,
                                   vars_file)
