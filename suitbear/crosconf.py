#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: crosconf.py
#   Author: xyy15926
#   Created: 2024-04-19 21:09:34
#   Updated: 2024-04-19 21:56:11
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
import json
from typing import Any, TypeVar
from collections.abc import Mapping
from collections import ChainMap
from itertools import product
import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace

from flagbear.parser import EnvParser
from flagbear.fliper import rebuild_dict, extract_field
from flagbear.fliper import regex_caster
from flagbear.exgine import rebuild_rec2df, trans_on_df, agg_on_df, EXGINE_ENV

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def cproduct_aggs_and_filters(
    aggs: list,
    filters: list,
    varn_fmt: str = "{}_{}",
) -> pd.DataFrame:
    """Generate Cartesian Product for aggregations and filters.

    1. Cartesian Product of condition-groups will be generated with the
      filters in `filters` without aggregations.
    2. Aggregations and the condition-groups will be grouped together later
      to get configuration of how to aggregate, namely the outer Cartesian
      Product.

    Params:
    -----------------------
    aggs: List of 3-tuples[NAME, FUNCTION, COMMENT, LVUP_FLAGS] describing the
      aggregations.
      LVUP_FLAGS: Flags propogated for later hanlder. This is useful when some
        infomation can't be gotten currently and should be handled after
        joining with other parts.
    filters: List of 3-tuples[NAME, FILTER, COMMENT] describing the
      the filters.
    varn_fmt: Formation string for constructing the key's names.

    Return:
    -----------------------
    DataFrame with columns[key, conds, aggs, cmt], which fits for
      `exgine.agg_part`.
    """
    conds = []
    # Outer Cartesian Product of aggregations and condition-groups.
    for agg_varn, agg_fn, agg_cmt, *lvup_flags in aggs:
        lvup_flags = lvup_flags[0] if lvup_flags else None

        # Inner Cartesian Product of conditions.
        for cond_grp in product(*filters):
            # Zip to get the list of NAMEs, CONDITIONS and COMMENTs.
            cond_varn, cond_cond, cond_cmt = [
                # FILTER in conditions may be None, which need to be filtered
                # out.
                list(filter(lambda x: x is not None, ele))
                for ele in zip(*cond_grp)
            ]
            cond_varn_str = "_".join(cond_varn)
            cond_cond_str = " & ".join([f"({ele})" for ele in cond_cond])
            cond_cmt_str = "".join(cond_cmt)

            conds.append((varn_fmt.format(cond_varn_str, agg_varn),
                          cond_cond_str,
                          agg_fn,
                          cond_cmt_str + agg_cmt,
                          lvup_flags))

    return pd.DataFrame.from_records(conds,
                                     columns=["key", "conds", "aggs",
                                              "cmt", "lvup_flags"])


# %%
def cross_aggs_and_filters(
    cros: tuple[list, list],
    aggs_D: dict[str, list],
    filters_D: dict[str, list | dict],
    key_fmt: str = "{}_{}",
) -> pd.DataFrame:
    """Cross the aggregations and filters according to `cros`.

    There may be many aggregations and filters in some area while exploring the
    data. But not all of them can be combined together with simple Cartesian
    Product or not all the branches in a filter are compatible with the
    aggregation.
    So `cros` is provided here to specify the reasonable pairs of aggregation
    and filter, and Cartesian Product will be done only on the specified pairs.

    Params:
    ----------------------
    cros: 2-Tuple[AGG-CODE-LIST, FILTER-CODE-LIST], indicating how to combine
      aggregations and filters.
      AGG-CODE-LIST: List of str indicating the aggregations.
      FILTER-CODE-LIST: List of str or tuple indicating the filters.
        tuple: [FILTER-CODE, FILTER-BRANCHES,...]
        str-tuple: <FILTER-CODE>.[FILTRE-BRANCHES,...]
        simple str: filter-code from `filters_D`
    aggs_D: Dict[AGG-CODE, 3-Tuple aggregation description], from which to get
      the aggregation description with the code.
    filters_D: Dict[FILTER-CODE, dict or list of filter description], from which
      to get the filter description with the code.
      list: [(NAME, FILTER, COMMENT), ...]
      dict: {FILTER-BRANCH: (NAME, FILTER, COMMENT), ...}

    Return:
    ----------------------
    DataFrame with columns[key, conds, aggs, cmt], which fits for
      `exgine.agg_part`.
    """
    conf_dfs = []
    for agg_codes, filter_indcs in cros:
        # Get filter descriptions from the `filters_D`.
        filters_ = []
        for filter_indc in filter_indcs:
            # For filter indicator with format: (key, sub_key, sub_key,...)
            if isinstance(filter_indc, (tuple, list)):
                filter_code, *branches = filter_indc
            # For filter indicator with format: "key.[sub_key,sub_key]"
            else:
                filter_code, *branches = filter_indc.split(".")
                if branches:
                    branches = [ele.strip() for ele in
                                branches[0][1: -1].split(",")]

            filter_ = filters_D[filter_code]
            # Choose specified branches only.
            if branches:
                filters_.append([filter_[br] for br in branches])
            # Choose all branches from dict or list.
            elif isinstance(filter_, dict):
                filters_.append(list(filter_.values()))
            else:
                filters_.append(filter_)

        # Get aggregation descriptions from the `aggs_D`.
        aggs_ = [aggs_D[an] for an in agg_codes]

        # Generate Cartesian Product of aggregations and filters.
        df = cproduct_aggs_and_filters(aggs_, filters_, key_fmt)

        conf_dfs.append(df)

    if conf_dfs:
        cdf = pd.concat(conf_dfs, axis=0)
    else:
        cdf = pd.DataFrame(columns=["key", "conds", "aggs", "cmt"])

    return cdf


# %%
def agg_confs_from_dict(confs: dict):
    """Generate aggregation configurations.

    Params:
    --------------------------
    confs: Dict[PART-NAME, PART-CONF] of the whole aggregation configuration.
      PART-CONF: Dict of the part's aggregation configuration with:
        part: Part name that should be the same with PART-NAME.
        level: Granularity of the part.
        from_: The parts from which to get current part.
        prikey: The group key for aggregation, which should also be the
          join key when joining with other part with the same level.
        aggs: Dict of aggregations fitting for `cros_aggs_and_filters`.
        conds: Dict of filters fitting for `cros_aggs_and_filters`.
        cros: List of aggregation and filter pair fitting for
          `cross_aggs_and_filters`.

    Return:
    --------------------------
    pconfs: DataFrame with columns[part, level, from_, prikey] describing
      the relations among the parts.
    aconfs: DataFrame with columns[part, key, conds, aggs, cmt] describing
      the aggregations.
    """
    pconfs = []
    aconfs = {}
    for pname, pconf in confs.items():
        pname = pconf["part"]
        pconfs.append((pconf["part"], pconf["level"], pconf["from_"],
                       pconf["prikey"]))
        aconfs[pname] = cross_aggs_and_filters(pconf["cros"],
                                               pconf["aggs"],
                                               pconf["conds"],
                                               pconf["key_fmt"])

    pconfs = pd.DataFrame.from_records(pconfs,
                                       columns=["part", "level",
                                                "from_", "prikey"])
    aconfs = pd.concat(aconfs.values(), keys=aconfs.keys()).droplevel(level=1)
    aconfs.index.name = "part"

    return pconfs, aconfs


# %%
def cross_aggs_from_lower(
    agg_desc: pd.Series,
    cros_D: dict[str, list],
    aggs_D: dict[str, list],
    filters_D: dict[str, list | dict],
    key_fmt: str = "{}_{}",
) -> pd.DataFrame:
    """
    """
    key = agg_desc["key"]
    cmt = agg_desc["cmt"]
    flags = agg_desc["lvup_flags"] or []

    aggs_D = {agg_code: (name.format(key), func.format(key), cmt_.format(cmt))
              for agg_code, (name, func, cmt_) in aggs_D.items()}
    cros = [([flag,], cros_D[flag]) for flag in flags]

    return cross_aggs_and_filters(cros, aggs_D, filters_D, key_fmt)


