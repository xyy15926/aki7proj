#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: crosconf.py
#   Author: xyy15926
#   Created: 2024-09-30 09:28:51
#   Updated: 2024-11-06 14:30:59
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, TypeVar
from collections.abc import Callable, Iterator
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from IPython.core.debugger import set_trace

from itertools import product, chain

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def cross_aggs_and_filters(
    cros: tuple[list, list],
    aggs_D: dict[str, list],
    filters_D: dict[str, list | dict],
    key_fmt: str = "{cond}_{agg}",
) -> list[tuple]:
    """Cross the aggregations and filters according to `cros`.

    There may be many aggregations and filters in some area while exploring the
    data. But not all of them can be combined together with simple Cartesian
    Product or not all the branches in a filter are compatible with the
    aggregation.
    So `cros` is provided here to specify the reasonable pairs of aggregation
    and filter, and Cartesian Product will be done only on the specified pairs.

    Params:
    ----------------------
    cros: List of 2-Tuple[AGG-CODE-LIST, FILTER-CODE-LIST], indicating how to
      pair aggregations and filters.
      AGG-CODE-LIST: List of str indicating the aggregations.
      FILTER-CODE-LIST: List of str indicating the filters.
    aggs_D: Dict[AGG-CODE, aggregation description], from which to get
      the aggregation description with the code.
    filters_D: Dict[FILTER-CODE, dict or list of filter description], from
      which to get the filter description with the code.
      list: [(NAME, FILTER, COMMENT), ...]
      dict: {FILTER-BRANCH: (NAME, FILTER, COMMENT), ...}

    Return:
    ----------------------
    DataFrame with columns[key, conds, agg, cmt], which fits for
      `fxgine.agg_from_dfs`.
    """
    confs = []
    for agg_codes, filter_codes in cros:
        # Get filter and aggregation descriptions from the `filters_D`.
        filters_ = [filters_D[fn] for fn in filter_codes]
        aggs_ = [aggs_D[an] for an in agg_codes]

        conds = []
        # Outer Cartesian Product of aggregations and condition-groups.
        for agg_varn, agg_fn, agg_cmt in aggs_:
            # Inner Cartesian Product of conditions.
            for cond_grp in product(*filters_):
                # Zip to gather the list of NAMEs, CONDITIONs and COMMENTs.
                # And filter `NONE` in NAMEs, CONDITIONs and COMMENTs.
                cond_varn, cond_cond, cond_cmt = [
                    list(filter(lambda x: x is not None, ele))
                    for ele in zip(*cond_grp)
                ]
                cond_varn_str = "_".join(cond_varn)
                cond_cond_str = " & ".join([f"({ele})" for ele in cond_cond])
                cond_cmt_str = "".join(cond_cmt)

                # `_` will be repeated in some formation string with no filter set.
                key_str = (key_fmt.format(cond=cond_varn_str, agg=agg_varn)
                           .strip("_")
                           .replace("__", "_"))
                conds.append((key_str, cond_cond_str, agg_fn,
                              cond_cmt_str + agg_cmt))
        confs.extend(conds)

    # Drop duplicated conf items.
    conf_dict = {}
    for key, cond, agg, cmt in confs:
        conf_dict.setdefault(key, (cond, agg, cmt))
    confs = [(key, *val) for key, val in conf_dict.items()]

    return confs


# %%
def cross_graph_aggs_and_filters(
    cros: tuple[list, list],
    aggs_D: dict[str, list],
    filters_D: dict[str, list | dict],
    key_fmt: str = "{cond}_{agg}",
) -> list[tuple]:
    """Cross the aggregations and filters according to `cros`.

    There may be many aggregations and filters in some area while exploring the
    data. But not all of them can be combined together with simple Cartesian
    Product or not all the branches in a filter are compatible with the
    aggregation.
    So `cros` is provided here to specify the reasonable pairs of aggregation
    and filter, and Cartesian Product will be done only on the specified pairs.

    Params:
    ----------------------
    cros: List of dict[aggs: AGG-CODE-LIST, conds: FILTER-CODE-DICT], indicating
      how to pair aggregations and filters.
      AGG-CODE-LIST: List of str indicating the aggregations.
      FILTER-CODE-LIST: Dict indicating the filters for source, target or both.
    aggs_D: Dict[AGG-CODE, aggregation description], from which to get
      the aggregation description with the code.
    filters_D: Dict[FILTER-CODE, dict or list of filter description], from
      which to get the filter description with the code.
      list: [(NAME, FILTER, COMMENT), ...]
      dict: {FILTER-BRANCH: (NAME, FILTER, COMMENT), ...}

    Return:
    ----------------------
    DataFrame with columns[key, conds, agg, cmt], which fits for
      `fxgine.agg_from_dfs`.
    """
    confs = []
    for cros_item in cros:
        # Get filter and aggregation descriptions from the `filters_D`.
        aggs_ = [aggs_D[an] for an in cros_item["aggs"]]

        conds = []
        # Outer Cartesian Product of aggregations and condition-groups.
        for agg_varn, agg_fn, agg_cmt in aggs_:
            # Cross different conditions from different steps.
            for d_filter_grps in product(*[ele.items() for ele in
                                           cros_item["conds"]]):
                # Seperate directions and conds.
                directions, filter_grps = list(zip(*d_filter_grps))
                filters = [filters_D[fc] for fc in
                           chain.from_iterable(
                               chain.from_iterable(filter_grps))]
                fns = [(len(relc), len(nodec)) for relc, nodec in filter_grps]

                # Inner Cartesian Product of conditions.
                for cond_grp in product(*filters):
                    # Zip to gather the list of NAMEs, CONDITIONs and COMMENTs.
                    # And filter `NONE` in NAMEs, CONDITIONs and COMMENTs.

                    # ATTENTION: No None will be filtered out to keep the
                    # length unchanged.
                    cond_varn, cond_cond, cond_cmt = zip(*cond_grp)

                    cond_varn_str = "_".join(
                        filter(lambda x: x is not None, cond_varn))
                    cond_cond_strs = []
                    start, end = 0, 0
                    for direction, (rel_step, node_step) in zip(directions, fns):
                        end += rel_step
                        rel_ccs = " & ".join([f"({ele})"
                                              for ele in cond_cond[start: end]
                                              if ele is not None])
                        start = end

                        end += node_step
                        node_ccs = " & ".join([f"({ele})"
                                               for ele in cond_cond[start: end]
                                               if ele is not None])
                        start = end

                        cond_cond_strs.append((direction, rel_ccs, node_ccs))
                    cond_cmt_str = "".join(
                        filter(lambda x: x is not None, cond_cmt))

                    # `_` will be repeated in some formation string with no filter set.
                    key_str = (key_fmt.format(cond=cond_varn_str, agg=agg_varn)
                               .strip("_")
                               .replace("__", "_"))
                    conds.append((key_str, cond_cond_strs, agg_fn,
                                  cond_cmt_str + agg_cmt))
                confs.extend(conds)

    # Drop duplicated conf items.
    conf_dict = {}
    for key, cond, agg, cmt in confs:
        conf_dict.setdefault(key, (cond, agg, cmt))
    confs = [(key, *val) for key, val in conf_dict.items()]

    return confs
