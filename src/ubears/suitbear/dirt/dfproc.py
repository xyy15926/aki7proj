#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dfproc.py
#   Author: xyy15926
#   Created: 2025-02-15 21:06:34
#   Updated: 2025-03-08 18:49:34
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

import numpy as np
import pandas as pd
from scipy.stats import contingency

from ubears.flagbear.str2.dups import rename_duplicated
from ubears.ringbear.sortable import tree_cut
from ubears.ringbear.numeric import edge_encode
from ubears.ringbear.metrics import cal_woes_from_ctab, cal_ivs
# from IPython.core.debugger import set_trace

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

# %%
VALUE_CHECK_CONF = {
    "sort_keys": {
        "id": True,
    },
    "rename_dupcol": True,
    "uni_keys": ["id"],
    "uni_keep": "first",
    "na_thresh": 0.99,
    "flat_thresh": 0.999,
    "fillna": {
        "__NUM__": -999999,
        "__CAT__": "ZNA",
    },
}
VALUE_TRANS_CONF = {
    "__NUM__": [
        {
            "ttype": "binize",
        },{
            "ttype": "woe"
        }
    ],
    "__CAT__": [
        {
            "ttype": "ordinize",
            "nafill": -999999,
        },{
            "ttype": "woe",
        },{
            "ttype": "map",
            "ref": {
                "NA": "ZNA",
            },
            "default": None,
        }
    ],
}
DROP_FIELDS_CONF = {
    "manual": [],
    "pcorr_thresh": 1,
    "iv_thresh": 0.01,
}


# %%
def fixup_df(
    data: pd.DataFrame,
    conf: dict = None,
    exclude: list = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fix DataFrame up according to specified procedures.

    According to the `conf`, one or more process will be done on `data`:
      1. Sort by key.
      2. Rename duplicated columns.
      3. Drop duplicates.
      4. Drop columns with NAs or single values.
      5. Fill NAs.

    ATTENTION:
    1. Index will be reset after `groupby` during drop the duplicates.
      So take care of the correspondant label if necessary.

    Params:
    ---------------------
    data: Data to be fixed.
    conf: Dict specified the fixup procedure.
      sort_keys: Dict[field name, sort order]
      rename_dupcol: if to rename duplicated columns.
      uni_keys: Unique, or group, Keys for drop duplicates.
      uni_keep: How to drop duplicates.
      na_thresh: The threshold for dropping columns with too many NAs.
      flat_thresh: The theshold for dropping columns with too many single
        values.
      fillna: Dict[field name, value to fill NAs].
        __NUM__: Numeric fields.
        __CAT__: Non-numerci fields.
    exclude: The columns to be excluded during the procedure, mainly for the
      `__NUM__` and `__CAT__` in `conf`.

    Return:
    ---------------------
    DataFrame after a chain of process specified in `conf`.
    """
    num_cols = data.select_dtypes(include=np.number).columns
    cat_cols = data.select_dtypes(exclude=np.number).columns
    if exclude is not None:
        num_cols = [coln for coln in num_cols if coln not in exclude]
        cat_cols = [coln for coln in cat_cols if coln not in exclude]

    # Sort.
    sort_keys = conf.get("sort_keys", None)
    if sort_keys is not None and len(sort_keys) > 0:
        sort_by = list(sort_keys.keys())
        sort_ascs = list(sort_keys.values())
        data = data.sort_values(by=sort_by, ascending=sort_ascs)
        logger.info(f"Sort data by: {sort_by}.")

    # Rename duplicated column names.
    rename_dupcol = conf.get("rename_dupcol", None)
    if rename_dupcol:
        colns = rename_duplicated(data.columns)
        if colns is not None:
            data.columns = colns
        logger.info("Rename duplicated column names.")

    # Drop duplicates.
    uni_keys = conf.get("uni_keys", None)
    uni_keep = conf.get("uni_keep", "first")
    if uni_keys is not None and len(uni_keys) > 0:
        # Use default aggregation powered by `drop_duplicates` if possible.
        if uni_keep in ["first", "last", False]:
            data = data.drop_duplicates(subset=uni_keys, keep=uni_keep)
        # Group by unique keys and apply aggregation for each fields.
        else:
            data = (data.groupby(uni_keys, as_index=False, sort=False)
                    .aggregate(uni_keep))
        # Sort again since groupby will shuffle the order.
        if sort_keys:
            data = data.sort_values(by=sort_by, ascending=sort_ascs)
        logger.info(f"Drop duplicated records according to "
                    f"primary keys: {uni_keys}.")

    # Drop columns with too many NAs or single value.
    na_thresh = conf.get("na_thresh", None)
    flat_thresh = conf.get("flat_thresh", None)
    drop_cols = []
    row_n = data.shape[0]
    if na_thresh is not None or flat_thresh is not None:
        for colname in data.columns:
            ser = data[colname]
            vcnts = ser.value_counts(sort=True, ascending=False,
                                     normalize=False, dropna=True)
            nna_n = vcnts.sum()
            na_r = (row_n - nna_n) / row_n
            if ((na_thresh is not None and na_r >= na_thresh)
                or (flat_thresh is not None
                    and len(vcnts) > 0
                    and vcnts.iloc[0] / row_n >= flat_thresh)):
                drop_cols.append(colname)
        if drop_cols:
            data = data.drop(drop_cols, axis=1)
            logger.info(f"Drop columns with to many NAs or single value: "
                        f"{drop_cols}.")

    # Fill NAs for specified columns.
    fillna = conf.get("fillna", None)
    if fillna is not None and len(fillna) > 0:
        data = data.fillna(fillna)
        # Fill NAs for numeric and non-numeric columns seperately.
        num_fna = fillna.get("__NUM__", None)
        if num_fna is not None and len(num_cols):
            data = data.fillna({col: num_fna for col in num_cols})
        cat_fna = fillna.get("__CAT__", None)
        if cat_fna is not None and len(cat_cols):
            data = data.fillna({col: cat_fna for col in cat_cols})
        logger.info("Fill NAs.")

    return data


# %%
def trans_arr(
    arr: pd.Series | np.ndarray,
    label: pd.Series,
    conf: dict = None,
) -> np.ndarray:
    """Transfrom array according to specified procedures.

    Following transfromation may be applied according to config passed:
      1. Ordnize.
      2. Binize.
      3. WOE.

    Params:
    ----------------------
    arr: Sequence to be applied transformation.
    label: Correspondant label, if necessary.
    conf: List of dict specifying the transformation.
      ttype: Transformation type.
      OTHERS: Params for the transformation.

    Return:
    ----------------------
    Transformed array.
    """
    arr = np.asarray(arr)
    label = np.asarray(label)
    for tconf in conf:
        ttype = tconf["ttype"]
        if ttype == "ordinize":
            # Pandas modify the `pd.factorize` in 1.4 and na-fill can't be
            # specified any more.
            ret, codes = pd.factorize(arr)
            arr = ret
        elif ttype == "binize":
            nbin = tconf.get("nbin", 7)
            if np.isscalar(nbin):
                edges, ctab = tree_cut(arr, label, nbin)
                arr = edge_encode(arr, edges)
            else:
                arr = edge_encode(arr, nbin)
        elif ttype == "woe":
            (ux, uy), ctab = contingency.crosstab(arr, label)
            woes, ivs = cal_woes_from_ctab(ctab)
            # Pandas requires to cast to compatiable dtype explicitly first.
            arr = woes[np.searchsorted(ux, arr)]
        elif ttype == "map":
            ref = tconf.get("ref", {})
            z = tconf.get("default", None)
            arr = np.asarray([ref.get(ele, z) for ele in arr]
                             if hasattr(ref, "get")
                             else [ref(ele) for ele in arr])
        else:
            logger.warning(f"Unrecognized transformation: {ttype}.")
    return arr


# %%
def trans_df(
    data: pd.DataFrame,
    label: pd.Series,
    conf: dict = None,
    exclude: list = None,
) -> pd.DataFrame:
    """Transfrom array according to specified procedures.

    Following transfromation may be applied on specified fields according
    to config passed:
      1. Ordnize.
      2. Binize.
      3. WOE.

    Params:
    ----------------------
    data: DF to be applied transformation.
    label: Correspondant label, if necessary.
    conf: Dict[field name, list of dict specifying the transformation]
      Field name:
        __NUM__: Numeric columns.
        __CAT__: Non-numerci columns.
      Transformation config dict:
        ttype: Transformation type.
        OTHERS: Params for the transformation.
    exclude: The columns to be excluded during the procedure, mainly for the
      `__NUM__` and `__CAT__` in `conf`.

    Return:
    ----------------------
    Transformed DataFrame.
    """
    num_cols = data.select_dtypes(include=np.number).columns
    cat_cols = data.select_dtypes(exclude=np.number).columns
    if exclude is not None:
        num_cols = [coln for coln in num_cols if coln not in exclude]
        cat_cols = [coln for coln in cat_cols if coln not in exclude]
    for coln, cconf in conf.items():
        if coln == "__NUM__" or coln == "__CAT__":
            icols = num_cols if coln == "__NUM__" else cat_cols
            for icoln in icols:
                if icoln in conf:
                    continue
                val = trans_arr(data[icoln], label, cconf)
                data[icoln] = val
        else:
            val = trans_arr(data[coln], label, cconf)
            data[coln] = val

    return data


# %%
def drop_fields(
    data: pd.DataFrame,
    label: pd.Series,
    conf: dict = None,
    exclude: list = None,
) -> pd.DataFrame:
    """Drop fields according to specified thresholds.

    Drop fields according to specified thresholds:
      1. manual: Specified manually.
      2. iv_thresh: Information value.
      3. pcorr_thresh: Pearson correlation coefficient inner-data fields.

    Params:
    ----------------------
    data: DF to be applied transformation.
    label: Correspondant label, if necessary.
    conf: Dict[threshold type, threshold]

    Return:
    ----------------------
    DataFrame with fields dropped.
    """
    # Put excluded fields aside.
    excluded = data[exclude]
    data = data.drop(exclude, axis=1)

    # Drop specified fields manually.
    man_dropc = conf.get("manual", None)
    if man_dropc is not None:
        data = data.drop(man_dropc, axis=1)

    # Drop fields with little information value.
    iv_thresh = conf.get("iv_thresh", None)
    if iv_thresh is not None:
        df_num = data.select_dtypes(np.number)
        ivs = cal_ivs(df_num.values, label.values)
        iv_dropc = df_num.columns[ivs < iv_thresh]
        data = data.drop(iv_dropc, axis=1)

    # Drop columns with strong correlation.
    pcorr_thresh = conf.get("pcorr_thresh", None)
    if pcorr_thresh is not None:
        df_num = data.select_dtypes(np.number)
        cols = df_num.columns
        pcorr = df_num.corr(method="pearson").abs()
        ivs = cal_ivs(df_num.values, label.values)

        pcorr_dropc = np.ones(len(cols), dtype=np.bool_)
        for col in cols:
            bmap = pcorr[col] >= pcorr_thresh
            if bmap.sum() > 1:
                rivs = np.select([bmap, ], [ivs, ], 0)
                bmap.iloc[np.argmax(rivs)] = False
                pcorr_dropc &= ~bmap
        pcorr_dropc = df_num.columns[~pcorr_dropc]
        data = data.drop(pcorr_dropc, axis=1)

    data[exclude] = excluded
    return data
