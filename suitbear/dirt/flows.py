#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: flows.py
#   Author: xyy15926
#   Created: 2024-03-14 09:52:48
#   Updated: 2025-01-14 20:19:57
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

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import contingency

from modsbear.dflater.exenv import EXGINE_ENV
from flagbear.str2.dups import rename_duplicated
from ringbear.sortable import tree_cut
from ringbear.numeric import edge_encode
from ringbear.metrics import cal_lifts_from_ctab, cal_woes_from_ctab
from ringbear.metrics import cal_lifts, cal_woes, cal_ivs
from modsbear.spanner.dflog import DFLoggerMixin


Keys = TypeVar("DataFrame Keys")
CAT_NA_FLAG = "ZNA"
NUM_NA_FLAG = -999999
FACTOR_MAX = 10


logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# TODO: Annotations.
class DataProc(DFLoggerMixin):
    def __init__(
        self,
        data: pd.Series | pd.DataFrame,
        label: pd.Series,
        factors: list | dict | int = FACTOR_MAX,
        sort_keys: Keys = None,
        uni_keys: Keys = None,
        uni_keep: Keys = None,
        na_thresh: float = 0.99,
        flat_thresh: float = 0.99,
        key_accs: dict[str, list] = None,
    ):
        """
        1. `data` and `label` should share the same row index.

        Attrs:
        """
        super().__init__()
        self.data = data.copy()
        self.label = label.copy()

        self.factors = factors
        self.sort_keys = sort_keys      # {key: asc, ...}
        self.uni_keys = uni_keys        # [key, ...]
        self.uni_keep = uni_keep        # {key: keep, ...} | str

        self.na_thresh = na_thresh
        self.flat_thresh = flat_thresh
        self.key_accs = {} if key_accs is None else key_accs

    @DFLoggerMixin.rclog(ltag="Rename")
    def check_index(self):
        """Prepare data.

        1. Sort rows.
        2. Rename duplicated column names.
        3. Fill nan with `NUM_NA_FLAG` or `CAT_NA_FLAG`.
        """
        df = self.data

        # Sort rows.
        sort_keys = self.sort_keys
        if sort_keys:
            sort_by = list(sort_keys.keys())
            sort_ascs = list(sort_keys.values())
            df.sort_values(by=sort_by, ascending=sort_ascs, inplace=True)

        # Rename duplicated column names.
        new_colnames = rename_duplicated(df.columns)
        if new_colnames is not None:
            df.columns = new_colnames

        self.label = self.label[df.index]

    @DFLoggerMixin.valog
    @DFLoggerMixin.rclog(ltag="DropDup")
    def drop_duplicates(self):
        """Drop dupicated rows.

        1. Drop duplicated rows according to `uni_keys`.
        2. `DataFrame.drop_duplicates` will be called for simple keep-strategy
            for all columns.
        3. `DataFrame.groupby` and `aggregate` will be called for each column
            with different strategies.
        """
        uni_keys = self.uni_keys
        if not uni_keys:
            return
        df = self.data

        if uni_keys:
            keep = self.uni_keep if self.uni_keep is not None else "first"
            # Use default aggregation powered by `drop_duplicates` if possible.
            if isinstance(keep, str):
                df = df.drop_duplicates(subset=uni_keys, keep=keep)
            # Group by unique keys and apply aggregation for each fields.
            elif isinstance(keep, dict):
                for key in keep:
                    if isinstance(keep[key], str):
                        keep[key] = EXGINE_ENV[key]
                df = df.groupby(uni_keys, sort=False).aggregate(keep)

        # Update `label`.
        self.label = self.label[df.index]
        self.data = df

    @DFLoggerMixin.rclog(ltag="DropNA")
    def drop_flat(self):

        """Drop null, flat columns.
        """
        df = self.data
        na_thresh = self.na_thresh
        flat_thresh = self.flat_thresh

        # No duplicated column names are allowed, since column names are used
        # to mark Series.
        na_cols = []
        flat_cols = []
        row_n = df.shape[0]
        for colname in df.columns:
            ser = df[colname]
            vcnts = pd.value_counts(ser, sort=True, ascending=False,
                                    normalize=False, dropna=True)
            nna_n = vcnts.sum()
            na_r = (row_n - nna_n) / nna_n
            if na_thresh is not None and na_r >= na_thresh:
                na_cols.append(colname)
            if (flat_thresh is not None and len(vcnts) > 0
                    and vcnts.iloc[0] / nna_n >= flat_thresh):
                flat_cols.append(colname)

        # Drop inplace.
        df.drop(na_cols + flat_cols, axis=1, inplace=True)

    @DFLoggerMixin.malog
    def fillna(self):
        df = self.data

        # Fill NaN for numeric columns.
        num_df = df.select_dtypes(include=np.number)
        if num_df is not None and not num_df.empty:
            df.fillna({col: NUM_NA_FLAG for col in num_df.columns},
                      inplace=True)

        # Fill NaN for categorical columns.
        cat_df = df.select_dtypes(exclude=np.number)
        if cat_df is not None and not cat_df.empty:
            df.fillna({col: CAT_NA_FLAG for col in cat_df.columns},
                      inplace=True)

    @DFLoggerMixin.malog
    def ordinize(self, cols: list | tuple | None = None):
        """Convert data into quantitative dtype.

        1. Encode categorical columns with OrdinalEncoder.
        2. NaN will be filled with `NUM_NA_FLAG`.
        """
        df = self.data
        if cols is None:
            cols = df.select_dtypes(exclude=np.number).columns

        # Return directly.
        if len(cols) == 0:
            return

        oe = OrdinalEncoder(dtype=np.int_, encoded_missing_value=NUM_NA_FLAG)
        df.loc[:, cols] = oe.fit_transform(df[cols])

        # mapper = pd.DataFrame(oe.categories_, index=cols).stack()

    @DFLoggerMixin.valog
    @DFLoggerMixin.malog
    def binize(self, cols: list | tuple | None = None):
        """Cut columns into bins.
        """
        df = self.data
        label = self.label
        factor_n = (self.factors if isinstance(self.factors, int)
                    else len(self.factors))

        # All column will be binized by default.
        if cols is None:
            cols = df.columns
        if len(cols) == 0:
            return
        # Binize columns.
        for colname in cols:
            ser = df[colname]
            edges, ctab = tree_cut(ser.values, label.values, factor_n)
            df.loc[:, colname] = edge_encode(ser.values, edges)

            # edge_index = [f"{f},{l}" for f, l in zip(edges[:-1], edges[1:])]
            # ctab_df = pd.DataFrame(ctab, index=edge_index)
            # recs.append(ctab_df)

    @DFLoggerMixin.valog
    @DFLoggerMixin.malog
    def woeze(self, cols: list | tuple | None = None):
        df = self.data
        label = self.label

        # All column will be transformed into WOEs by default.
        if cols is None:
            cols = df.columns
        if len(cols) == 0:
            return
        # Calculate WOEs from crosstab and map values into WOEs.
        for colname in cols:
            ser = df[colname]
            (ux, uy), ctab = contingency.crosstab(ser.values, label.values)
            woes, ivs = cal_woes_from_ctab(ctab)
            df.loc[:, colname] = woes[np.searchsorted(ux, ser.values)]

    @DFLoggerMixin.rclog(ltag="DropPCorr")
    def drop_pcorr(self, thresh: float = 0.8):
        df = self.data
        label = self.label
        df_num = df.select_dtypes(include=np.number)
        cols = df_num.columns

        pcorr = df_num.corr(method="pearson").abs()
        ivs = cal_ivs(df_num.values, label.values)

        col_map = np.ones(len(cols), dtype=np.bool_)
        for col in cols:
            bmap = pcorr[col] >= thresh
            if bmap.sum() > 1:
                rivs = np.select([bmap, ], [ivs, ], 0)
                bmap[np.argmax(rivs)] = False
                col_map &= ~bmap
        col_map = df_num.columns[col_map]

        # Keep non-numeric columns.
        df_cat = df.select_dtypes(exclude=np.number)
        col_map = col_map.append(df_cat.columns)

        self.data = df.loc[:, col_map]
