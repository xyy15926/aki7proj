#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dlog.py
#   Author: xyy15926
#   Created: 2023-12-05 08:55:37
#   Updated: 2024-03-13 18:51:09
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

from functools import wraps, partial
from datetime import date
import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from collections import defaultdict
from scipy.stats import contingency
from sklearn.preprocessing import OrdinalEncoder

from flagbear.patterns import CALLABLE_ENV
from ringbear.numeric import std_outlier
from ringbear.metrics import cal_lifts_from_ctab, cal_woes_from_ctab
from ringbear.metrics import cal_lifts, cal_woes, cal_ivs
from ringbear.sortable import tree_cut
from ringbear.numeric import edge_encode
from flagbear.fliper import rename_duplicated


Keys = TypeVar("DataFrame Keys")
CAT_NA_FLAG = "ZNA"
NUM_NA_FLAG = -999999
FACTOR_MAX = 10
EPSILON = 1e-6

RC_NAME = "RC"
MA_NAME = "MAP"
VA_NAME = "VAL"
PCORR_NAME = "PCORR"
ROWN_NAME = "__ROW_NUM__"
NUM_NAME = "NUM"
CAT_NAME = "CAT"
INIT_STATE = "ori"


logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class DFLoggerMixin:
    """Logger for DataFrame processing.

    1. DataFrame will be used to record the infomation.
    2. Most of the logger methods are staticmethod intended for decorating to
      gathering the necessary sketches.

    Attrs:
      data: pd.DataFrame
        Data.
      label: pd.Series
        Labels.
      stage: str
        String indicating the stage of data processing.
      logs: dict[str: DataFrame]
        Dict storing DataFrame of
        {
          stage:{
            "__RC__": DF sketch down to column level,
            "__MAP__": DF of mapping of values,
            "__VAL__": DF sketch down to value level,
            "__PCORR__": DF of pearson correlation,
          }
        }
    """
    def __init__(self):
        self.data = None
        self.label = None
        self.stage = INIT_STATE
        self.logs = {}

    @staticmethod
    def rclog(mark: str = None, type_: str = RC_NAME):
        def decorate(func: callable):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                new_stage = kwargs.pop("new_stage", None)
                if new_stage is not None:
                    self.stage = new_stage
                ori_cols = self.data.columns
                ori_rows = self.data.index

                ret = func(self, *args[1:], **kwargs)

                df = self.data
                nonlocal mark
                if mark is None:
                    mark = func.__name__

                log = dict.fromkeys(set(ori_cols) - set(df.columns), 1)
                log.update(dict.fromkeys(set(df.columns) - set(ori_cols), -1))
                log[ROWN_NAME] = len(df.index) - len(ori_rows)
                log = pd.DataFrame({mark: log})

                slogs = self.logs.setdefault(self.stage, {})
                if type_ in slogs:
                    slogs[type_] = pd.concat([slogs[type_], log], axis=1)
                else:
                    slogs[type_] = log

                pcorr = self.data.select_dtypes(
                    include=["integer", "floating"]
                ).corr(method="pearson")
                slogs[PCORR_NAME] = pcorr

                return ret
            return wrapper
        return decorate

    @staticmethod
    def valog(func: callable = None, type_: str = RC_NAME):
        if func is None:
            return partial(__class__.valog, type_=type_)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            new_stage = kwargs.pop("new_stage", None)
            if new_stage is not None:
                self.stage = new_stage

            ret = func(self, *args[1:], **kwargs)

            # Sketch for data down to both value level and column level.
            # Value level: crosstabs, lifts, woes, ivs
            # Column level: Chi2, IV
            df = self.data
            label = self.label
            uni_dfs = []
            col_Ds = []
            cols = df.columns
            for colname in cols:
                ser = df[colname]
                uni_df, col_D = serlog(ser, label)
                uni_dfs.append(uni_df)
                col_Ds.append(col_D)

            slogs = self.logs.setdefault(self.stage, {})
            slogs[VA_NAME] = pd.concat(uni_dfs, keys=cols)

            log = pd.DataFrame(col_Ds, index=cols)
            slogs = self.logs.setdefault(self.stage, {})
            if type_ in slogs:
                slogs[type_] = pd.concat([slogs[type_], log], axis=1)
            else:
                slogs[type_] = log

            return ret
        return wrapper

    @staticmethod
    def malog(func: callable = None, type_: str = MA_NAME):
        if func is None:
            return partial(__class__.malog, type_=type_)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            new_stage = kwargs.pop("new_stage", None)
            if new_stage is not None:
                self.stage = new_stage
            df_ori = self.data.copy()

            ret = func(self, *args[1:], **kwargs)

            # Build the mappings DataFrame.
            df_new = self.data
            cols = df_new.columns
            colmaps_cat = {}
            colmaps_num = {}
            for colname in cols:
                colmap = serdiffm(df_ori[colname], df_new[colname])
                if colmap is None:
                    continue
                elif colmap.index.nlevels == 1:
                    colmaps_cat[colname] = colmap
                elif colmap.index.nlevels == 2:
                    colmaps_num[colname] = colmap

            # Set for categorical and numeric columns seperately.
            slogs = self.logs.setdefault(self.stage, {})
            if colmaps_cat:
                slogs[f"{type_}_{CAT_NAME}"] = pd.concat(
                    colmaps_cat.values(),
                    keys=colmaps_cat.keys())
            if colmaps_num:
                slogs[f"{type_}_{NUM_NAME}"] = pd.concat(
                    colmaps_num.values(),
                    keys=colmaps_num.keys())

            return ret
        return wrapper

    def log2excel(self, dest: str):
        xlwriter = pd.ExcelWriter(dest, mode="w")
        for stage, slogs in self.logs.items():
            for log_type, log in slogs.items():
                log.to_excel(xlwriter, f"{stage}_{log_type}")
        xlwriter.save()


# %%
# TODO: Annotations.
def read_dflog_file(log_fname: str, stage: str) -> dict[str, pd.DataFrame]:
    log_fname = "ka.xlsx"
    xlr = pd.ExcelFile(log_fname)
    logdfs = {}
    for sheet_name in xlr.sheet_names:
        stage, type_s = sheet_name.split("_")
        slog = logdfs.setdefault(stage, {})
        slog["_".join(type_s)] = pd.read_excel(xlr, sheet_name)
    return slog


# %%
# TODO: Annotation
# TODO: Is't this necessary with `pd.factorize`?
class SeriesEncoder():
    def __init__(self):
        pass

    def fit_transform(self, label: pd.Series):
        codes, unis = pd.factorize(label, sort=True)
        self.factors = unis
        self.invs_L = unis.insert(len(unis), None)
        self.trans_D = {v: k for k, v in enumerate(unis)}
        return codes

    def fit(self, label: pd.Series):
        self.fit_transform(label)
        return self

    def transform(self, label):
        return label.map(self.trans_D).fillna(-1).astype(int)

    def inverse(self, codes):
        return pd.Series(self.invs_L.take(codes), index=codes.index)


# %%
# TODO: Annotations.
def serlog(ser: pd.Series, label: pd.Series):
    """
        "t1_lift": lifts.max(),
        "t1_alift": acc_lifts.max(),
        "acc_corr": kcorr,
        "acc_pv": pv,
        "ACC_KEYS": ",".join([str(i) for i in acc_keys]),
        "pearson": pcorr,
        "kendall": kcorr,
        "spearman": scorr,
        "chi": chi,
        "chi_pv": pv,
        "IV": IV,
    """
    # 1. `contingency.crosstab` can only handle sortable array, namely NA and
    #   mixture dtype are not allowed.
    # 2. `pd.crosstab` will ignore NA.
    # 3. `contigency.crosstab` is 20 times faster than `pd.crosstab`.
    #
    # BUG: `contigency.crosstab` depends on `np.unique`, but `np.unqiue` may
    #   return duplicate elements when np.ndarray.dtype is objects and np.nan
    #   exists.
    # >>> a = np.array([1, np.nan, 1], dtype=object)
    # >>> np.unique(a)
    # [1, np.nan, 1]
    try:
        (ux, uy), ctab = contingency.crosstab(ser.values, label.values)
        uni_df = pd.DataFrame(ctab, index=ux, columns=uy)
    except TypeError as e:
        logger.warning(f"{e}. And an encoder will applied silently.")

        # sse = SeriesEncoder().fit(ser)
        # serv = sse.transform(ser)
        # lse = SeriesEncoder().fit(label)
        # labelv = lse.transform(label)

        codesx, fux = pd.factorize(ser, sort=True)
        fux = fux.insert(len(fux), pd.NA)
        codesy, fuy = pd.factorize(label, sort=True)
        fuy = fuy.insert(len(fuy), pd.NA)

        (ux, uy), ctab = contingency.crosstab(codesx, codesy)
        uni_df = pd.DataFrame(ctab, index=fux.take(ux), columns=fuy.take(uy))
    col_D = {}

    # Frequency.
    uni_df["freqr"] = ctab.sum(axis=1) / ctab.sum()

    # Chi-square stats.
    chi2, chi2_pv, chi2_dof, *ii = contingency.chi2_contingency(ctab)
    col_D["chi2"] = chi2
    col_D["chi2_pv"] = chi2_pv

    # Crosstab, woes, and lifts for binary label.
    if len(uy) == 2:
        woes, ivs = cal_woes_from_ctab(ctab)
        uni_df["woes"] = woes
        uni_df["ivs"] = ivs
        col_D["IV"] = ivs.sum()

        lifts, acc_lifts, lkcorr, lkpv = cal_lifts_from_ctab(ctab)
        uni_df["lifts"] = lifts
        if lkcorr > 0:
            uni_df["acc_lifts"] = acc_lifts
        else:
            uni_df["acc_lifts"] = acc_lifts[::-1]
        col_D["acl_kcorr"] = lkcorr
        col_D["acl_kpv"] = lkpv
        col_D["t1_lift"] = lifts.max()
        col_D["t1_acl"] = acc_lifts.max()

    # Correlation rate, outliers for numeric series.
    if (infer_dtype(ser) in ["integer", "floating"]
            and infer_dtype(label) in ["integer", "floating"]):
        for m in ["pearson", "kendall", "spearman"]:
            # Integers with None will be inferred as `integer`, but Error will
            # be raised when calculating correlations.
            col_D[m] = ser.astype(float).corr(label.astype(float), method=m)

    return uni_df, col_D


# %%
# TODO: Annotations.
def serdiffm(sero: pd.Series, sern: pd.Series,
             to_interval:bool = False) -> pd.Series:
    assert len(sero) == len(sern)
    # (uo, un), ctab = contingency.crosstab(sero.values, sern.values)

    # `np.unique` bases on sort which will:
    # 1. Cast data type to down to the same.
    # 2. Can't handle no-comparable value, `None` for example.
    #
    # `pd.unique` instead of `set` directly is used to filter repeated
    # `np.nan`.
    mapper = (pd.Series(sern.values, index=sero.values)
              .groupby(level=0, dropna=False, sort=True)
              .aggregate(lambda x:tuple(pd.unique(x))))

    mapper_1n = mapper.apply(len) > 1
    if mapper_1n.sum() > 0:
        logger.warning(f"Invalid 1-N mapping {mapper[mapper_1n]}.")
    else:
        dest = [ele[0] for ele in mapper]
        if np.all(mapper.index == dest):
            return None

    # Set intervals for numeric series.
    if to_interval and (infer_dtype(sero) in
                        ["integer", "floating", "mixed-integer-floating"]):
        range_mapper = {}
        ti = iter(mapper.items())
        start, (last, *ele) = next(ti)
        end = start
        for cur_edge, (cur, *ele) in ti:
            # Break when encountering `nan`.
            # ATTENTION: It's assumed that `nan` will always be the last item
            # in `mapper` from `ser.groupby`.
            if np.isnan(cur_edge):
                range_mapper[(start, end)] = last
                range_mapper[(cur_edge, cur_edge)] = cur
                break
            if last == cur:
                continue
            end = cur_edge

            range_mapper[(start, end - EPSILON)] = last
            start = end - EPSILON
            last = cur
        else:
            range_mapper[(start, end)] = last

        return (pd.Series(range_mapper)
                .rename_axis(["left[", "right)(EXCEPT_LAST)"])
                .rename("to"))
    else:
        return mapper.map(lambda x:x[0]).rename_axis("from").rename("to")


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

    @DFLoggerMixin.rclog(mark="Rename")
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
    @DFLoggerMixin.rclog(mark="DropDup")
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
                        keep[key] = CALLABLE_ENV[key]
                df = df.groupby(uni_keys, sort=False).aggregate(keep)

        # Update `label`.
        self.label = self.label[df.index]
        self.data = df

    @DFLoggerMixin.rclog(mark="DropNA")
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
        num_df = df.select_dtypes(include=["integer", "floating", "complex"])
        if num_df is not None and not num_df.empty:
            df.fillna({col: NUM_NA_FLAG for col in num_df.columns},
                      inplace=True)

        # Fill NaN for categorical columns.
        cat_df = df.select_dtypes(exclude=["integer", "floating", "complex"])
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
            cols = df.select_dtypes(
                exclude=["integer", "floating", "complex"]).columns

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

    @DFLoggerMixin.rclog(mark="DropPCorr")
    def drop_pcorr(self, thresh: float = 0.8):
        df = self.data
        label = self.label
        df_num = df.select_dtypes(include=["integer", "floating"])
        cols = df_num.columns

        pcorr = df_num.corr(method="pearson")
        ivs = cal_ivs(df_num.values, label.values)

        col_map = np.ones(len(cols), dtype=np.bool_)
        for col in cols:
            bmap = pcorr[col] >= thresh
            if bmap.sum() > 1:
                rivs = np.select([bmap, ], [ivs, ], 0)
                bmap[np.argmax(rivs)] = False
                col_map &= ~bmap

        self.data = df.loc[:, col_map]
