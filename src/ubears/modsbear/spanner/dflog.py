#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dlog.py
#   Author: xyy15926
#   Created: 2023-12-05 08:55:37
#   Updated: 2024-03-27 15:22:39
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
import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from scipy.stats import contingency

from ubears.ringbear.metrics import cal_lifts_from_ctab, cal_woes_from_ctab


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
    -----------------------------
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
          "RC": DF sketch down to column level,
          "MAP": DF of mapping of values,
          "VAL": DF sketch down to value level,
          "PCORR": DF of pearson correlation,
        }
      }
    """
    def __init__(self):
        self.data = None
        self.label = None
        self.stage = INIT_STATE
        self.logs = {}

    @staticmethod
    def rclog(ltag: str = None, type_: str = RC_NAME):
        """Decorator logging for row or column changes.

        1. The final log will be stored in the log-dict for the stage.
        2. The log will be a dataframe with Column[ltag] and
          Index[data.columns, ROWN_NAME], which will be like:
                    ltag
            Col_1
            Col_2
            ...
            ROWN
        3. Logs generated with the same `type_` will be concatenated along the
          row axis, namely the final log, `logs[stage][type_]`, will be:
                    ltag_1  ltag_2
            Col_1
            Col_2
            ...
            ROWN

        Params(keyword-only) added for the decorated:
        ----------------
        new_stage: New stage name.

        Params:
        ----------------
        ltag: Tag remarking the log
          `func.__name__` will be set as default if None is passed.
        type_: Log type.
          Log type indicates the content in the log, and log of the same type
            in the same stage will be concatenated.
        """
        def decorate(func: callable):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                # Update stage and record the original index and columns.
                new_stage = kwargs.pop("new_stage", None)
                if new_stage is not None:
                    self.stage = new_stage
                ori_cols = self.data.columns
                ori_rows = self.data.index

                ret = func(self, *args[1:], **kwargs)

                df = self.data
                nonlocal ltag
                if ltag is None:
                    ltag = func.__name__

                # Check and record the changes from index and columns.
                log = dict.fromkeys(set(ori_cols) - set(df.columns), 1)
                log.update(dict.fromkeys(set(df.columns) - set(ori_cols), -1))
                log[ROWN_NAME] = len(df.index) - len(ori_rows)
                log = pd.DataFrame({ltag: log})

                # Update the log with the `type_` of current stage.
                slogs = self.logs.setdefault(self.stage, {})
                if type_ in slogs:
                    slogs[type_] = pd.concat([slogs[type_], log], axis=1)
                else:
                    slogs[type_] = log

                pcorr = (self.data.select_dtypes(include=np.number)
                         .corr(method="pearson"))
                slogs[PCORR_NAME] = pcorr

                return ret
            return wrapper
        return decorate

    @staticmethod
    def valog(func: callable = None,
              rctype_: str = RC_NAME,
              vatype_: str = VA_NAME):
        """Decorator logging for description of values columns.

        1. This log includes both column-level and factor-level granularity
          for each column.
        2. For column-level granularity, stats such as Chi, P-value, IV will be
          calculated and updated to log dict, namely the log will be like:
                    Chi     P-value     IV
            Col_1
            Col_2
            ...
        3. For factor-level granularity, stats such as value-count, IV, WOE
          will be calculated for each factor.
                            Count   IV      WOE
            Col_1   Val_1
                    Val_2
                    ...
            Col_2   Val_1
            ...

        Params(keyword-only) added for the decorated:
        ----------------
        new_stage: New stage name.

        Params:
        ----------------
        rctype_: Log type for column-level granularity stats.
          Log type indicates the content in the log, and log of the same type
            in the same stage will be concatenated.
        vatype_: Log type for factor-level granularity stats.
          Log type indicates the content in the log, and log of the same type
            in the same stage will be concatenated.
        """
        if func is None:
            return partial(__class__.valog, rctype_=rctype_, vatype_=vatype_)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Update stage.
            new_stage = kwargs.pop("new_stage", None)
            if new_stage is not None:
                self.stage = new_stage

            ret = func(self, *args[1:], **kwargs)

            # Sketch for data down to both factor level and column level.
            # 1. Factor level: crosstabs, lifts, woes, ivs
            # 2. Column level: Chi2, IV
            df = self.data
            label = self.label
            uni_dfs = []
            col_Ds = []
            cols = df.columns

            # Log for each column.
            for colname in cols:
                ser = df[colname]
                uni_df, col_D = serlog(ser, label)
                uni_dfs.append(uni_df)
                col_Ds.append(col_D)

            # Update factor-level granularity log.
            log = pd.concat(uni_dfs, axis=0, keys=cols)
            slogs = self.logs.setdefault(self.stage, {})
            if vatype_ in slogs:
                slogs[vatype_] = pd.concat([slogs[vatype_], log], axis=1)
            else:
                slogs[vatype_] = log

            # Update column-level granularity log.
            log = pd.DataFrame(col_Ds, index=cols)
            slogs = self.logs.setdefault(self.stage, {})
            if rctype_ in slogs:
                slogs[rctype_] = pd.concat([slogs[rctype_], log], axis=1)
            else:
                slogs[rctype_] = log

            return ret
        return wrapper

    @staticmethod
    def malog(func: callable = None, type_: str = MA_NAME):
        """Decorator logging for mapping of values.

        1. `serdiffm` will be called to build the mapping relations between
          the data before and after processed by func.
        2. Mostly the mapping could be represented by a 1-to-1 mapper, which
          will be the format:
                    from_       to_
            Col_1   val_1       new_1
                    val_2       new_2
                    ...
            Col_2   ...
            ...
        3. But the mapping for numeric columns may be from interval to scalar,
          which will be stored seperately with the format:
                    start       end         to_
            Col_1   s1[         e1)        new_1
                    e1[         e2)        new_2
                    ...
                    e_n-1[      e_n]       new_n
            Col_2   ...
            ...

        Params(keyword-only) added for the decorated:
        ----------------
        new_stage: str
          New stage name.
        to_interval: bool
          If to build the mapping from interval to a scalar.

        Params:
        ------------------
        type_: Log type.
          Log type indicates the content in the log.
          Only the latest log of the same type in the same stage will be kept.
        """
        if func is None:
            return partial(__class__.malog, type_=type_)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Update stage and make a copy of the original data.
            to_interval = kwargs.pop("to_interval", False)
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
                colmap = serdiffm(df_ori[colname], df_new[colname],
                                  to_interval=to_interval)
                if colmap is None:
                    continue
                elif colmap.index.nlevels == 1:
                    colmaps_cat[colname] = colmap
                elif colmap.index.nlevels == 2:
                    colmaps_num[colname] = colmap

            # Set log for categorical and numeric columns seperately.
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

    def log2excel(self, dest: str) -> None:
        """Output logs to excel.

        Each sheet will store only one log type of one stage with stage and
        log type as its name.

        Params:
        ----------------------
        dest: File name.
        """
        xlwriter = pd.ExcelWriter(dest, mode="w")
        for stage, slogs in self.logs.items():
            for log_type, log in slogs.items():
                log.to_excel(xlwriter, f"{stage}_{log_type}")
        xlwriter.save()

    @staticmethod
    def read_dflog_file(filename: str) -> dict[str, pd.DataFrame]:
        """Read logs from excel.

        Params:
        ----------------------
        filename: File name.

        Return:
        -----------------------
        dict["stage_type", DataFrame of log]
        """
        xlr = pd.ExcelFile(filename)
        logdfs = {}
        for sheet_name in xlr.sheet_names:
            stage, type_s = sheet_name.split("_")
            slog = logdfs.setdefault(stage, {})
            slog["_".join(type_s)] = pd.read_excel(xlr, sheet_name)
        return slog


# %%
# TODO: Check flags.
def serlog(ser: pd.Series, label: pd.Series = None,
           with_woe: bool = True,
           with_lift: bool = True):
    """Decribe Series.

    Params:
    ---------------------
    ser: Series to be described.
    label: Label.
    with_woe: If to calculate WOEs and related etc.
    with_lift: If to calculate Lift and related etc.

    Return:
    ---------------------
    uni_df: DataFrame of factor-level granularity index.
            labels...     freqr       WOEs    IVs     Lifts     AccLifts
        f1
        f2
        ...
    col_D: Dict of column-level granularity index.
        chi_2: Chi-square.
        chi2_pv: P-value of Chi-square.
        IV: IV,
        t1_lift: Maximum lift.
        t1_acl: Maximum accumulating lift.
        acc_kcorr: Kendell correlations for lifts and values.
        acc_kpv: P-value of Kendell correlations.
        pearson: Pearson correltions for values and labels.
        kendall: Ditto.
        spearman: Ditto.
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
        if with_woe:
            woes, ivs = cal_woes_from_ctab(ctab)
            uni_df["woes"] = woes
            uni_df["ivs"] = ivs
            col_D["IV"] = ivs.sum()

        if with_lift:
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
def serdiffm(sero: pd.Series, sern: pd.Series,
             to_interval:bool = False) -> pd.Series:
    """Match, compare and build mapping between 2 series.

    1. 2 Series should share the same index.

    Params:
    --------------------
    sero: Source Series to be compared.
    sern: Destinated Series to be compared.
    to_interval: If to build the mapping from interval to a scalar.

    Return:
    --------------------
    Series representing the mapping from `sero` to `sern`.
    Series[factor from `sero`, factor from `sern`]
    Series[interval from `sero`, factor from `sern`]
    """
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

    # Check if one value will be map to multiple value.
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
