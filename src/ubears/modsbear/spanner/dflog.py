#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dlog.py
#   Author: xyy15926
#   Created: 2023-12-05 08:55:37
#   Updated: 2025-02-15 18:31:33
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, TypeVar, TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path
from collections.abc import Callable, Iterator

from collections import Counter
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import contingency

from ubears.flagbear.slp.pdsl import save_with_excel
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
def serdesc(
    ser: pd.Series | np.ndarray,
    label: pd.Series | np.ndarray = None,
    with_woe: bool = True,
    with_lift: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Describe the sequences.

    Description of factors and the whole sequences will be returned
    seperately.

    Params:
    ---------------------
    seq: Series to be described.
    label: Label.
    with_woe: If to calculate WOEs and related etc.
    with_lift: If to calculate Lift and related etc.

    Return:
    ---------------------
    factor_log: DataFrame of factor-level granularity index.
          labels_Cnt    FreqR   WOEs    IVs     Lifts     AccLifts  AccLiftsR
      F1
      F2
      ...
    seq_log: Series of column-level granularity index.
      Chi2: Chi-square.
      Chi2PV: P-value of Chi-square.
      IV: IV,
      T1Lift: Maximum lift.
      T1Acc: Maximum accumulating lift.
      T1AccR: Maximum accumulating lift in reversed order.
      AccKenCorr: Kendell correlations for lifts and values.
      AccKenPV: P-value of Kendell correlations.
      Pearson: Pearson correltions for values and labels.
    """
    if label is None:
        larr = np.ones(len(ser), dtype=np.int32)
    else:
        larr = np.asarray(label)
    seq = np.asarray(ser)

    seq_log = {}
    # Factorize if not numeric so to `contingency.crosstab`.
    # As `pd.factorize` will always encode `pd.isna` with -1, so insert
    # `np.nan` to the end of `fux` and `fuy` will be compatiable to recover
    # the orginal values back in construct DataFrame of crosstab.
    # 1. No matter `pd.isna` exists or not.
    # 2. All `pd.isna` will be treated as the same `np.nan`, including `None`,
    #   `pd.NA` and etc.
    fux, fuy = None, None
    if not is_numeric_dtype(seq):
        seq, fux = pd.factorize(seq, sort=True)
        fux = np.concatenate([fux, [np.nan,]])
    if not is_numeric_dtype(larr):
        larr, fuy = pd.factorize(larr, sort=True)
        fuy = np.concatenate([fuy, [np.nan,]])

    (ux, uy), ctab = contingency.crosstab(seq, larr)
    factor_log = pd.DataFrame(ctab,
                              index=fux.take(ux) if fux is not None else ux,
                              columns=fuy.take(uy) if fuy is not None else uy)
    # Frequency.
    factor_log["FreqR"] = ctab.sum(axis=1) / ctab.sum()

    # Chi-square stats.
    if len(uy) > 1:
        chi2, chi2_pv, chi2_dof, *ii = contingency.chi2_contingency(ctab)
        seq_log["Chi2"] = chi2
        seq_log["Chi2PV"] = chi2_pv

    # Crosstab, woes, and lifts for binary label.
    if len(uy) == 2 and with_woe:
        woes, ivs = cal_woes_from_ctab(ctab)
        factor_log["WOEs"] = woes
        factor_log["IVs"] = ivs
        seq_log["IV"] = ivs.sum()
    if len(uy) == 2 and with_lift:
        lifts, acc_lifts, acc_lifts_r, lkcorr, lkpv = cal_lifts_from_ctab(ctab)
        factor_log["Lifts"] = lifts
        factor_log["AccLifts"] = acc_lifts
        factor_log["AccLiftsR"] = acc_lifts_r
        seq_log["AccKenCorr"] = lkcorr
        seq_log["AccKenPV"] = lkpv
        seq_log["T1Lift"] = lifts.max()
        seq_log["T1Acc"] = acc_lifts.max()
        seq_log["T1RAcc"] = acc_lifts_r.max()

    # Correlation rate, outliers for numeric series.
    if is_numeric_dtype(ser) and is_numeric_dtype(label):
        seq_log["Pearson"] = np.corrcoef(seq, larr)[0, 1]

    return factor_log, pd.Series(seq_log, dtype=np.float64)


# %%
def serdiffm(
    sero: np.ndarray | pd.Series,
    sern: np.ndarray | pd.Series,
    to_interval: bool = False,
) -> pd.Series:
    """Match, compare and build mapping between 2 sequences.

    Series with 2-level index will be returned if `to_interval` is set and
    `sero` is numeric.
    And the cutting edges, as the values in index, will the middle point of
    the changine points in `sero`.

    Params:
    --------------------
    sero: Source Series to be compared.
    sern: Destinated Series to be compared.
    to_interval: If to build the mapping from interval to a scalar.

    Return:
    --------------------
    Series representing the mapping from `sero` to `sern` with index:
      1. `FROM`: `to_interval` unset.
      2. `LEFT, `RIGHT`: `to_interval` set.
    """
    if isinstance(sero, pd.Series) and isinstance(sern, pd.Series):
        dt = pd.concat([sero, sern], axis=1)
        sero, sern = dt.values[:, 0], dt.values[:, 1]
    elif len(sero) != len(sern):
        logger.warning("Diffmap can't be applied on no-Series sequences of"
                       " different length.")
        return None

    # Construct mapper from `sero` to `sern`.
    mapper = {}
    changed = False
    for vo, vn in zip(sero, sern):
        # Unify all `nan` to `np.nan`, as `float("nan")` are not equal.
        # ATTENTION: `np.isnan` only accept numeric value.
        if pd.isna(vo):
            vo = np.nan
            if not pd.isna(vn):
                changed = True
        elif vo != vn:
            changed = True
        vr = mapper.setdefault(vo, vn)
        if vr != vn and vr is not vn:
            logger.warning(f"The value {vo} has been mapped to multiple values"
                           f": {vn}, {vr}.")
    if not changed:
        return None

    # Set intervals for numeric series.
    if to_interval and is_numeric_dtype(sero):
        mapper = pd.Series(mapper).sort_index()
        range_mapper = {}
        ti = iter(mapper.items())
        left, last = next(ti)
        right = left
        for cur_edge, cur in ti:
            # It's assumed that `nan` will always be the last item in
            # `pd.Series.sort_index`.
            # So break and set for `np.nan` when encountering `np.nan`.
            if np.isnan(cur_edge):
                range_mapper[(left, right)] = last
                range_mapper[(cur_edge, cur_edge)] = cur
                break
            # Continue for continuous mapping result.
            if last == cur:
                right = cur_edge
                continue

            mid = (right + cur_edge) / 2
            range_mapper[(left, mid)] = last
            left, right = mid, cur_edge
            last = cur
        else:
            range_mapper[(left, right)] = last

        ret = (pd.Series(range_mapper)
               .rename_axis(["LEFT", "RIGHT"])
               .rename("TO"))
        return ret

    ret = pd.Series(mapper, name="TO").rename_axis("FROM")
    return ret


# %%
def dfdesc(
    data: pd.DataFrame,
    label: pd.Series | np.ndarray = None,
    with_woe: bool = True,
    with_lift: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Describe the DataFrame.

    Description of both column-level and factor-level granularity will
    be calculated and returned seperately.

    Params:
    ---------------------
    data: DataFrame to be described.
    label: Label.
    with_woe: If to calculate WOEs and related etc.
    with_lift: If to calculate Lift and related etc.

    Return:
    ---------------------
    factor_df: DataFrame of factor-level granularity description.
                labels_Cnt  FreqR   WOEs    IVs     Lifts     AccLifts  AccLiftsR
      Col1  F1
            F2
            ...
      Col2  ...
    col_log: DataFrame of column-level granularity description.
            Chi2  Chi2PV  IV  T1Lift  T1Acc   T1AccR  AccKenCorr  AccKenPV    Pearson
      Col1
      Col2
      ...
    """
    factor_logs, ser_logs = [], []
    # Log for each column.
    for coln in data.columns:
        fl, sl = serdesc(data[coln], label)
        factor_logs.append(fl)
        ser_logs.append(sl)

    # Construct DataFrame of log.
    fdf = pd.concat(factor_logs, axis=0, keys=data.columns)
    sdf = pd.DataFrame.from_records(ser_logs, index=data.columns)

    return fdf, sdf


# %%
def dfdiffm(
    dfo: np.ndarray | pd.Series,
    dfn: np.ndarray | pd.Series,
    to_interval: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Match, compare and build mapping between columns from 2 DataFrame.

    1. Series with 2-level index will be returned if `to_interval` is set and
      `sero` is numeric.
      1.1 And the cutting edges, as the values in index, will the middle point of
      the changine points in `sero`.
      1.2 `num_df` returned will always be empty if `to_interval` not set.
    2. Columns with no change and columns that don't exist in both DataFrame
      will be ignored.

    Params:
    --------------------
    dfo: Source DataFrame to be compared.
    dfn: Destinated DataFrame to be compared.
    to_interval: If to build the mapping from interval to a scalar.

    Return:
    --------------------
    Series representing the mapping from `sero` to `sern` with index:
      `cat_df`: 1-to-1 mapper
                FROM
         Col1   Val1    New1
                Val2    New2
                ...
         Col2   ...
         ...
      `num_df`: interval-to-1 mapper for numeric with `to_interval` set
                LEFT        RIGHT
         Col_1  MIN[        E1)         New1
                E1[         E2)         New2
                ...
                En[         MAX]        NewN
         Col_2  ...
         ...
    """
    # Build the mappings DataFrame.
    cat_map = {}
    num_map = {}
    for colname in dfn.columns:
        if colname not in dfo.columns:
            logger.info(f"Column {colname} doesn't exists in original"
                        f" DataFrame.")
            continue
        colmap = serdiffm(dfo[colname], dfn[colname],
                          to_interval=to_interval)
        if colmap is None:
            continue
        elif colmap.index.nlevels == 1:
            cat_map[colname] = colmap
        elif colmap.index.nlevels == 2:
            num_map[colname] = colmap

    cat_df = (pd.Series(dtype=object, name="TO") if len(cat_map) == 0
              else pd.concat(cat_map.values(), keys=cat_map.keys()))
    num_df = (pd.Series(dtype=object, name="TO") if len(num_map) == 0
              else pd.concat(num_map.values(), keys=num_map.keys()))

    return cat_df, num_df


# %%
class ProcessLogger:
    """Logger for DataFrame processing.

    1. DataFrames will be used to record the information.

    Attrs:
    -----------------------------
    proc_logs: Dict[str, DataFrame]
      Dict storing DataFrame recording the information.
    log_cnt: Counter
      Counter dict records the number of each type of the logs.
    """
    def __init__(self):
        self.proc_logs = {}
        self.log_cnt = Counter()

    def maplog(self, dfo: pd.DataFrame,
               dfn: pd.DataFrame,
               to_interval: bool = True,
               ltag: str = "TRANS") -> None:
        """Match, compare and build mapping between columns from 2 DataFrame.

        Series representing the mapping from `sero` to `sern` with index:
          1. `cat_df`: 1-to-1 mapper
                    FROM
             Col1   Val1    New1
                    Val2    New2
                    ...
             Col2   ...
             ...
          2. `num_df`: interval-to-1 mapper for numeric with `to_interval` set
                    LEFT        RIGHT
             Col1   MIN[        E1)         New1
                    E1[         E2)         New2
                    ...
                    En[         MAX]        NewN
             Col2   ...
             ...
        will be stored in `proc_logs` with key constructed from `ltag`
        seperately.

        Params:
        --------------------
        dfo: Source DataFrame to be compared.
        dfn: Destinated DataFrame to be compared.
        to_interval: If to build the mapping from interval to a scalar.
        ltag: The key of the log Series in `proc_logs`.

        Return:
        --------------------
        None
        """
        proc_logs = self.proc_logs
        log_cnt = self.log_cnt
        log_cnt[ltag] += 1
        ltag_cnt = log_cnt[ltag]
        cat_df, num_df = dfdiffm(dfo, dfn, to_interval)
        if not cat_df.empty:
            proc_logs[f"catmap_{ltag}_{ltag_cnt}"] = cat_df
        if not num_df.empty:
            proc_logs[f"nummap_{ltag}_{ltag_cnt}"] = num_df

    def vallog(self, data: pd.DataFrame,
               label: pd.Series | np.ndarray,
               with_woe: bool = True,
               with_lift: bool = True,
               ltag: str = "ABST"):
        """Describe the DataFrame.

        DataFrame of description:
          1. factor_df: DataFrame of factor-level granularity description.
                      labels_Cnt  FreqR   WOEs    IVs     Lifts...
            Col1  F1
                  F2
                  ...
            Col2  ...
          2. col_log: DataFrame of column-level granularity description.
                  Chi2  Chi2PV  IV  T1Lift  T1Acc   T1AccR  AccKenCorr...
            Col1
            Col2
            ...
          3. Pearson corelations matrix.
        will be stored in `proc_logs` with key constructed from `ltag`
        seperately.

        Params:
        ---------------------
        data: DataFrame to be described.
        label: Label.
        with_woe: If to calculate WOEs and related etc.
        with_lift: If to calculate Lift and related etc.
        ltag: The key of the log Series in `proc_logs`.

        Return:
        --------------------
        None
        """
        proc_logs = self.proc_logs
        log_cnt = self.log_cnt
        log_cnt[ltag] += 1
        ltag_cnt = log_cnt[ltag]
        fdesc, cdesc = dfdesc(data, label, True, True)
        if not fdesc.empty:
            proc_logs[f"fdesc_{ltag}_{ltag_cnt}"] = fdesc
        if not cdesc.empty:
            proc_logs[f"cdesc_{ltag}_{ltag_cnt}"] = cdesc

        # Pearson correlation.
        pcorr = data.select_dtypes(include=np.number).corr(method="pearson")
        if not pcorr.empty:
            proc_logs[f"pcorr_{ltag}_{ltag_cnt}"] = pcorr

    def log2excel(self, fname: str = "proclog/proc_log.xlsx") -> Path:
        """Save DataFrame of log as excel."""
        return save_with_excel(self.proc_logs, fname)
