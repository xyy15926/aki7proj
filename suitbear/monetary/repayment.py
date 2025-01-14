#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: repayment.py
#   Author: xyy15926
#   Created: 2023-10-07 14:46:51
#   Updated: 2025-01-13 11:17:09
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

import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from flagbear.llp.parser import EnvParser
from modsbear.dflater.ex4df import trans_on_df, agg_on_df
from modsbear.dflater.exenv import EXGINE_ENV
from ringbear.timser.ovdd import ovdd_from_duepay_records, month_date

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

DUM_OVDD = 181
DUM_OVDP = 7


# %%
def addup_obovd(
    records: pd.DataFrame,
    ob_date: str = "monthend",
    dum_ovdd: int = DUM_OVDD,
    dum_ovdp: int = DUM_OVDP,
) -> pd.DataFrame:
    """Add MOB and overdue status for given observation dates.

    This should be applied only upon one account's repayment records. Use
    `groupby(oid).apply(addup_ob_records)` for records will multiple
    accounts.

    Params:
    ------------------------
    records: DataFrame[due_date, ovd_days, due_amt, rem_amt]
      due_date: Datetime64 convertable sequence representing the duepayment
        dates.
      ovd_days: Day past due for each duepay dates.
      due_amt: Duepay amount.
      rem_amt: Remaining amount.
      START: The start date of the account.
        The minimum of the due date will used as the default.
      MOB: The MOB of each records.
        The gap of month between the due date the start date will used as
        the default.
      DUMMY: The date when the account turns into being dummy.
    ob_date: Observation dates rules passed to `month_date` to generate the
      actually observations dates.
    dum_ovdd: The overdue days filled for the periods after being dummy.
    dum_ovdp: The overdue periods filled for the periods after being dummy.

    Return:
    ------------------------
    DataFrame with MOB, ever/stop overdue days/periods and duepay/remaining
      amounts.
    """
    # Sort by due date.
    recs = records.sort_values("due_date")
    due_date = recs["due_date"]
    # Add MOB.
    if "MOB" not in recs:
        if "START" in recs:
            start_date = recs["START"].dt.to_period("M")
        else:
            start_date = due_date.min().to_period("M")
        mobs = (due_date.dt.to_period("M") - start_date).apply(lambda x:x.n)
        recs["MOB"] = mobs
    else:
        mobs = recs["MOB"]
    # Add observer records.
    obd = month_date(due_date, ob_date)
    (ever_ovdd, stop_ovdd, ever_ovdp, stop_ovdp, ever_ovda, stop_ovda,
     ever_rema, stop_rema, ever_duea, stop_duea) = ovdd_from_duepay_records(
         recs["due_date"], recs["ovd_days"], obd,
         recs["due_amt"], recs["rem_amt"])
    recs["ob_date"] = obd
    # Add dummy month.
    if "DUMMY" in records:
        dum_date = recs["DUMMY"]
        ever_ovdd[due_date >= dum_date] = dum_ovdd
        stop_ovdd[due_date >= dum_date] = dum_ovdd
        ever_ovdp[due_date >= dum_date] = dum_ovdp
        stop_ovdp[due_date >= dum_date] = dum_ovdp

    recs["ever_ovdd"] = ever_ovdd
    recs["stop_ovdd"] = stop_ovdd
    recs["ever_ovdp"] = ever_ovdp
    recs["stop_ovdp"] = stop_ovdp
    recs["ever_ovda"] = ever_ovda
    recs["stop_ovda"] = stop_ovda
    recs["ever_duea"] = ever_duea
    recs["stop_duea"] = stop_duea
    recs["ever_rema"] = ever_rema
    recs["stop_rema"] = stop_rema

    # Convert dtype to categorical so to keep null rows and columns in crosstab.
    # cats = pd.CategoricalDtype([f"M{i}" for i in range(8)])
    # recs["ever_ovdp"] = recs["ever_ovdp"].apply(lambda x: f"M{x}").astype(cats)
    # recs["stop_ovdp"] = recs["stop_ovdp"].apply(lambda x: f"M{x}").astype(cats)

    return recs


# %%
def addup_obrec(
    records: pd.DataFrame,
    ob_date: list | np.ndarray | str | int = "monthend",
    start_date: pd.Series = None,
    dum_date: pd.Series = None,
    dum_ovdd: int = DUM_OVDD,
    dum_ovdp: int = DUM_OVDP,
) -> pd.DataFrame:
    """Add MOB, overdue status for given observation dates.

    Records of different account will be grouped up by `oid` and then
    observation records will be added up for them seperately.

    Params:
    ------------------------
    records: DataFrame[oid, due_date, ovd_days, due_amt, rem_amt]
      due_date: Datetime64 convertable sequence representing the duepayment
        dates.
      ovd_days: Day past due for each duepay dates.
      due_amt: Duepay amount.
      rem_amt: Remaining amount.
    ob_date: Observation dates rules passed to `month_date` to generate the
      actually observations dates.
    start_date: Start date of the account.
      The first duepay dates will be used as the start date if not provided.
    dum_date: Date from when the account is treated as the dummy account.
    dum_ovdd: The overdue days filled for the periods after being dummy.
    dum_ovdp: The overdue periods filled for the periods after being dummy.

    Return:
    ------------------------
    DataFrame with MOB, ever/stop overdue days/periods and duepay/remaining
      amounts.
    """
    # Join start date and dummy date if necessary.
    if start_date is not None:
        start_date = start_date.rename("START")
        start_date.index.name = "oid"
        records = records.join(start_date, on="oid")
    if dum_date is not None:
        dum_date = dum_date.rename("START")
        dum_date.index.name = "oid"
        records = records.join(dum_date, on="oid")

    tqdm.pandas(desc="Add Obrec")
    ret = records.groupby("oid").progress_apply(
        addup_obovd, ob_date=ob_date, dum_ovdd=dum_ovdd, dum_ovdp=dum_ovdp)

    return ret


# %%
def edge_crosstab(
    recs: pd.DataFrame,
    start: pd.Timestamp | int | str,
    end: pd.Timestamp | int | str = None,
    factor: str = "ever_ovdp",
    normalize: bool = True,
    *,
    values: str = None,
    aggfunc: str | Callable = None,
) -> dict[str, pd.DataFrame]:
    """Calculate crosstab from observation records.

    1. Start-timetag and end-timetag are necessary to calculate the crosstab
      matrix. If no `end` provided, the next MOB or `ob_date` will be used
      as default to calculate the rolling or migration precisely.
    2. Records that don't overpass the whole given performance period will
      be filtered out.
    3. Rolling rates of both `ever_ovdp` and `stop_ovdp` will be returned.

    Params:
    ---------------------------
    recs: DataFrame[oid, ob_date or MOB, ...]
    start: The time tag for the start of the performance period.
      str | Timestamp: Use the observation dates as the time tag.
        Both the observation dates and the start tag will be casted into
          "M8[M]" to the determining the records.
      int: Use the MOB as the time tag.
    end: The time tag for the end of the performance period.
      The next MOB or `ob_date` of the `start` will be used as default.
    factor: The field for crosstab.
      The values of factor will be the columns and rows of the ctab.
    normalize: Normalize along the index.
    values: Values to aggregate according to the factor.
    aggfunc: Aggregation function for the values.

    Return:
    ---------------------------
    Crosstab matrix with `ever_ovdp` and simple record count as example:
      START   M1    M2    ...     M7
      END
      M1
      M2
      ...
      M7
    """
    # Select begin and end records for roll rate.
    if isinstance(start, int):
        if end is None:
            end = start + 1
        end_recs = recs[recs["MOB"] == end]
        start_recs = recs[recs["MOB"] == start]
    else:
        start = np.datetime64(start, "M")
        ob_mon = np.asarray(recs["ob_date"], "M8[M]")
        if end is None:
            end = start + np.timedelta64(1, "M")
        else:
            end = np.datetime64(end, "M")
        end_recs = recs[ob_mon == end]
        start_recs = recs[ob_mon == start]

    # Exclude the records that don't cover the whole performance period.
    recs = pd.merge(start_recs, end_recs, on="oid")
    if len(recs) != len(start_recs) or len(recs) != len(end_recs):
        logger.info(f"Only {len(recs)} records are selected from "
                    f"{len(start_recs)} start-records and "
                    f"{len(end_recs)} end-records for calculation.")

    start_ovdp = recs[f"{factor}_x"]
    end_ovdp = recs[f"{factor}_y"]
    # Use the values of the start-edge.
    if values is not None:
        values = recs[f"{values}_x"]
    ctab = pd.crosstab(start_ovdp, end_ovdp,
                       rownames=["START"], colnames=["END"],
                       values=values, aggfunc=aggfunc,
                       margins=True, margins_name="SUM",
                       dropna=False).astype(float)
    if normalize:
        ctab.values[:, :-1] /= ctab.values[:, -1:]

    return ctab


# %%
def mob_align(
    recs: pd.DataFrame,
    rectag: pd.Series = None,
    cond: str = "ever_ovdp >= 1",
    agg: str = "count(_)",
    *,
    agg_rules: list[tuple] = None,
    trans_rules: list[tuple] = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Pivot the observation records to align the MOB.

    1. `rectag` and MOB in `rec` will be used as group key to group the whole
      `recs`. And `cond`, `agg` or `rules` will passed to EnvParser to apply
      aggregation on the groups so to align the MOBs.
    2. Only single DataFrame will be returned if `rules` is None, else a dict
      of keys in `rules` and DataFrame of the aggregation result will be
      returned.


    Params:
    ---------------------------
    recs: DataFrame[oid, due_date, MOB, ...]
    rectag: Series[oid, tag]
      Mapping `oid` to some group tags for following caluclations.
      Vintag, the month of the `due_date`, will be used as default if not
        provided.
    cond: Single condition for `agg_on_df` to apply on MOB and rectag group.
    agg：Single aggregation for `agg_on_df`.
    agg_rules: A list of rules for `agg_on_df`.
      NOTE: `cond` and `agg` will be ignored if valid rules passed.
    trans_rules: A list of rules for `trans_of_df`.

    Return:
    ---------------------------
    None-rules: DataFrame with Index[oid] and Column[MOB].
    Valid rules: Dict of DataFrame with keys from rules.
    """
    # Construct vintage tags with the first duepayment dates if record tag
    # not provided.
    if rectag is None:
        rectag = (recs.groupby("oid")["due_date"].agg("min")
                  .astype("M8[M]").rename("rectag"))
    rectag.name = "rectag"
    mob_recs = recs.join(rectag, on="oid")

    # Construct aggregation rules.
    __NONE__ = "__NONE__"
    if agg_rules is None:
        agg_rules = [(__NONE__, cond, agg)]
    envp = EnvParser(EXGINE_ENV)
    tqdm.pandas(desc="MOB Alignment")
    agg_ret = mob_recs.groupby(["rectag", "MOB"]).progress_apply(
        agg_on_df, rules=agg_rules, envp=envp)
    if trans_rules:
        agg_ret = trans_on_df(agg_ret, trans_rules, how="inplace", envp=envp)

    # Unstack aggregation result to align MOB and rectag.
    if agg_ret.shape[1] == 1 and agg_ret.columns[0] == __NONE__:
        ret = agg_ret[__NONE__].unstack(1)
    else:
        ret = {}
        for col in agg_ret.columns:
            ret[col] = agg_ret[col].unstack(1)

    return ret
