#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: ovdd.py
#   Author: xyy15926
#   Created: 2024-03-12 11:02:29
#   Updated: 2025-05-14 15:09:39
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
from typing import Any, TypeVar
from collections import deque
from collections.abc import Callable, Iterator, Sequence, Container
# from IPython.core.debugger import set_trace
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self

import numpy as np
import pandas as pd

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
def snap_ovd(
    due_date: list | np.ndarray,
    rep_date: list | np.ndarray = None,
    ovd_days: list | np.ndarray = None,
    ob_date: list | np.ndarray = None,
    due_amt: list | np.ndarray = None,
    rem_amt: list | np.ndarray = None,
) -> tuple[np.ndarray]:
    """Calculate the snapshot from the repayment cashflow.

    1. Use two queues, that differs in if to include the edge records, to
      store the continuous overdue records.
      1.1 In the outer loop for obdates, the queues retain the records
        overpass the former obdates.
      1.2 In the inner loop for duedates, the queues retain the continuous
        overdue records.
    2. Be careful with following corner cases:
      2.1 duedate == obdate or duedate == repdate: not overdue
    3. `rep_date` and `ovd_days` should always satisfy:
      `rep_date = due_date + ovd_days`.
      So anyone of `rep_date` and `ovd_days` passed will be fine, and
      `rep_date` will be used if both passed.
    4. `due_amt` and `rem_amt` could be 2D-NDA with the the same shape, of
      which the correspondant columns represent one pair of process-volume and
      status-volume. And `XXX_rema`, `XXX_ovda`, `XXX_duea` will spawned in
      `ovd-amount` as the result.

    Assumption:
    ----------------------
    1. At most 1 record for one day.
    2. The repayment dates must be in ascending order along with the duedates.
    3. Gaps between duepay dates are equal and gasp between obdates are
      equal, so the first period could be used directly and no more check
      need to be done for continuous overdue periods.

    Params:
    ----------------------
    due_date: Sequence[N] of duepayment dates, which should be datetime64 or
      string that could be casted into datetime64 by NumPy.
    rep_date: Sequence[N] of repayment dates.
      This will calculated from the `due_date` and `ovd_days` if not passed.
    ovd_days: Sequence[N] of overdue days of each repayments.
    ob_date: Sequence[M] of observation dates.
      If no argument passed, this will be the `due_date` after shifting
      out the first duepay date and including a faraway date.
    due_amt: Sequence of duepay amount.
      None: All zeros will be used.
    rem_amt: Sequence of remaining amount after the duepay amount.
      None: All zeros will be used.

    Return:
    ----------------------
    ovd-time: NDArray[N, 4]
      ever_ovdd: NDArray of maximum of overdue days ever occured during two
        responsible point of observation.
      ever_ovdp: Maximum of overdue periods ever.
        NOTE: The number periods counted here will be more precise as the days
        of each of month is not the same.
      stop_ovdd: NDArray of overdue days at the point of observation.
      stop_ovdp: Ditto.
    ovd-amount: NDArray[N, 6 * M], M is the number of columns of `due_amt`.
      ever_rema * M: Maximum (or the first) remainal amount ever.
      ever_ovda * M: Maximum (or the first) overdue amount ever.
      ever_duea * M: Maximum (or the first) duepay amount ever.
      stop_rema * M: Ditto.
      stop_ovda * M: Ditto.
      stop_duea * M: Ditto.
    MOB: The index of the last period before the oberservation.
      -1: No period before the observation.
    stop_recs: List of the records that affect the observation point.
      Record: Tuple[duei, dued, repd, ovdd, duea, rema].
    """
    # Prepare the data.
    dueds = np.asarray(due_date, dtype="M8[D]")
    if ob_date is None:
        obds = np.concatenate(
            [dueds[1:], np.array(["2999-12-31"], dtype="M8[D]")])
    else:
        obds = np.asarray(ob_date, dtype="M8[D]")
    if rep_date is None:
        ovdds = np.asarray(ovd_days, dtype="m8[D]")
        ovdds[np.isnat(ovdds)] = np.timedelta64(0, "D")
        repds = dueds + ovdds
    else:
        repds = np.asarray(rep_date, dtype="M8[D]")
        ovdds = repds - dueds

    # Align `due_amt` and `rem_amt`.
    if due_amt is None and rem_amt is None:
        das = np.zeros(ovdds.shape[0], 1)
        ras = np.zeros(ovdds.shape[0], 1)
    elif due_amt is None:
        ras = np.asarray(rem_amt)
        das = np.zeros_like(ras)
    elif rem_amt is None:
        das = np.asarray(due_amt)
        ras = np.zeros_like(das)
    else:
        das = np.asarray(due_amt)
        ras = np.asarray(rem_amt)
    if len(ras.shape) == 1:
        ras = ras.reshape(-1, 1)
    if len(das.shape) == 1:
        das = das.reshape(-1, 1)
    amt_n = das.shape[1]

    mob, ovdt, ovda = [], [], []
    stop_recs = []
    duei = 0
    # Actually, `sconti_recs` ares just used to store the records with
    # `rep-date(former) == due-date == ob-date`, so to calculate the STOPs.
    # While `conti_recs` will drop those records except the last one, since
    # they can't be continuous with the following records.
    conti_recs = deque()            # Excludes the edge records.
    sconti_recs = deque()           # Includes the edge records.
    for obd in obds:
        # set_trace()
        # Set initial values.
        if len(conti_recs) == 0:
            if duei < len(dueds):
                ever_rema = das[duei] + ras[duei]
            else:
                # No more valid records, so the last `rem_amt` will be kept
                # unchanged.
                # So is the `stop_rema`.
                ever_rema = ras[-1]
        else:
            ever_rema = conti_recs[0][-2] + conti_recs[0][-1]

        ever_ovdd, ever_ovdp = 0, 0
        ever_ovda, ever_duea = np.zeros(amt_n), np.zeros(amt_n)
        # Traverse all the duepayments before or on the obdate to get the EVERs.
        while duei < len(dueds) and dueds[duei] <= obd:
            dued, repd, ovdd = dueds[duei], repds[duei], ovdds[duei]
            duea, rema = das[duei], ras[duei]
            if len(conti_recs) == 0:
                conti_recs.append((duei, dued, repd, ovdd, duea, rema))
                sconti_recs.append((duei, dued, repd, ovdd, duea, rema))
            else:
                hduei, hdued, hrepd, hovdd, hduea, hrema = conti_recs[0]
                tduei, tdued, trepd, tovdd, tduea, trema = conti_recs[-1]
                if hrepd < dued:
                    if tdued == trepd:
                        ever_ovdd = max(hovdd, ever_ovdd)
                        ever_ovdp = max(len(conti_recs) - 1, ever_ovdp)
                        va = np.sum([i[-2] for i in conti_recs], axis=0)
                        ever_ovda = np.max([va - tduea, ever_ovda], axis=0)
                        ever_duea = np.max([va, ever_duea], axis=0)
                    else:
                        ever_ovdd = max(hovdd, ever_ovdd)
                        ever_ovdp = max(len(conti_recs), ever_ovdp)
                        va = np.sum([i[-2] for i in conti_recs], axis=0)
                        ever_ovda = np.max([va, ever_ovda], axis=0)
                        ever_duea = np.max([va, ever_duea], axis=0)
                # As `trepd >= hrepd == dued > tdued`,
                # Don't need to compared `trepd` and `tdued`.
                elif hrepd == dued:
                    ever_ovdd = max(hovdd, ever_ovdd)
                    ever_ovdp = max(len(conti_recs), ever_ovdp)
                    va = np.sum([i[-2] for i in conti_recs], axis=0)
                    ever_ovda = np.max([va, ever_ovda], axis=0)
                    ever_duea = np.max([va + duea, ever_duea], axis=0)
                # Pop uncontinuous overdue periods out.
                while len(conti_recs) > 0 and conti_recs[0][2] <= dued:
                    conti_recs.popleft()
                while len(sconti_recs) > 0 and sconti_recs[0][2] < dued:
                    sconti_recs.popleft()
                conti_recs.append((duei, dued, repd, ovdd, duea, rema))
                sconti_recs.append((duei, dued, repd, ovdd, duea, rema))
            duei += 1

        # TODO
        # 1. Adjacent obdates with no records cutting in, namely no duedate
        #   lies between two obdates and no repdate overpass the former obdate.
        # 2. Or observe before all records.
        # 3. Or observe after all records and no repdate overpass the former
        #   obdate.
        # set_trace()
        if len(conti_recs) == 0:
            if duei < len(dueds):
                if len(ovdt) == 0:
                    logger.warning("Observe before all records.")
                else:
                    logger.warning("Adjacent obdates with no records cutting in.")
            ever_ovdd, ever_ovdp = 0, 0
            ever_ovda, ever_duea = np.zeros(amt_n), np.zeros(amt_n)
        else:
            # Check the last continuous overdued periods to update EVERs.
            hduei, hdued, hrepd, hovdd, hduea, hrema = conti_recs[0]
            tduei, tdued, trepd, tovdd, tduea, trema = conti_recs[-1]
            # TODO: No-equal gap between obdates.
            ever_ovdd = max(min(hrepd, obd) - hdued, ever_ovdd)
            if tdued == trepd or tdued == obd:
                ever_ovdp = max(len(conti_recs) - 1, ever_ovdp)
                va = np.sum([i[-2] for i in conti_recs], axis=0)
                ever_ovda = np.max([va - tduea, ever_ovda], axis=0)
                ever_duea = np.max([va, ever_duea], axis=0)
            else:
                ever_ovdp = max(len(conti_recs), ever_ovdp)
                va = np.sum([i[-2] for i in conti_recs], axis=0)
                ever_ovda = np.max([va, ever_ovda], axis=0)
                ever_duea = np.max([va, ever_duea], axis=0)

            # Pop out records repayed before obdate.
            # TODO: No-equal gap may be solved here.
            while len(conti_recs) > 0 and conti_recs[0][2] <= obd:
                hduei, hdued, hrepd, hovdd, hduea, hrema = conti_recs.popleft()
                ever_ovdd = max(hovdd, ever_ovdd)

        if len(sconti_recs) == 0:
            # As no post-effect from the fromer recoreds, just consider the
            # very next record will be fine.
            if duei < len(dueds):
                stop_rema = das[duei] + ras[duei]
            else:
                stop_rema = ras[-1]
            stop_ovdd, stop_ovdp = 0, 0
            stop_ovda, stop_duea = np.zeros(amt_n), np.zeros(amt_n)
        else:
            # Try init `stop_rema` with latest `rem_amt` first.
            # If no the continuous overdue record achieve the obdate, this
            #   will be kept unchanged.
            # Else continuous overdue records will be used to update.
            stop_rema = sconti_recs[-1][-1]
            while len(sconti_recs) > 0 and sconti_recs[0][2] < obd:
                sconti_recs.popleft()

            # Check current records in queue to get the STOPs.
            if len(sconti_recs) == 0:
                stop_ovdd, stop_ovdp = 0, 0
                stop_ovda, stop_duea = np.zeros(amt_n), np.zeros(amt_n)
            else:
                stop_ovdd = obd - sconti_recs[0][1]
                stop_rema = sconti_recs[0][-2] + sconti_recs[0][-1]
                va = np.sum([i[-2] for i in sconti_recs], axis=0)
                # ATTENTION:
                # Copy is required or `stop_duea` will be exactly the `stop_ovda`.
                stop_duea = va.copy()
                stop_ovda = va.copy()
                stop_ovdp = len(sconti_recs)
                # Check the if the last period is overdued.
                if dued == repd or dued == obd:
                    stop_ovda -= duea
                    stop_ovdp -= 1

        # set_trace()
        ovdt.append((ever_ovdd, ever_ovdp, stop_ovdd, stop_ovdp))
        ovda.append(np.concatenate([ever_rema, ever_ovda, ever_duea,
                                    stop_rema, stop_ovda, stop_duea]))
        stop_recs.append(list(sconti_recs))
        mob.append(duei - 1)

    # `dtype` is passed to `np.asarray` here because `TypeError` will be raised
    # for the `np.timedelta64` in `ovdt`.
    return (np.asarray(ovdt).astype(np.int_),
            np.asarray(ovda, dtype=np.float64),
            np.asarray(mob, dtype=np.int_),
            stop_recs)


# %%
def ovdd_from_duepay_records(
    due_date: list | np.ndarray,
    ovd_days: list | np.ndarray,
    ob_date: list | np.ndarray = None,
    due_amt: list | np.ndarray = None,
    rem_amt: list | np.ndarray = None,
) -> tuple[np.ndarray]:
    """Calculate overdue days from duepay records.

    1. Mostly, the repayment records will be with format like:

        | ORDER_ID | PAYMENT_N | DUEPAY_DATE | OVERDUE_DAYS |
        |----------+-----------+-------------+--------------|

      That will be easy to get the day past due for each duepayment, but not
      convenient to get the overdue days at some specific time point, A.K.A.
      observation point.
      1.1 Here, `due_date`, along with other arrays, is assumed to be sorted
        in ascending order.
      1.2 The `overdue_days` are the day-delta between the duepayment dates
        and the repayment dates.

    2. There are 2 kind of overdue days for a period of time:

      2.1 Ever: The maximum overdue days ever occuring during the period
        of time between observation points.
      2.2 Stop: The overdue days at the observation point.

      Take the following time line for example:
        * due day: 2022-01-11
        * repayment day: 2022-01-28
        * observation: 2022-01-28
      The perception above could be calculated to:
        * ever overdue days: 17D
        * stop overdue days: 0D
        * ever overdue period: 1
        * stop overdue period: 0
      Explanation:
        At 2022-01-28, the repayment is done, so the whole overdue days is
        determined to be 17D.
        And when looking at the very end of 2022-01-28, the account isn't
        overdued any longer, and the whole overdue thing could be totally
        observed at 2022-01-28 and won't effect the following repayments.
        (Looking back from T+1 seems good.)

    3. A queue is used to maintain the periods that the responsible overdue
      days overpassed the former observation.
      The following assumptions are necessary for "ever"s:

      3.1 The observation should be strictly later than the duepay date.
        The overdue days is provided at each period during the iteration, and
        it will be used calculated the repayment date and overdue status. But
        if the observation is the duepay day, the infomation from overdue
        days shouldn't be used, which will bring the uncompatiablilty of
        overdue days and overdue periods.
      3.2 The gaps between repayments and the gaps between the observations
        should be (independently) close to equal. Or if the overdue periods
        overlap partly, the oldest period may not be the longest.
      3.3 The duepay amount and the remaining amount are calculated for the
        very point of the `ever` instead of the `ob_date`.
      3.4 The first longest continuous overdue periods will be chosen if two
        or more overdue periods are of the same length, without comparision.

    4. Take the following timeline as a example:

        | date        | dpd | last_dpd | due_gap | ob_gap | od_gap | ever | stop |
        |-------------+-----+----------+---------+--------+--------+------+------|
        | 2022-01-11  | 0   | 0        | 31      | 0*     |        |      |      |
        | 2022-01-28* |     |          |         | 0*     | 17     | 0    | 0    |
        | 2022-02-11  | 3   | 0        | 28      | 31     | 31     | 0    | 0    |
        | 2022-02-28* |     |          |         | 31     | 17     | 3    | 0    |
        | 2022-03-11  | 4   | 0        | 31      | 28     | 28     | 3    | 0    |
        | 2022-03-28* |     |          |         | 28     | 17     | 4    | 0    |
        | 2022-04-11  | 37  | 0        | 30      | 31     | 31     | 4    | 0    |
        | 2022-04-28* |     |          |         | 31     | 17     | 17   | 17   |
        | 2022-05-11  | 4   | 37       | 31      | 30     | 30     | 30   | 30   |
        | 2022-05-28* |     |          |         | 30     | 17     | 37   | 0    |
        | 2022-06-11  | 0   | 0        | 30      | 31     | 31     | 37   | 0    |
        | 2022-06-28* |     |          |         | 31     | 17     | 0    | 0    |
        | 2022-07-11  | 35  | 0        | 31      | 30     | 30     | 0    | 0    |
        | 2022-07-28* |     |          |         | 30     | 17     | 17   | 17   |
        | 2022-08-11  | 75  | 35       | 31      | 31     | 31     | 31   | 31   |
        | 2022-08-28* |     |          |         | 31     | 17     | 35   | 17   |
        | 2022-09-11  | 46  | 75       | 30      | 31     | 31     | 35   | 31   |
        | 2022-09-28* |     |          |         | 31     | 17     | 48   | 48   |
        | 2022-10-11  | 20  | 75       | 31      | 30     | 30     | 61   | 61   |
        | 2022-10-28* |     |          |         | 30     | 17     | 75   | 17   |
        | 2022-11-11  | 0   | 0        | 30*     | 31     | 31     | 75   | 0    |
        | 2022-11-28* |     |          |         | 31     | 17     | 20   | 0    |
        | inf*        |     |          |         | inf    | inf    | 0    | 0    |

      Note:
      * last_dpd: Former overdue days that effects current period.
      * due_gap: Gap of days between two adjacent duepayment date.
      * ob_gap: Gap of days between two adjacent observation date.
      * od_gap: Gap of days between duepayment date and responsible observation
        date.
      * *: Assumption for observation date or some setting for convenience.

    Params:
    ----------------------
    due_date: Sequence of duepayment date, which should be datetime64 or
      string that could be casted into datetime64 by NumPy.
    ovd_days: Sequence of overdue days of each repayments.
    ob_date: Sequence of observation date for each duepayment point.
      If no argument passed, this will be the `due_date` after shifting
      out the first duepay date and including a faraway date.
      ATTENTION: The `ob_date` should be strictly later than the `due_date` or
        the `duea` and `rema` will be hard to check.
      ATTENTION: As the repayment date is determined only after the day pasted,
        namely the next day and ', the `ob_date` should be defered for 1 day from the
        DDL, the 1st day for the month-end for example.
    due_amt: Sequence of duepay amount.
    rem_amt: Sequence of remaining amount.

    Return:
    ----------------------
    ever_ovdd: NDArray of maximum of overdue days ever occured during two
      responsible point of observation.
    stop_ovdd: NAArray of overdue days at the point of observation.
    ever_ovdp: Maximum of overdue periods ever.
      NOTE: The number periods counted here will be more precise as the days
      of each of month is not the same.
    stop_ovdp: Ditto, the same below.
    ever_ovda: Maximum (or the first) overdue amount ever.
    stop_ovda: Ditto.
    ever_rema: Maximum (or the first) remainal amount ever.
    stop_rema: Ditto.
    ever_duea: Maximum (or the first) duepay amount ever.
    stop_duea: Ditto.
    """
    dueds = np.asarray(due_date, dtype="datetime64[D]")
    if ob_date is None:
        obds = np.concatenate(
            [dueds[1:], np.array(["2999-12-31"], dtype="datetime64[D]")])
    else:
        obds = np.asarray(ob_date, dtype="datetime64[D]")
    ovdds = np.asarray(ovd_days, dtype="timedelta64[D]")
    ovdds[np.isnat(ovdds)] = np.timedelta64(0, "D")
    das = [0] * len(ovd_days) if due_amt is None else due_amt
    ras = [0] * len(ovd_days) if rem_amt is None else rem_amt

    OTD = np.timedelta64(0, "D")
    rec_N = len(due_date)

    ever_ovdd = [0] * rec_N     # Number of overdue days ever occuring during the period.
    stop_ovdd = [0] * rec_N     # Number of overdue days at the end of the period.
    ever_ovdp = [0] * rec_N     # Number of overdue periods ever occuring during the period.
    stop_ovdp = [0] * rec_N     # Number of overdue periods at the end of the period.
    ovd_Q = deque()
    ever_duea = [0] * rec_N
    stop_duea = [0] * rec_N
    ever_rema = [0] * rec_N
    stop_rema = [0] * rec_N
    ever_ovda = [0] * rec_N
    stop_ovda = [0] * rec_N

    for idx, dued, ovdd, obd, duea, rema in zip(range(rec_N), dueds, ovdds,
                                                obds, das, ras):
        if dued > obd:
            logger.warning(f"Invalid observation date in records at {dued}.")

        # if dued == np.datetime64("2022-04-11", "D"):
        #     set_trace()
        repd = dued + ovdd
        if len(ovd_Q) == 0:
            if repd < obd:
                ever_ovdd[idx] = ovdd
                stop_ovdd[idx] = OTD
                ever_ovdp[idx] = 1 if ovdd > 0 else 0
                stop_ovdp[idx] = 0

                ever_duea[idx] = duea
                stop_duea[idx] = 0
                ever_ovda[idx] = duea if (ovdd > 0 and obd > dued) else 0
                stop_ovda[idx] = 0
                ever_rema[idx] = rema + ever_duea[idx]
                stop_rema[idx] = rema + stop_duea[idx]
            else:
                # set_trace()
                # Append current repayment status iff repayment date overpass the
                # observation point strictly, which representes the rest of the
                # overdue days that can only be observed later.
                if repd > obd:
                    ovd_Q.append((dued, repd, duea, rema))

                ever_ovdd[idx] = obd - dued
                stop_ovdd[idx] = obd - dued
                ever_ovdp[idx] = 1 if ever_ovdd[idx] > 0 else 0
                stop_ovdp[idx] = 1 if stop_ovdd[idx] > 0 else 0

                ever_duea[idx] = duea
                stop_duea[idx] = duea
                ever_ovda[idx] = duea if (ovdd > 0 and obd > dued) else 0
                stop_ovda[idx] = duea if (ovdd > 0 and obd > dued) else 0
                ever_rema[idx] = rema + ever_duea[idx]
                stop_rema[idx] = rema + stop_duea[idx]
        else:
            last_dued, last_repd, last_duea, last_rema = ovd_Q[0]
            # set_trace()
            if last_repd > repd:
                logger.warning(f"Invalid overdue day in records at {dued}.")

            if last_repd > obd:
                if obd > dued:
                    ovd_Q.append((dued, repd, duea, rema))

                ever_ovdd[idx] = obd - last_dued
                stop_ovdd[idx] = obd - last_dued
                ever_ovdp[idx] = len(ovd_Q)
                stop_ovdp[idx] = len(ovd_Q)

                ever_ovda[idx] = sum([ele[2] for ele in ovd_Q])
                stop_ovda[idx] = sum([ele[2] for ele in ovd_Q])

                if obd == dued:
                    ovd_Q.append((dued, repd, duea, rema))

                # If `duea[idx] + rema[idx] == rema[idx-1]`, the following
                # assignment must lead to the same result:
                #   ever_rema[idx] = rema + ever_duea[idx]
                #   stop_rema[idx] = rema + stop_duea[idx]
                # Else the perception of remain amount isn't defined well.
                ever_duea[idx] = sum([ele[2] for ele in ovd_Q])
                stop_duea[idx] = sum([ele[2] for ele in ovd_Q])
                ever_rema[idx] = last_rema + last_duea
                stop_rema[idx] = last_rema + last_duea
            else:
                # set_trace()
                # `ever_ovdd` is assigned directly here only under the
                # assumption that both the gaps between repayments and the gaps
                # between the observations are independently equal. Or the gaps
                # between the last repayment and duepayment may not the largest
                # ever during current period.
                ever_ovdd[idx] = last_repd - last_dued
                ever_rema[idx] = last_rema + last_duea

                # `>` represent that the duepay amount won't be included at the
                # duepay day.
                if last_repd > dued:
                    ever_ovdp[idx] = len(ovd_Q) + 1
                    ever_duea[idx] = sum([ele[2] for ele in ovd_Q]) + duea
                    ever_ovda[idx] = sum([ele[2] for ele in ovd_Q]) + duea
                elif last_repd == dued:
                    ever_ovdp[idx] = len(ovd_Q)
                    ever_duea[idx] = sum([ele[2] for ele in ovd_Q]) + duea
                    ever_ovda[idx] = sum([ele[2] for ele in ovd_Q])
                else:
                    ever_ovdp[idx] = len(ovd_Q)
                    ever_duea[idx] = sum([ele[2] for ele in ovd_Q])
                    ever_ovda[idx] = sum([ele[2] for ele in ovd_Q])

                if repd > obd and obd > dued:
                    ovd_Q.append((dued, repd, duea, rema))

                while len(ovd_Q) > 0:
                    last_dued, last_repd, last_duea, last_rema = ovd_Q[0]
                    if last_repd == obd:
                        stop_ovdd[idx] = obd - last_dued
                        stop_ovdp[idx] = len(ovd_Q)
                        stop_ovda[idx] = sum([ele[2] for ele in ovd_Q])
                        stop_duea[idx] = sum([ele[2] for ele in ovd_Q])
                        stop_rema[idx] = last_rema + last_duea
                        while len(ovd_Q) > 0 and last_repd <= obd:
                            last_dued, last_repd, last_duea, last_rema = ovd_Q.popleft()
                        break
                    elif last_repd > obd:
                        stop_ovdd[idx] = obd - last_dued
                        stop_ovdp[idx] = len(ovd_Q)
                        stop_ovda[idx] = sum([ele[2] for ele in ovd_Q])
                        stop_duea[idx] = sum([ele[2] for ele in ovd_Q])
                        stop_rema[idx] = last_rema + last_duea
                        break
                    else:
                        ovd_Q.popleft()
                else:
                    stop_ovdd[idx] = OTD
                    stop_ovdp[idx] = 0
                    stop_ovda[idx] = 0
                    stop_duea[idx] = 0
                    stop_rema[idx] = rema + stop_duea[idx]

                if repd > obd and obd == dued:
                    ovd_Q.append((dued, repd, duea, rema))

    return (np.asarray(ever_ovdd).astype(np.int_),
            np.asarray(stop_ovdd).astype(np.int_),
            np.asarray(ever_ovdp), np.asarray(stop_ovdp),
            np.asarray(ever_ovda), np.asarray(stop_ovda),
            np.asarray(ever_rema), np.asarray(stop_rema),
            np.asarray(ever_duea), np.asarray(stop_duea))


# %%
def month_date(
    dates: pd.Series | list | np.ndarray,
    rule: str | int = 28,
    forced: bool = True,
) -> np.ndarray:
    """Generate dates for given date sequences.

    Mostly, this is called to generate observation date according to some
    predefined rules.

    Params:
    ----------------------
    due_date: Sequence of datetime64 or values could be casted into datetime64.
    rule: Rule to set observation date.
      nextdue: Next due date as the observation date, the last observation date
        will be 30days after the last due date.
      nextdue_noend: Ditto, but the last observation date will be 2099-12-31.
      monthend: The end of month for each due date.
      int: The fixed date of each month.
        1-28: The fixed date of the month for each due date.
        101-128: The fixed date of next month for each due date.
    forced: If to moved 1 month forward to ensure all the dates in result
      succeed the corresponding given dates.
      This only take effective when `rule` is an integer between 1 and 28.

    Return:
    ----------------------
    np.ndarray of Datetime64.
    """
    due_date = np.asarray(dates, dtype="M8[D]")

    # The same day of the next month may not exists, so 30 days is added up
    # instead of moving a month forward.
    if rule == "nextdue":
        stop_date = due_date.max() + np.timedelta64(30, "D")
        ob_date = np.concatenate(
            [due_date[1:], np.asarray([stop_date], dtype="M8[D]")])
    elif rule == "nextdue_noend":
        ob_date = np.concatenate(
            [due_date[1:], np.array(["2099-12-31"], dtype="M8[D]")])
    elif rule == "monthend":
        ob_date = (due_date.astype("M8[M]") + np.timedelta64(1, "M")
                   - np.timedelta64(1, "D"))
    elif isinstance(rule , int) and 1 <= rule <= 28:
        ob_date = due_date.astype("M8[M]") + np.timedelta64(rule - 1, "D")
        if np.any(ob_date < due_date):
            if forced:
                logger.info("Another month move forward to ensure the result "
                            "succeed the given dates.")
                ob_date = (due_date.astype("M8[M]") + np.timedelta64(1, "M")
                           + np.timedelta64(rule - 1, "D"))
    elif isinstance(rule , int) and 101 <= rule <= 128:
        ob_date = (due_date.astype("M8[M]")
                   + np.timedelta64(1, "M")
                   + np.timedelta64(rule - 101, "D"))
    else:
        raise ValueError(f"Invalid observeration date setting: {rule}.")

    return ob_date
