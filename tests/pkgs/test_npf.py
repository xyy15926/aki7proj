#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_npf.py
#   Author: xyy15926
#   Created: 2024-07-12 15:02:14
#   Updated: 2024-07-12 21:50:19
#   Description:
# ---------------------------------------------------------

# %%
from typing import Tuple, List, Any
import numpy as np
import numpy_financial as npf
from scipy import optimize


# %%
# `fv` the value at the end of the last period.
# fv +
# pv ** (1 + r) ** n +
# pmt * (1 + r * [end 0, begin 1]) * ((1 + r) ** n - 1) / r == 0
def test_future_value():
    # Terms, the compound interests will be computed for each terms.
    n_months = 36
    annual_rate = 0.05
    monthly_rate = annual_rate / 12

    # The negative sign represents cash flow out.
    present_value = -100
    payment_each = -100

    # Payment is paid at the end of each period, namely the last payment
    # is paid at the same time with the return cash in.
    npf_fve = npf.fv(rate=monthly_rate,
                     nper=n_months,
                     pv=present_value,
                     pmt=payment_each,
                     when="end")

    # Payment is paid at the begin of each period, namely the firat payment
    # is paid at the same time with the present value cash out.
    npf_fvb = npf.fv(rate=monthly_rate,
                     nper=n_months,
                     pv=present_value,
                     pmt=payment_each,
                     when="begin")

    fve = 0
    fve += present_value * (1 + monthly_rate) ** n_months
    fve += payment_each * ((1 + monthly_rate) ** n_months - 1) / monthly_rate

    fvb = 0
    fvb += present_value * (1 + monthly_rate) ** n_months
    fvb += payment_each * ((1 + monthly_rate) ** (n_months + 1)
                           - 1 - monthly_rate) / monthly_rate

    assert np.isclose(npf_fve + fve, 0)
    assert np.isclose(npf_fvb + fvb, 0)


# %%
# `pv` the value at the begin of the first period.
def test_present_value():
    # Terms, the compound interests will be computed for each terms.
    n_months = 36
    annual_rate = np.array([0.05, 0.06, 0.07])
    monthly_rate = annual_rate / 12

    # The negative sign represents cash flow out.
    payment_each = 100
    future_value = 100

    npf_pve = npf.pv(rate=monthly_rate,
                     nper=n_months,
                     pmt=payment_each,
                     fv=future_value,
                     when="end")

    npf_pvb = npf.pv(rate=monthly_rate,
                     nper=n_months,
                     pmt=payment_each,
                     fv=future_value,
                     when="begin")

    fve = 0
    fve += future_value
    fve += payment_each * ((1 + monthly_rate) ** n_months - 1) / monthly_rate
    pve = fve / (1 + monthly_rate) ** n_months

    fvb = 0
    fvb += future_value
    fvb += (payment_each * (1 + monthly_rate)
            * ((1 + monthly_rate) ** n_months - 1) / monthly_rate)
    pvb = fvb / (1 + monthly_rate) ** n_months

    assert np.all(np.isclose(npf_pve + pve, 0))
    assert np.all(np.isclose(npf_pvb + pvb, 0))


# %%
def test_npv():
    pmt = 100
    nper = 3
    r = 0.05
    # `fv` is the future values at the end of each period.
    fv = np.repeat(pmt, nper)

    npf_npv = npf.npv(r, fv)
    nfv = pmt * ((1 + r) ** nper - 1) / r
    npv = nfv / (1 + r) ** (nper - 1)
    assert np.isclose(npf_npv, npv)


# %%
def test_pmt():
    r = 0.05
    nper = 3 * 12
    pv = -20000
    fv = 10000

    # Equal loan payment.
    npf_pmte = npf.pmt(r, nper, pv, fv, "end")
    npf_pmtb = npf.pmt(r, nper, pv, fv, "begin")

    pmte = (fv + pv * (1 + r) ** nper) / ((1 + r) ** nper - 1) * r
    pmtb = (fv + pv * (1 + r) ** nper) / ((1 + r) ** nper - 1) / (1 + r) * r

    assert np.isclose(npf_pmte + pmte, 0)
    assert np.isclose(npf_pmtb + pmtb, 0)

    # The period start from 1 for both `end` and `begin`.
    npf_ipmte = npf.ipmt(r, np.arange(1, nper + 1), nper, pv, fv, "end")
    npf_ipmtb = npf.ipmt(r, np.arange(1, nper + 1), nper, pv, fv, "begin")
    npf_ppmte = npf.ppmt(r, np.arange(1, nper + 1), nper, pv, fv, "end")
    npf_ppmtb = npf.ppmt(r, np.arange(1, nper + 1), nper, pv, fv, "begin")

    assert np.all(np.isclose(npf_ipmte + npf_ppmte + pmte, 0))
    assert np.all(np.isclose(npf_ipmtb + npf_ppmtb + pmtb, 0))


# %%
def test_irr():
    r = 0.05
    nper = 3 * 12
    pv = -20000
    fv = 10000
    npf_pmte = npf.pmt(r, nper, pv, fv, "end")

    # Future value and the last payment are all at the end of the last period.
    vals = np.concatenate([[pv], np.repeat(npf_pmte, nper - 1), [npf_pmte + fv]])
    npf_irr = npf.irr(vals)

    def irrx(r):
        rr = np.logspace(0, len(vals) - 1, len(vals), base=1 + r)
        return np.sum(vals / rr)
    irr = optimize.bisect(irrx, 0, 1)

    assert np.isclose(npf_irr, r)
    assert np.isclose(irr, r)


def test_mirr():
    vals = np.array([100, -100, 100, -100])
    # Cash-in cost rate.
    finr = 0.02
    # Cash-out income rate.
    invr = 0.05

    finrr = np.logspace(0, len(vals) - 1, len(vals), base=1 + finr)
    invrr = np.logspace(0, len(vals) - 1, len(vals), base=1 + invr)[::-1]

    in_come_fv = (vals * invrr)[vals > 0].sum()
    cost_pv = -(vals / finrr)[vals < 0].sum()
    mirr = np.power(in_come_fv / cost_pv, 1 / (len(vals) - 1)) - 1
    npf_mirr = npf.mirr(vals, finr, invr)

    assert np.isclose(mirr, npf_mirr)


# %%
def irr2rate(
    irr: float = 0.092,
    nper: int = 36,
) -> Any:
    mirr = irr / 12
    # The payment for per 1 * period.
    mr = (1 + mirr) ** nper / ((1 + mirr) ** nper - 1) * mirr

    # The rate of the whole time.
    allr = mr * nper - 1
    # The rate for each month.
    mmr = allr / nper
    yr = mmr * 12

    print(f"万元系数：{mr * 10000} \n",
          f"总费率：{allr * 100:0.4f}% \n",
          f"年费率：{yr * 100:0.4f}% \n",
          f"月息：{mmr * 1000:0.4f}%")

    return mr * 10000, allr * 100, mmr * 1000, yr * 100
