#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: gotcha.py
#   Author: xyy15926
#   Created: 2024-11-21 14:22:18
#   Updated: 2024-11-21 20:27:50
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, Any

import json

from ubears.modsbear.locale.pronoun import (
    rand_certno,
    rand_mobile,
    rand_nname,
    rand_email,
    rand_orgname,
    rand_orgno,
    rand_addr,
)
from ubears.flagbear.str2.fliper import reset_field
from ubears.flagbear.slp.finer import get_assets_path


# %%
def mosaic_sensitives(ctt: dict):
    """Replace sensetive fields in PBOC reports.
    """
    # Replace names.
    reps = [
        ("PRH:PA01:PA01B:PA01BQ01", rand_nname),
        ("PRH:PA01:PA01B:PA01BI01", rand_certno),
        ("PRH:PA01:PA01B:PA01BI02", rand_orgno),
        ("PIM:PB01:PB01A:PB01AQ01", rand_email),
        ("PIM:PB01:PB01A:PB01AQ02", rand_addr),
        ("PIM:PB01:PB01A:PB01AQ03", rand_addr),
        ("PMM:PB02:PB020I01", rand_certno),
        ("PMM:PB02:PB020Q01", rand_nname),
        ("PMM:PB02:PB020Q02", rand_orgname),
        ("PMM:PB02:PB020Q03", rand_mobile),
        ("PIM:PB01:PB01B:PB01BH:[_]:PB01BQ01", rand_mobile),
        ("PRM:PB03:[_]:PB030Q01", rand_addr),
        ("PRM:PB03:[_]:PB030Q02", rand_mobile),
        ("POM:PB04:[_]:PB040Q01", rand_orgname),
        ("POM:PB04:[_]:PB040Q02", rand_addr),
        ("POM:PB04:[_]:PB040Q03", rand_mobile),
        ("PHF:PF05:[_]:PF05AQ04", rand_orgname),
    ]
    for steps, cval in reps:
        ctt = reset_field(ctt, steps, cval)

    return ctt


# %%
if __name__ == "__main__":
    reports_path = get_assets_path() / "pboc_reports"
    files = list(filter(lambda x: x.name.startswith('2024'),
                        reports_path.iterdir()))
    for file in files:
        ctt = json.load(open(file, "r"))
        ctt = mosaic_sensitives(ctt)
        json.dump(ctt, open(file, "w"), ensure_ascii=False, indent=4)
