#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_pronoun.py
#   Author: xyy15926
#   Created: 2025-02-14 10:23:18
#   Updated: 2025-02-14 10:50:33
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.modsbear.locale import pronoun
    reload(pronoun)

import string
from ubears.modsbear.locale.pronoun import (
    certno_parity,
    check_certno,
    rand_certno,
    rand_mobile,
    rand_nname,
    rand_email,
    rand_orgname,
    orgno_parity,
    check_orgno,
    rand_orgno,
    rand_addr,
    LASTNAME_ALTS,
)


# %%
def test_certno():
    for i in range(10):
        rct = rand_certno()
        assert len(rct) == 6 + 8 + 4
        for ele in rct:
            assert ele in string.digits or ele == "X"

        # Specify government region id.
        rct = rand_certno("11")
        assert len(rct) == 6 + 8 + 4
        assert rct.startswith("11")
        for ele in rct:
            assert ele in string.digits or ele == "X"

        # Validation Check.
        assert check_certno(rct)
        assert certno_parity(rct[:-1]) == rct[-1]


# %%
def test_orgno():
    for i in range(10):
        rct = rand_orgno()
        assert len(rct) == 18
        for ele in rct:
            assert ele in string.digits + string.ascii_uppercase

        # Validation Check.
        assert check_orgno(rct)
        assert orgno_parity(rct[:-1]) == rct[-1]


# %%
def test_rand_others():
    mobile = rand_mobile()
    assert len(mobile) == 11

    nname = rand_nname()
    assert len(nname) < 5
    assert nname[0] in LASTNAME_ALTS

    email = rand_email()
    assert "@" in email and "." in email
