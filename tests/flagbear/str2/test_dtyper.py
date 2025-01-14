#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_dtyper.py
#   Author: xyy15926
#   Created: 2024-11-11 11:56:24
#   Updated: 2025-01-14 20:18:17
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear.llp import lex
    from flagbear.str2 import dtyper
    reload(lex)
    reload(dtyper)

import numpy as np
from datetime import datetime
import re

from flagbear.const.tokens import LEX_ENDFLAG
from flagbear.llp.lex import Lexer
from flagbear.str2.dtyper import stype_spec, regex_caster, str_caster


# %%
def test_stype_spec():
    regex = stype_spec("INT", "regex")
    assert re.fullmatch(regex, "23424")
    assert not re.fullmatch(regex, "234d24")
    caster = stype_spec("INT", "caster")
    assert caster("23424") == 23424

    regex = stype_spec("FLOAT", "regex")
    assert re.fullmatch(regex, "23424.2")
    assert re.fullmatch(regex, "+23424.2")
    assert re.fullmatch(regex, "-23424.2")
    assert re.fullmatch(regex, "23424.0")
    assert not re.fullmatch(regex, "23424")
    assert not re.fullmatch(regex, "234d24")
    caster = stype_spec("FLOAT", "caster")
    assert caster("-23424.0") == -23424

    regex = stype_spec("DATE", "regex")
    assert re.fullmatch(regex, "2020-10-10")
    assert re.fullmatch(regex, "2020-02-29")
    assert re.fullmatch(regex, "2021-02-29")
    assert not re.fullmatch(regex, "2021-10-10T")
    assert not re.fullmatch(regex, "2021-10-10T10:12:12")
    caster = stype_spec("DATE", "caster")
    # DATE and DATETIME caster are the same for convinience, right now.
    assert (caster("2023-01-01T10:12:12")
            == datetime.fromisoformat("2023-01-01T10:12:12"))

    regex = stype_spec("DATETIME", "regex")
    assert not re.fullmatch(regex, "2020-10-10")
    assert not re.fullmatch(regex, "2020-02-29")
    assert not re.fullmatch(regex, "2021-02-29")
    assert not re.fullmatch(regex, "2021-10-10T")
    assert re.fullmatch(regex, "2021-10-10T10:12:12")
    assert re.fullmatch(regex, "2021-10-10TT10:12:12")
    assert re.fullmatch(regex, "2021-10-1010:12:12")
    assert not re.fullmatch(regex, "2021-10-10TTT10:12:12")
    caster = stype_spec("DATETIME", "caster")
    assert (caster("2023-01-01T10:12:12")
            == datetime.fromisoformat("2023-01-01D10:12:12"))
    # `datetime.fromisoformat` accept any seperator besides `T`,
    # while `np.datetime64` only accept standard `T`.
    assert (caster("2023-01-01D10:12:12")
            == datetime.fromisoformat("2023-01-01D10:12:12"))
    caster = stype_spec("DATETIME", "caster", False)
    assert (caster("2023-01-01T10:12:12")
            == datetime.fromisoformat("2023-01-01D10:12:12"))
    assert (caster("2023-01-01D10:12:12")
            == datetime.fromisoformat("2023-01-01D10:12:12"))


# %%
def test_regex_caster():
    assert regex_caster("0") == (0, "INT")
    assert regex_caster("2342") == (2342, "INT")
    assert regex_caster("-2342") == (-2342, "INT")
    assert regex_caster("+2342") == (2342, "INT")
    assert regex_caster("2,342") is None
    assert regex_caster("2,342", match_ratio=0.6) == (342, "INT")
    assert regex_caster("2,342", match_ratio=0.1) == (2, "INT")

    token_types = ["DATE", "FLOAT", "INT"]
    token_specs = {tt: (stype_spec(tt, "regex"), stype_spec(tt, "caster"))
                   for tt in token_types}
    lexer = Lexer(token_specs, {}, set(), LEX_ENDFLAG)

    assert regex_caster("2342.23", lexer) == (2342.23, "FLOAT")
    assert regex_caster("-2342.23") == (-2342.23, "FLOAT")
    assert regex_caster("+2342.23") == (2342.23, "FLOAT")
    assert regex_caster("2,342.2323") == (342.2323, "FLOAT")

    # DATETIME is not included in `lexer`.
    assert regex_caster("2023-01-01T11:11:11", lexer) is None
    assert (regex_caster("2023-01-01T11:11:11")
            == (datetime.fromisoformat("2023-01-01T11:11:11"), "DATETIME"))
    assert (regex_caster("2023-01-01T11:11:11")
            == (datetime.fromisoformat("2023-01-01D11:11:11"), "DATETIME"))
    assert (regex_caster("2023-01-01TT11:11:11")
            == (datetime.fromisoformat("2023-01-01D11:11:11"), "DATETIME"))
    assert (regex_caster("2023-01-0111:11:11")
            == (datetime.fromisoformat("2023-01-01D11:11:11"), "DATETIME"))
    assert (regex_caster("2023/01/01T11:11:11")
            == (datetime.fromisoformat("2023-01-01D11:11:11"), "DATETIME"))
    assert regex_caster("2023-01-01d") == (datetime.fromisoformat("2023-01-01"), "DATE")
    assert regex_caster("d2023-01-01") == (datetime.fromisoformat("2023-01-01"), "DATE")
    assert regex_caster("2023/01/01") == (datetime.fromisoformat("2023-01-01"), "DATE")
    assert regex_caster("2023/01/01d") == (datetime.fromisoformat("2023-01-01"), "DATE")
    assert regex_caster("d2023/01/01") == (datetime.fromisoformat("2023-01-01"), "DATE")


# %%
def test_str_caster():
    assert str_caster("0", None) == 0
    assert str_caster("2342") == 2342
    assert str_caster("-2342") == -2342
    assert str_caster("+2342") == 2342

    assert str_caster("0", "AUTO") == 0
    assert str_caster("2342", "AUTO") == 2342
    assert str_caster("-2342", "AUTO") == -2342
    assert str_caster("+2342", "AUTO") == 2342

    assert str_caster("2342.23") == 2342.23
    assert str_caster("-2342.23") == -2342.23
    assert str_caster("+2342.23") == 2342.23
    assert str_caster("2,342.2323") == 342.2323

    with pytest.raises(ValueError):
        assert str_caster("0", "INT2")

    assert str_caster("ca2", "INT", dforced=False) == "ca2"
    assert np.isnan(str_caster("ca2", "INT", dforced=True))
    assert str_caster("ca2", "INT", dforced=True, dfill=0) == 0
