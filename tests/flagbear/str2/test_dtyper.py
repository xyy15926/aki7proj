#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_dtyper.py
#   Author: xyy15926
#   Created: 2024-11-11 11:56:24
#   Updated: 2024-11-11 11:58:01
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear.llp import lex, patterns
    from flagbear.str2 import dtyper
    reload(patterns)
    reload(lex)
    reload(dtyper)

import numpy as np
from datetime import date, time

from flagbear.llp.patterns import REGEX_TOKEN_SPECS, LEX_ENDFLAG
from flagbear.llp.lex import Lexer
from flagbear.str2.dtyper import regex_caster, str_caster


# %%
def test_regex_caster():
    assert regex_caster("0") == (0, "INT")
    assert regex_caster("2342") == (2342, "INT")
    assert regex_caster("-2342") == (-2342, "INT")
    assert regex_caster("+2342") == (2342, "INT")
    assert regex_caster("2,342") is None
    assert regex_caster("2,342", match_ratio=0.6) == (342, "INT")
    assert regex_caster("2,342", match_ratio=0.1) == (2, "INT")

    lexer = Lexer(REGEX_TOKEN_SPECS, {}, set(), LEX_ENDFLAG)

    assert regex_caster("2342.23", lexer) == (2342.23, "FLOAT")
    assert regex_caster("-2342.23") == (-2342.23, "FLOAT")
    assert regex_caster("+2342.23") == (2342.23, "FLOAT")
    assert regex_caster("2,342.2323") == (342.2323, "FLOAT")

    assert regex_caster("2023-01-01", lexer) == (date.fromisoformat("2023-01-01"), "DATE")
    assert regex_caster("2023-01-01d") == (date.fromisoformat("2023-01-01"), "DATE")
    assert regex_caster("d2023-01-01") == (date.fromisoformat("2023-01-01"), "DATE")

    assert regex_caster("2023/01/01", lexer) == (date.fromisoformat("2023-01-01"), "DATE")
    assert regex_caster("2023/01/01d") == (date.fromisoformat("2023-01-01"), "DATE")
    assert regex_caster("d2023/01/01") == (date.fromisoformat("2023-01-01"), "DATE")

    assert regex_caster("12:12:12", lexer) == (time.fromisoformat("12:12:12"), "TIME")
    assert regex_caster("T12:12:12") == (time.fromisoformat("12:12:12"), "TIME")
    assert regex_caster("12:12:12T") == (time.fromisoformat("12:12:12"), "TIME")


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
