#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dtyper.py
#   Author: xyy15926
#   Created: 2024-11-11 09:18:28
#   Updated: 2024-12-16 22:23:03
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any, TypeVar
from collections.abc import Iterable, Callable, Mapping

import logging
import json
from collections import deque

from functools import lru_cache, partial
from ubears.flagbear.llp.lex import Lexer
from ubears.flagbear.const.patterns import REGEXS
from ubears.flagbear.const.tokens import LEX_ENDFLAG

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
@lru_cache
def stype_spec(
    dtype: str,
    spec: str = "regex",
    extended: bool = True,
) -> Any:
    """Get specifications for string conversion.

    Params:
    ------------------------------
    dtype: Data type.
    spec: Specifications for the data type.
    extended: Import numpy or some other modules for convinience or
      representation string compatiability for dtype conversion.

    Return:
    ------------------------------
    Specifcation for the data type.
    """
    if extended:
        import numpy as np
        nan = np.nan
        nat = np.datetime64("NaT")

        def datetime64(x):
            x = x.replace("/", "-")
            if len(x) > 10:
                x = x[:10] + " " + x[-8:]
            return np.datetime64(x, "s")
    else:
        from datetime import datetime
        nan = float("nan")
        nat = None

        def datetime64(x):
            x = x.replace("/", "-")
            if len(x) > 10:
                x = x[:10] + " " + x[-8:]
            return datetime.fromisoformat(x)

    DTYPE_SPECS = {
        "FLOAT": {
            "dtype": "FLOAT",
            "regex": REGEXS["sfloat"],
            "caster": float,
            "default": nan,
            "nptype": "float",
        },
        "INT": {
            "dtype": "INT",
            "regex": REGEXS["sint"],
            "caster": int,
            "default": nan,
            "nptype": "int",
        },
        "DATE": {
            "dtype": "DATE",
            "regex": REGEXS["date"],
            "caster": datetime64,
            "default": nat,
            "nptype": "M8[D]",
        },
        "DATETIME": {
            "dtype": "DATETIME",
            "regex": REGEXS["date"] + ".{0,2}" + REGEXS["time"],
            "caster": datetime64,
            "default": nat,
            "nptype": "M8[s]",
        },
        "*CHAR*": {
            "dtype": "*CHAR*",
            "regex": r".*",
            "caster": str,
            "default": "",
            "nptype": "string",
        },
        "OBJECT": {
            "dtype": "OBJECT",
            "regex": None,
            "caster": None,
            "default": None,
            "nptype": "object",
        }
    }

    dtype = dtype.upper()
    if dtype in DTYPE_SPECS:
        sps = DTYPE_SPECS[dtype]
    elif dtype.startswith("VARCHAR") or dtype.startswith("CHAR"):
        sps = DTYPE_SPECS["*CHAR*"]
    else:
        sps = DTYPE_SPECS["OBJECT"]
    sps = sps.get(spec, None)

    return sps


# %%
def regex_caster(
    words: str,
    lexer: Lexer | None = None,
    match_ratio: float = 0.8,
    extended: bool = False,
) -> tuple[Any, str]:
    """Cast string into possible dtype with regex.

    Params:
    ------------------
    words: Input string.
    lexer: Lexer to parse input string.
      Lexer parse DATETIME, DATE, FLOAT and INT will be used as default.
    match_ratio: The threshold for match.
    extended: Import numpy or some other modules for convinience or
      representation string compatiability for dtype conversion.

    Return:
    ------------------
    token.val: Any
    token.type: str
    """
    # Only int, float, date or datetime will be casted from string.
    if lexer is None:
        # Keep the order for the master regex.
        token_types = ["DATETIME", "DATE", "FLOAT", "INT"]
        token_specs = {tt: (stype_spec(tt, "regex"), stype_spec(tt, "caster"))
                       for tt in token_types}
        lexer = Lexer(token_specs, {}, set(), LEX_ENDFLAG)

    words_len = len(words)
    for token in lexer.input(words):
        if token.len / words_len >= match_ratio:
            return token.val, token.type
    return None


# %%
def str_caster(
    words: str,
    dtype: str = None,
    extended: bool = False,
    dfill: Any = None,
    dforced: bool = False,
) -> Any:
    """Cast string into other dtype.

    Params:
    ---------------------
    words: String to be casted to other dtype.
    dtype: str | AUTO | None
      AUTO: Call `regex_caster` to cast string to any proper dtype.
      str: Cast string to dtype defined in `stype_spec` by call `str_caster`.
        INT:
        FLOAT:
        DATE:
        DATETIME:
    extended: Import numpy or some other modules for convinience or
      representation string compatiability for dtype conversion.
    dfill: The default value after dtype casting fails.
      This will override the default values from `stype_spec` if not None.
    dforced: If to rollback to used `dfill` if converters in `stype_spec` fails.
      NOTE: The `forced` here means to ensure the return must of the `dtype`.

    Return:
    ---------------------
    Any
    """
    ret = words
    if dtype is None or dtype == "AUTO":
        ret = regex_caster(ret)
        if ret is not None:
            ret = ret[0]
    else:
        # `stype_spec` stores dtype with capital letters.
        dtype = dtype.upper()
        convers = stype_spec(dtype, "caster", extended)
        if convers is not None:
            try:
                ret = convers(ret)
            except Exception as e:
                logger.info(e)
                if dforced:
                    if dfill is None:
                        ret = stype_spec(dtype, "default", extended)
                    else:
                        ret = dfill
        else:
            raise ValueError(f"Unrecognized dtype {dtype}.")

    return ret
