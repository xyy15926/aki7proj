#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: dtyper.py
#   Author: xyy15926
#   Created: 2024-11-11 09:18:28
#   Updated: 2024-11-11 09:19:53
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import Any, TypeVar
from collections.abc import Iterable, Callable, Mapping

import logging
import json
from collections import deque

from flagbear.llp.lex import Lexer
from flagbear.llp.patterns import REGEX_TOKEN_SPECS, LEX_ENDFLAG

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def regex_caster(
    words: str,
    lexer: Lexer | None = None,
    match_ratio: float = 0.8,
) -> tuple[Any, str]:
    """Cast string into possible dtype with regex.

    Params:
    ------------------
    words: Input string.
    lexer: Lexer to parse input string.
      Lexer inited with `REGEX_TOKEN_SPECS` will be used as default.

    Return:
    ------------------
    token.val: Any
    token.type: str
    """
    lexer = (Lexer(REGEX_TOKEN_SPECS, {}, set(), LEX_ENDFLAG)
             if lexer is None
             else lexer)
    words_len = len(words)
    for token in lexer.input(words):
        if token.len / words_len >= match_ratio:
            return token.val, token.type
    return None


# %%
def str_caster(
    words: str,
    dtype: str = None,
    dforced: bool = False,
    dfill: Any = None,
    regex_specs: Mapping = REGEX_TOKEN_SPECS,
) -> Any:
    """Cast string into other dtype.

    Params:
    ---------------------
    words: String to be casted to other dtype.
    dtype: str | AUTO | None
      AUTO: Call `regex_caster` to cast string to any proper dtype.
      str: Casting string to indicating dtype in `REGEX_TOKEN_SPECS`(default)
        INT:
        FLOAT:
        TIME:
        DATE:
    dforced: If to rollback to used `dfill` if converters in
      `REGEX_TOKEN_SPECS` fails.
    dfill: The default value after dtype casting fails.
      This will override the default values in `regex_specs` if not None.
    regex_specs: Mapping[dtype, (regex, convert-function, default,...)]
      Mapping storing the dtype name and the handler.

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
        # `REGEX_TOKEN_SPECS` stores dtype with capital letters.
        dtype = dtype.upper()
        convers = regex_specs.get(dtype)
        if convers is not None:
            try:
                ret = convers[1](ret)
            except Exception as e:
                logger.info(e)
                if dforced:
                    if dfill is None:
                        ret = convers[2]
                    else:
                        ret = dfill
        else:
            raise ValueError(f"Unrecognized dtype {dtype}.")

    return ret
