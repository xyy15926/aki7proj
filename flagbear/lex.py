#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: lex.py
#   Author: xyy15926
#   Created: 2023-11-29 20:17:03
#   Updated: 2023-12-03 21:07:01
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import copy
import re
from collections import defaultdict, deque
from sys import maxsize as MAXINT
from typing import Any, NamedTuple
from flagbear.patterns import TOKEN_SPECS, RESERVEDS, ENDFLAG

from IPython.core.debugger import set_trace


# %%
class Token(NamedTuple):
    type: str
    value: str
    lineno: int = -1


# %%
class Lexer:
    def __init__(
        self, token_specs: list = TOKEN_SPECS,
        reserveds: dict = RESERVEDS,
        end_flag: str = ENDFLAG,
    ) -> None:
        self.token_specs = token_specs
        self.reserveds = reserveds
        self.end_flag = end_flag
        self.master_re = None
        self.build_master_re()

    def build_master_re(self) -> None:
        token_res = []
        for key, val in self.token_specs.items():
            if isinstance(val, (tuple, list)):
                token_res.append((key, val[0]))
            else:
                token_res.append((key, val))
        master_re = "|".join(f"(?P<{name}>{ptn})" for name, ptn in token_res)
        self.master_re = re.compile(master_re)

    def input(self, data: str) -> None:
        """
        Description:
        Token iterator generator to yield `Token`.
        """
        # Make local copy of frequently used local attributes.
        master_re = self.master_re
        token_specs = self.token_specs
        reserveds = self.reserveds
        lineno = 1

        # Look for regular expression matchs.
        # Lexical errors are skip by `finditer` here, which `mo.span` may help.
        for mo in master_re.finditer(data):
            if mo is None:
                continue
            tok_type = mo.lastgroup
            tok_value = mo.group()

            # Update `lineno` for log.
            for _c in tok_value:
                lineno += _c == "\n"

            # Check `tok_type`.
            # 1. Skip this token if `tok_type` starts with `SKIP`.
            if tok_type.startswith("SKIP"):
                continue

            # Transfer token's value.
            if isinstance(token_specs[tok_type], (tuple, list)):
                tok_value = token_specs[tok_type][1](tok_value)

            # Check if identifier is reserved words.
            tok_type = reserveds.get(tok_value, tok_type)

            yield Token(tok_type, tok_value, lineno)

        yield Token(self.end_flag, "", -1)
