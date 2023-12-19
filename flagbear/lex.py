#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: lex.py
#   Author: xyy15926
#   Created: 2023-11-29 20:17:03
#   Updated: 2023-12-18 20:06:45
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import re
from typing import NamedTuple, Any
from collections.abc import Generator
from flagbear.patterns import (LEX_TOKEN_SPECS, LEX_RESERVEDS,
                               LEX_ENDFLAG, LEX_SKIPS)

# from IPython.core.debugger import set_trace


# %%
class Token(NamedTuple):
    type: str
    val: Any
    lineno: int = -1
    len: int = -1


# %%
# TODO: State Control
class Lexer:
    """Lex analyzer implemented with regex.

    Lex analyzer implemented with regular expression.
    1. This class will compile a master regex to detect tokens in given input.
    2. END_FLAG token will be used to mark the end of token stream.

    Attrs:
    -------------
    token_specs: dict of {TOKEN_TYPE: REGEX_PATTERN}
      Token type names and their regex patterns.
      Attention: Token patterns could override each others, but preceding
        patterns will be detected first for the `re.finditer` feature. So do
        remember to order the tokens carefully.
      Attention: Master regular are constructed with the advantage of named
        group feature in python's re. So only no-captureing version of
        parentheses are allowed in tokens' regexs.
    master_re: re.Pattern
      Master regex constructed from `token_specs` with named group.
    reserveds: dict of {TOKEN_VALUE: TOKEN_TYPE}
      Special tokens share the same token regex pattern but should be treated
      differently. Tokens detected will be checked and converted to reserveds
      if true.
    skpis: set of {TOKEN_TYPE}
      Special tokens to be skiped.
    end_flag: string
      Speical token type name to mark the end of a token stream.
    """
    def __init__(
        self, token_specs: dict = LEX_TOKEN_SPECS,
        reserveds: dict = LEX_RESERVEDS,
        skips: set = LEX_SKIPS,
        end_flag: str = LEX_ENDFLAG,
    ) -> None:
        """Init the instance based on attributes.

        Params:
        ------------------
        token_specs: dict of {TOKEN_TYPE: REGEX_PATTERN}
          Token type names and their regex patterns.
          Atttention: Token patterns could override each others, but preceding
            patterns will be detected first for the `re.finditer` feature. So
            do remember to order the tokens carefully.
          Attention: Master regular are constructed with the advantage of named
            group feature in python's re. So only no-captureing version of
            parentheses are allowed in tokens' regexs.
        reserveds: dict of {TOKEN_VALUE: TOKEN_TYPE}
          Special tokens share the same token regex pattern but should be
          treated differently. Tokens detected will be checked and converted to
          reserveds if true.
        skpis: set of {TOKEN_TYPE}
          Special tokens to be skiped.
        end_flag: string
          Speical token type name to mark the end of a token stream.
        """
        self.token_specs = token_specs
        self.master_re = None
        self.build_master_re()
        self.reserveds = reserveds
        self.skips = skips
        self.end_flag = end_flag

    def build_master_re(self) -> None:
        """Build master regex.

        Named group feature `(?P<NAME>PATTERN)` will be used to construct
        master regex for convinience to extract detections.
        """
        token_res = []
        for key, val in self.token_specs.items():
            if isinstance(val, (tuple, list)):
                token_res.append((key, val[0]))
            else:
                token_res.append((key, val))
        master_re = "|".join(f"(?P<{name}>{ptn})" for name, ptn in token_res)
        self.master_re = re.compile(master_re)

    def input(self, data: str) -> Generator[Token]:
        """Parse input into tokens.

        Return Token iterator generator to yield `Token` parsed from input
        string.
        1. Token(END_FLAG, "", -1) will be yielded at last to mark the end of
          the token stream.

        Params:
        ---------------
        data: str
          Input string.

        Yield:
        ---------------
        Token(type, value, lineno)

        Raise:
        ---------------
        StopIteration
        """
        # Make local copy of frequently used local attributes.
        master_re = self.master_re
        token_specs = self.token_specs
        reserveds = self.reserveds
        skips = self.skips
        lineno = 1

        # Look for regular expression matchs.
        # Lexical errors are skip by `finditer` here, which `mo.span` may help.
        for mo in master_re.finditer(data):
            if mo is None:
                continue
            tok_type = mo.lastgroup
            tok_value = mo.group()
            tok_len = len(tok_value)

            # Update `lineno` for log.
            # If the `data` is not raw string, `\n` will be escaped
            # automaticlly and `lineno` will be updated wrongly.
            for _c in tok_value:
                lineno += _c == "\n"

            # Check `tok_type`.
            if tok_type in skips:
                continue

            # Check if identifier is reserved words.
            tok_type = reserveds.get(tok_value, tok_type)

            # Transfer token's value.
            # Reserveds' token type may not exists in `token_specs`.
            if isinstance(token_specs.get(tok_type), (tuple, list)):
                tok_value = token_specs[tok_type][1](tok_value)

            yield Token(tok_type, tok_value, lineno, tok_len)

        yield Token(self.end_flag, "", -1, -1)
