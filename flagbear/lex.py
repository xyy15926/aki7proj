#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: lex.py
#   Author: xyy15926
#   Created: 2023-11-29 20:17:03
#   Updated: 2024-05-27 12:14:14
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

from typing import Any
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from collections.abc import Iterator, Callable
from collections.abc import Generator, Mapping

import re
import logging
from functools import lru_cache
from flagbear.patterns import (LEX_TOKEN_SPECS, LEX_RESERVEDS,
                               LEX_ENDFLAG, LEX_SKIPS)
from flagbear.patterns import LEX_TOKEN_PRECS

# from IPython.core.debugger import set_trace

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


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
        token_precs: dict = LEX_TOKEN_PRECS,
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
        token_precs: Dict[token_type, int representing priority].
        """
        self.token_specs = token_specs
        self.master_re = None
        self.build_master_re()
        self.reserveds = reserveds
        self.skips = skips
        self.end_flag = end_flag
        self.token_precs = LEX_TOKEN_PRECS
        self.env = None

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

    @lru_cache(1000)
    def compile(self, words: str) -> list[Token]:
        """Transfrom infix notion into postfix notion.

        1. `()` are used to enclose the sub-sequence of tokens of higher
          priority, which should have opposite priority and be opposed after
          pushed into temporary stack.
        2. Postfix notion, A.K.A. Reversed Polished Notion, is used for
          machine to comprehend expression easily, which could be used as a
          restricted expression hanlder without syntaxer.

        Params:
        ---------------------
        words: String of infix notion.

        Return:
        ---------------------
        List[Token]
        """
        LPAR = "LPAR"
        RPAR = "RPAR"
        token_precs = {k: v[0] for k,v in self.token_precs.items()}

        rpn = []                                        # List of postfix notion.
        tok_st = [Token(LPAR, "(", -1, -1)]           # Temporary stack.
        for tok in self.input(words):
            out_type, *ele = tok
            # Break at the end of the tokens.
            if out_type == self.end_flag:
                break
            if out_type not in token_precs:
                rpn.append(tok)
                continue

            out_level = token_precs[out_type]
            # Oppose the priority of the LPAR and RPAR out of stack.
            if out_type == LPAR or out_type == RPAR:
                out_level *= -1

            in_level = token_precs[tok_st[-1][0]]

            # Pop the token in temporary stack if token at the top of the
            # stack has higher priority than token outside.
            while in_level > out_level:
                rpn.append(tok_st.pop())
                in_level = token_precs[tok_st[-1][0]]

            if out_type == RPAR:
                assert tok_st[-1][0] == LPAR
                tok_st.pop()
            else:
                tok_st.append(tok)

        # Pop the rest of the toknes out.
        while len(tok_st) > 1:
            rpn.append(tok_st.pop())

        return rpn

    def bind_env(self, env: Mapping) -> None:
        """Bind environment.

        Params:
        ----------------------
        env: The dict to get the value of ID from.

        Return:
        ----------------------
        self
        """
        self.env = env
        return self

    def exec(self, toks: list[Token]) -> Any:
        """Execute the token list in reversed polished order.

        Only reductions represented by operators defined in `self.token_precs`
        are supported.
        1. Append value of oprand node to temperary stack directly.
        2. Reduce part of the values in temperary stack for operator, with
          the number of values for reduction popped out.

        Params:
        --------------------
        toks: Tokens of oprands or operators in reversed polished order.

        Return:
        --------------------
        Execution result.
        """
        token_precs = self.token_precs
        env = self.env

        val_st = []             # Stack storing values temporarily.
        for tok_type, tok_val, *ele in toks:
            if tok_type in token_precs:
                priority, arg_n, call = token_precs[tok_type]
                cur_opts = val_st[-arg_n:]
                # Reconstruct stack storing values temporarily.
                val_st = val_st[:-arg_n] + [call(*cur_opts),]
            else:
                # Identifier must be in bound environment.
                if tok_type == "ID":
                    try:
                        tok_val = env[tok_val]
                    except KeyError:
                        raise KeyError(f"Unrecognized variable {tok_val}"
                                       f"in bound envrionment.")
                val_st.append(tok_val)

        return val_st[-1]

    def parse(self, words: str) -> Any:
        """Compile and execute the words.

        Params:
        --------------------
        words: Input words.

        Return:
        --------------------
        Execution result.
        """
        rpn = self.compile(words)
        return self.exec(rpn)
