#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: parser.py
#   Author: xyy15926
#   Created: 2023-12-02 21:04:21
#   Updated: 2023-12-03 22:09:46
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

from typing import Any
from collections import namedtuple
from flagbear.patterns import ARITH_PRODS
from flagbear.lex import Token, Lexer
from flagbear.syntax import Production, LRItem, LRState, Syntax

from IPython.core.debugger import set_trace


# %%
class LRParser:
    # TODO
    def __init__(
        self, lexer: Lexer = None,
        syntax: Syntax = None,
        productions: list = ARITH_PRODS,
    ):
        self.lexer = lexer or Lexer()
        self.calls = {}
        if syntax is None:
            productions, precedences = self.parse_productions(productions)
            syntax = Syntax(productions)
            syntax.precedences = precedences
            syntax.find_nullables()
            syntax.digraph_first()
            syntax.digraph_follow()
            syntax.lookahead()
            syntax.LALR_states_and_gotos()
            self.syntax = syntax
        else:
            self.syntax = syntax

    def parse_productions(self, prod_strs: list):
        calls = self.calls
        productions = []
        precedences = {}
        for prod_str, call, prec in prod_strs:
            lp, rp = prod_str.split("->")
            lp = lp.strip()
            rp = rp.strip().split()
            prod = Production(lp, tuple(rp))
            calls[prod] = call
            precedences[prod] = prec
            productions.append(prod)
        return productions, precedences

    def parse(self, words):
        lexer = self.lexer
        syntax = self.syntax
        calls = self.calls

        def red(x, y):
            if x == "Token":
                return y.value
            else:
                return calls[x](y)
        return syntax.reduce_tokens(lexer.input(words), red)


# %%
def bind_precedences(self, prec_L: list | dict):
    if isinstance(prec_L, dict):
        self.precedences = prec_L
        return
    precedences = self.precedences
    productions = self.productions
    # Set precedences.
    for prec, assoc, items in enumerate(prec_L):
        for item in items:
            if isinstance(item, int):
                precedences[productions[item]] = prec
            else:
                precedences[item] = prec


def pprint(syntax):
    gotos = syntax.gotos
    pgoto = {}
    for (state, sym), sl in gotos.items():
        ss = f"I{state.index}"
        sls = f"R{sl.prod_idx}" if isinstance(sl, LRItem) else f"I{sl.index}"
        pgoto.setdefault(ss, {})[sym] = sls

    pconflicts = {}
    conflicts = syntax.conflicts
    for (state, sym), sl in conflicts.items():
        ss = f"I{state.index}"
        sls = sl
        pconflicts.setdefault(ss, {})[sym] = sls

    return syntax.LRItems, pgoto, pconflicts


# %%
def dict2namedtuple(d: dict) -> list:
    names = set()
    # Collect names.
    dd = {}
    for key, val in d.items():
        if isinstance(val, dict):
            names.update(val.keys())
        # Reconstruct dict with structure: {(row, col): val}
        else:
            dd.setdefault(key[0], {})[key[1]] = val
            names.add(key[1])
    # Determine structured dict
    dd = dd or d
    names = ["KEY", *sorted(names)]
    NamedT = namedtuple("NT", names)
    nts = []
    for key, val in dd.items():
        nt = NamedT(key, **val)
        nts.append(nt)
    return nts


