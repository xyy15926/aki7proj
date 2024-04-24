#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_parser.py
#   Author: xyy15926
#   Created: 2023-12-14 21:14:49
#   Updated: 2023-12-19 15:03:25
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear import lex, syntax, parser, patterns, graph, tree
    reload(lex)
    reload(syntax)
    reload(parser)
    reload(patterns)
    reload(graph)
    reload(tree)

from flagbear.lex import Lexer
from flagbear.syntax import Production, LRItem, Syntaxer
from flagbear.patterns import (LEX_TOKEN_SPECS, LEX_SKIPS,
                               LEX_RESERVEDS, LEX_ENDFLAG,
                               SYN_STARTSYM, SYN_EXPR_PRODS)
from flagbear.parser import LRParser, EnvParser


# %%
def test_Parser():
    parser = LRParser(LEX_TOKEN_SPECS, LEX_RESERVEDS, LEX_SKIPS,
                      SYN_EXPR_PRODS, SYN_STARTSYM, LEX_ENDFLAG)
    exprs = [
        "{(2+4)*8+-1000*9+1-1.5, 9}",
        "[1, 2, 3]",
        "[1, 2, 3, [], [1,2+3,(2+4)*4]]",
        "1 in {2,3}",
        "1 in {1, 2}",
        "1+3 in {4, 5}",
        "[1, 2, 3][2]",
    ]
    for expr in exprs:
        assert parser.parse(expr) == eval(expr)


# %%
# TODO
def test_EnvParser():
    env = {"a": 1, "b": 2}
    envp = EnvParser().bind_env(env)
    assert envp.parse("a + b") == 3
    assert envp.parse("a in [1, 2]")
    assert envp.parse("count([1, 2])") == 2
    assert envp.parse("sum([a, a, b])") == 4
    assert envp.parse("a==1")
    assert envp.parse("sum([a, b])") == 3
