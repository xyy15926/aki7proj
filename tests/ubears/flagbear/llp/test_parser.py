#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_parser.py
#   Author: xyy15926
#   Created: 2023-12-14 21:14:49
#   Updated: 2025-01-20 17:41:37
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.llp import lex, syntax, parser, graph
    reload(lex)
    reload(syntax)
    reload(parser)
    reload(graph)

from ubears.flagbear.llp.lex import Lexer
from ubears.flagbear.llp.syntax import Production, LRItem, Syntaxer
from ubears.flagbear.const.prods import (
    SYN_STARTSYM,
    SYN_EXPR_PRODS,
)
from ubears.flagbear.llp.parser import EnvParser


# %%
# TODO
def test_EnvParser():
    envp = EnvParser()
    env = {"a": 1, "b": 2}
    envp = envp.bind_env(env)
    assert envp.parse("len(_)") == 2
    assert envp.parse("a + b") == 3
    assert envp.parse("a in [1, 2]")
    assert envp.parse("len([1, 2])") == 2
    assert envp.parse("sum([a, a, b])") == 4
    assert envp.parse("a==1")
    assert envp.parse("sum([a, b])") == 3
    assert envp.parse("sum([a, b])") == 3
    assert envp.parse("-sum([a, b])") == -3
    assert envp.parse("_") == env

    env = {"a": 1, "b": 3, "c": 4}
    assert envp.bind_env(env).parse("a + b") == 4
    assert envp.bind_env(env).parse("c") == 4

    env = {"a": 1, "b": 2}
    envp = envp.bind_env(env)
    assert envp.parse("len(_)") == 2
    assert envp.parse("a + b") == 3
    assert envp.parse("a in [1, 2]")
    assert envp.parse("len([1, 2])") == 2
    assert envp.parse("sum([a, a, b])") == 4
    assert envp.parse("a==1")
    assert envp.parse("sum([a, b])") == 3
    assert envp.parse("sum([a, b])") == 3
    assert envp.parse("-sum([a, b])") == -3
    assert envp.parse("_") == env
