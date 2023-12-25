#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_fliper.py
#   Author: xyy15926
#   Created: 2023-12-18 19:42:15
#   Updated: 2023-12-21 20:41:14
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear import lex, syntax, parser, patterns, graph, tree, fliper
    reload(graph)
    reload(tree)
    reload(patterns)
    reload(lex)
    reload(syntax)
    reload(parser)
    reload(fliper)

from datetime import date, time
from flagbear.patterns import REGEX_TOKEN_SPECS, LEX_ENDFLAG
from flagbear.lex import Lexer
from flagbear.parser import EnvParser
from flagbear.fliper import (regex_caster, extract_field, rebuild_dict,
                             rename_duplicated)


# %%
# TODO
def test_rename_duplicated():
    pass


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
def test_extract_field():
    env = {
        "a": 1,
        "b": 2,
        "c": {
            "ca": "ca",
            "cb": 2,
            "cc": {
                "cca": 1,
                "ccb": [
                    {"ccba": 1, "ccbb": 2, },
                    {"ccba": 2, "ccbb": 4, },
                    {"ccba": 1, "ccbb": 4, },
                    {
                        "ccba": 2,
                        "ccbb": 2,
                        "ccbc": [
                            {"ccbca": 1, "ccbcb": 2, },
                            {"ccbca": 2, "ccbcb": 4, },
                        ],
                    },
                    {
                        "ccba": 1,
                        "ccbb": 1,
                        "ccbc": [
                            {"ccbca": 1, "ccbcb": 2, },
                            {"ccbca": 2, "ccbcb": 4, },
                        ],
                    },
                ],
            },
        },
    }
    assert extract_field(env, "a") == 1
    assert extract_field(env, "a:b") is None
    assert extract_field(env, "a:b:c") is None
    assert extract_field(env, "c:cc:cca") == 1
    assert extract_field(env, "c:cc:ccb:[count(_)]") == 5
    assert extract_field(env, "c:cc:ccb:[]:ccba") == [1, 2, 1, 2, 1]
    assert extract_field(env, "c:cc:ccb:[max(_)]:ccba") == 2
    assert extract_field(env, "c:cc:ccb:[max(_)]:ccba") == 2
    assert extract_field(env, "c:cc:ccb:[]:ccbc:[]:ccbca") == [None, None, None,
                                                               [1, 2], [1, 2]]
    envp = EnvParser()
    assert extract_field(env, "c:cc:ccb:[]:ccbc:[]:ccbca", envp) == [None, None, None,
                                                                     [1, 2], [1, 2]]


# %%
def test_rebuild_dict():
    env = {
        "a": 1,
        "b": 2,
        "c": {
            "ca": "ca",
            "cb": 2,
            "cc": {
                "cca": 1,
                "ccb": [
                    {"ccba": 1, "ccbb": 2, },
                    {"ccba": 2, "ccbb": 4, },
                    {"ccba": 1, "ccbb": 4, },
                    {
                        "ccba": 2,
                        "ccbb": 2,
                        "ccbc": [
                            {"ccbca": 1, "ccbcb": 2, },
                            {"ccbca": 2, "ccbcb": 4, },
                        ],
                    },
                    {
                        "ccba": 1,
                        "ccbb": 1,
                        "ccbc": [
                            {"ccbca": 1, "ccbcb": 2, },
                            {"ccbca": 2, "ccbcb": 4, },
                        ],
                    },
                ],
            },
        },
    }
    rules = {
        ("a", None, "a", int),
        ("ca", None, "c:ca", int),
        ("ccb_count", None, "c:cc:ccb:[count(_)]", int),
        ("ccba", None, "c:cc:ccb:[]:ccba", list),
        ("ccbca", None, "c:cc:ccb:[]:ccbc:[]:ccbca", list),
    }
    rets = rebuild_dict(env, rules)
    envp = EnvParser()
    rets_p = rebuild_dict(env, rules, envp)
    assert rets == rets_p
