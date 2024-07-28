#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_fliper.py
#   Author: xyy15926
#   Created: 2023-12-18 19:42:15
#   Updated: 2024-07-28 20:00:52
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

import numpy as np
from datetime import date, time
from flagbear.patterns import REGEX_TOKEN_SPECS, LEX_ENDFLAG
from flagbear.lex import Lexer
from flagbear.parser import EnvParser
from flagbear.fliper import regex_caster, extract_field, rebuild_dict
from flagbear.fliper import rename_duplicated, rename_overlaped
from flagbear.fliper import str_caster


# %%
def test_rename_duplicated():
    ll = [1, 2, 3, 3, 2, 4, 5]
    nodup = rename_duplicated(ll)
    assert np.all([x == y if isinstance(y, int) else 1
                   for x, y in zip(ll, nodup)])

    unil = [1, 2, 3, 4, 5]
    nodup = rename_overlaped([unil, unil])
    assert np.all([f"{x}_2" == y for x, y in zip(*nodup)])

    nodup = rename_overlaped([unil, unil], ["a", "b"])
    assert np.all([f"{x}_b" == f"{y}" for x, y in zip(*nodup)])


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


# %%
def test_extract_field():
    env = {
        "a": 1,
        "b": 2,
        "c": {
            "ca": "ca",
            "cb": "2",
            "cc": {
                "cca": 1,
                "ccb": [
                    {"ccba": 1, "ccbb": 2, },
                    {"ccba": 2, "ccbb": 4, },
                    {"ccba": 1, "ccbb": 4, },
                    {
                        "ccba": "2",
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
    assert extract_field(env, "c:ca", dtype="INT") == "ca"
    assert extract_field(env, "c:cb", dtype="INT") == 2
    assert extract_field(env, "c:cb") == "2"
    assert extract_field(env, "c:cb", dtype="AUTO") == 2

    assert extract_field(env, "c:cc:cca") == 1
    assert extract_field(env, "c:cc:ccb:[count(_)]") == 5
    assert extract_field(env, "c:cc:ccb:[]:ccba") == [1, 2, 1, "2", 1]
    assert extract_field(env, "c:cc:ccb:[]:ccba", dtype="INT") == [1, 2, 1, 2, 1]
    assert extract_field(env, "c:cc:ccb:[]:ccba", dtype="AUTO") == [1, 2, 1, 2, 1]

    with pytest.raises(TypeError):
        assert extract_field(env, "c:cc:ccb:[max(_)]:ccba") == 2
    assert extract_field(env, "c:cc:ccb:[max(_)]:ccba", dtype="AUTO") == 2
    assert extract_field(env, "c:cc:ccb:[]:ccbc:[]:ccbca") == [None, None, None,
                                                               [1, 2], [1, 2]]
    envp = EnvParser()
    assert extract_field(env, "c:cc:ccb:[]:ccbc:[]:ccbca", envp) == [None, None, None,
                                                                     [1, 2], [1, 2]]


# %%
def test_extract_field_with_forced_dtype():
    env = {
        "a": 1,
        "b": 2,
        "c": {
            "ca": "ca2",
            "cb": "2",
            "cc": {
                "cca": 1,
                "ccb": [
                    {"ccba": 1, "ccbb": 2, },
                    {"ccba": 2, "ccbb": 4, },
                    {"ccba": 1, "ccbb": 4, },
                    {
                        "ccba": "2",
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
    envp = EnvParser()
    assert extract_field(env, "c:cb", envp=envp) == "2"
    assert extract_field(env, "c:cb") == "2"
    assert extract_field(env, "c:cb", dtype="INT") == 2
    assert extract_field(env, "c:ca", dtype="INT") == "ca2"
    assert np.isnan(extract_field(env, "c:ca", dtype="INT", dforced=True))
    assert extract_field(env, "c:ca", dtype="INT", dforced=True,
                         dfill=1234) == 1234
    assert extract_field(env, "c:ca", dtype="INT2", dforced=True) == "ca2"

    regex_specs = REGEX_TOKEN_SPECS.copy()
    regex_specs["INT"] = regex_specs["INT"][:2] + (1234,)
    assert extract_field(env, "c:ca", dtype="INT", dforced=True,
                         regex_specs=regex_specs) == 1234


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
                        "ccba": "2",
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
    rules = [
        ("a"            , None      , "a"                           , "INT"),
        ("ca"           , None      , "c:ca"                        , "INT"),
        ("ccb_count"    , None      , "c:cc:ccb:[count(_)]"         , "INT"),
        ("ccba"         , None      , "c:cc:ccb:[]:ccba"            , "INT"),
        ("ccbca"        , None      , "c:cc:ccb:[]:ccbc:[]:ccbca"   , "INT"),
    ]
    rets = rebuild_dict(env, rules)
    envp = EnvParser()
    rets_p = rebuild_dict(env, rules, envp)
    assert rets == rets_p

    rules = [
        ("a"            , None      , "[_]:a"                           , "INT"),
        ("ca"           , None      , "[_]:c:ca"                        , "INT"),
        ("ccb_count"    , None      , "[_]:c:cc:ccb:[count(_)]"         , "INT"),
        ("ccba"         , None      , "[_]:c:cc:ccb:[]:ccba"            , "INT"),
        ("ccbca"        , None      , "[_]:c:cc:ccb:[]:ccbc:[]:ccbca"   , "INT"),
    ]
    rets = rebuild_dict([env, env], rules)
    envp = EnvParser()
    rets_p = rebuild_dict([env, env], rules, envp)
    assert rets == rets_p


# %%
def test_rebuild_dict_with_forced_dtype():
    env = {
        "a": 1,
        "b": 2,
        "c": {
            "ca": "ca2",
            "cb": "2",
            "cc": {
                "cca": 1,
                "ccb": [
                    {"ccba": 1, "ccbb": 2, },
                    {"ccba": 2, "ccbb": 4, },
                    {"ccba": 1, "ccbb": 4, },
                    {
                        "ccba": "2",
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
    rules = [
        ("a"    , None  , "a"       , "INT"),
        ("ca"   , None  , "c:ca"    , "INT"),
        ("ca_2" , None  , "c:ca"    , "INT" , np.nan, 1),
        ("caf"  , None  , "c:ca"    , "INT" , 0),
    ]
    rets = rebuild_dict(env, rules)
    assert rets["a"] == 1
    assert rets["ca"] == "ca2"
    assert np.isnan(rets["ca_2"])
    assert rets["caf"] == 0


# %%
def test_rebuild_dict_with_forced_dtype_element_wise():
    rec = {
        "int": ["", ""],
        "varchar": ["", ""],
        "date": ["2024-05", "2024-06"]
    }
    rules = [
        ("int", None, "int", "INT", np.nan),
        ("varchar", "varchar", "VARCHAR(255)"),
        ("date", None, "date", "DATE", np.datetime64("NaT")),
    ]
    rets = rebuild_dict(rec, rules)

    rules_element_wise = [
        ("int", None, "int:[_]", "INT", np.nan),
        ("varchar", "varchar:[_]", "VARCHAR(255)"),
        ("date", None, "date:[_]", "DATE", np.datetime64("NaT")),
    ]
    element_wise = rebuild_dict(rec, rules_element_wise)

    assert rets["int"] != element_wise["int"]
    assert rets["varchar"] == element_wise["varchar"]
    assert rets["date"] != element_wise["date"]
