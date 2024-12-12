#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_fliper.py
#   Author: xyy15926
#   Created: 2023-12-18 19:42:15
#   Updated: 2024-12-12 15:33:41
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear.tree import tree
    from flagbear.llp import lex, syntax, parser, patterns, graph
    from flagbear.str2 import dtyper, fliper
    reload(tree)
    reload(graph)
    reload(patterns)
    reload(lex)
    reload(syntax)
    reload(parser)
    reload(dtyper)
    reload(fliper)

import numpy as np
from datetime import date, time
from flagbear.llp.patterns import LEX_ENDFLAG
from flagbear.llp.lex import Lexer
from flagbear.llp.parser import EnvParser
from flagbear.str2.fliper import extract_field, rebuild_dict
from flagbear.str2.fliper import reset_field


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

    assert extract_field(env, "c:ca", dtype="INT",
                         dforced=True,
                         dfill=1234) == 1234


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
    rets = rebuild_dict(env, rules, True)
    envp = EnvParser()
    rets_p = rebuild_dict(env, rules, envp=envp)
    assert rets == rets_p

    rules = [
        ("a"            , None      , "[_]:a"                           , "INT"),
        ("ca"           , None      , "[_]:c:ca"                        , "INT"),
        ("ccb_count"    , None      , "[_]:c:cc:ccb:[count(_)]"         , "INT"),
        ("ccba"         , None      , "[_]:c:cc:ccb:[]:ccba"            , "INT"),
        ("ccbca"        , None      , "[_]:c:cc:ccb:[]:ccbc:[]:ccbca"   , "INT"),
    ]
    rets = rebuild_dict([env, env], rules, True)
    envp = EnvParser()
    rets_p = rebuild_dict([env, env], rules, envp=envp)
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


# %%
def test_reset_field():
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
    reset_field(env, "c:cc:cca", 2)
    assert env["c"]["cc"]["cca"] == 2
    reset_field(env, "c:cc:ccc", 2)
    assert env["c"]["cc"].get("ccc") is None
    reset_field(env, "c:cc:ccb:[_]:ccbc:[_]:ccbca", 300)
    assert extract_field(env, "c:cc:ccb:[_]:ccbc:[_]:ccbca")[-1] == [300, 300]

    def val():
        return 2
    reset_field(env, "c:cc:ccb", val)
    assert env["c"]["cc"]["ccb"] == val()
