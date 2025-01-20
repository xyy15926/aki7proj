#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_grammer.py
#   Author: xyy15926
#   Created: 2023-11-30 09:49:53
#   Updated: 2025-01-14 20:16:26
#   Description:
# ---------------------------------------------------------

# %%
from pytest import mark
if __name__ == "__main__":
    from importlib import reload
    from ubears.flagbear.llp import lex, autom, syntax, parser, graph
    reload(autom)
    reload(lex)
    reload(syntax)
    reload(parser)
    reload(graph)

from ubears.flagbear.const.prods import SYN_ARITH_PRODS, SYN_EXPR_PRODS
from ubears.flagbear.llp.lex import Lexer
from ubears.flagbear.llp.syntax import Production, LRItem, Syntaxer


# %%
def test_LRItem():
    lri = LRItem(Production("S", ()))
    assert lri.nsym is None

    productions = [
        Production("S", ("expr")),
        Production("expr", ("expr", "ADD", "term")),
        Production("expr", ("expr", "SUB", "term")),
        Production("expr", ("term")),
        Production("term", ("term", "MUL", "factor")),
        Production("term", ("term", "DIV", "factor")),
        Production("term", ("factor")),
        Production("factor", ("NUMBER")),
        Production("factor", ("LPAR", "expr", "RPAR")),
    ]
    lri = LRItem(productions[1])
    lri2 = LRItem(productions[2])
    lris = list()
    lri.register_all(lris)
    lri2.register_all(lris)
    assert lri.nsym == "expr"
    assert lri.cur == 0
    assert lri2.start == 4
    assert next(lri2) == lri2.clone_next()
    assert repr(lri) == "expr -> . expr ADD term"


# %%
def test_LRItem_prefer():
    productions = [
        Production("S", ("expr"), None, 0, "L"),
        Production("expr", ("expr", "ADD", "expr"), None, 0, "L"),
        Production("expr", ("expr", "SUB", "expr"), None, 0, "L"),
        Production("expr", ("expr", "MUL", "expr"), None, 1, "L"),
        Production("expr", ("expr", "DIV", "expr"), None, 1, "L"),
        Production("expr", ("NUMBER"), None, 3, "L"),
        Production("expr", ("LPAR", "expr", "RPAR"), None, 2, "L"),
    ]
    lri = LRItem(productions[1])
    lri2 = LRItem(productions[2])
    lri3 = LRItem(productions[3])
    assert LRItem.prefer(lri, lri2) is lri
    assert LRItem.prefer(lri, lri3) is lri3


# %%
def test_Syntaxer_arith():
    syntaxer = Syntaxer(SYN_ARITH_PRODS)
    start, end = syntaxer.start, syntaxer.end

    lr_items = syntaxer.lr_items

    nonterms = syntaxer.nonterms
    assert list(nonterms.keys()) == ["S", "expr"]

    terminals = syntaxer.terminals
    assert terminals == {"INT", "FLOAT", "LPAR", "RPAR",
                         "ADD", "SUB", "MUL", "DIV"}

    syntaxer.find_nullables()
    nullables = syntaxer.nullables
    assert not nullables

    syntaxer.digraph_first()
    firsts = syntaxer.firsts
    assert firsts["expr"] == {"FLOAT", "INT", "LPAR", "SUB"}
    assert firsts["INT"] == {"INT"}
    assert firsts["FLOAT"] == {"FLOAT"}
    assert firsts["LPAR"] == {"LPAR"}
    assert firsts["RPAR"] == {"RPAR"}
    assert firsts["ADD"] == {"ADD"}
    assert firsts["SUB"] == {"SUB"}
    assert firsts["MUL"] == {"MUL"}
    assert firsts["DIV"] == {"DIV"}

    syntaxer.digraph_follow()
    follows = syntaxer.follows
    assert len(follows) == len(lr_items)
    for lri_item, follow in follows.items():
        lp = lri_item.lp
        nsym = lri_item.nsym
        if lp == start and nsym is None:
            assert follow == {end}
        elif nsym == "expr":
            assert follow == {"FLOAT", "INT", "LPAR", "SUB"}
        elif nsym is None:
            assert follow == {"ADD", "SUB", "MUL", "DIV", "RPAR", end}
        else:
            assert follow == {nsym}


# %%
def test_Syntaxer_nullable_arith():
    productions = [
        ("S"        , ("expr", )                    , lambda x: x[0]            , 0     , "R"),
        ("expr"     , ("FLOAT", )                   , lambda x: x[0]            , 0     , "R"),
        ("expr"     , ("SUB", "FLOAT")              , lambda x: -x[1]           , 3     , "R"),
        ("expr"     , ("INT", )                     , lambda x: x[0]            , 0     , "R"),
        ("expr"     , ("SUB", "INT")                , lambda x: -x[1]           , 3     , "R"),
        ("expr"     , ("expr", "ADD", "expr")       , lambda x: x[0] + x[2]     , 1     , "L"),
        ("expr"     , ("expr", "SUB", "expr")       , lambda x: x[0] - x[2]     , 1     , "L"),
        ("expr"     , ("expr", "MUL", "expr")       , lambda x: x[0] * x[2]     , 2     , "L"),
        ("expr"     , ("expr", "DIV", "expr")       , lambda x: x[0] / x[2]     , 2     , "L"),
        ("expr"     , ("LPAR", "expr", "RPAR")      , lambda x: x[1]            , 0     , "R"),
        ("eles"     , ()                            , lambda x: []              , 0     , "L"),
        ("eles"     , ("expr", "COMMA")             , lambda x: [x[0], ]        , 0     , "L"),
        ("eles"     , ("expr", "COMMA", "expr")     , lambda x: [x[0], x[2]]    , 0     , "L"),
        ("eles"     , ("expr", "COMMA", "eles")     , lambda x: [x[0], *x[2]]   , 0     , "L"),
        ("expr"     , ("LBPAR", "eles", "RBPAR")    , lambda x: frozenset(x[1]) , 0     , "L"),
    ]
    syntaxer = Syntaxer(productions)
    start, end = syntaxer.start, syntaxer.end
    lr_items = syntaxer.lr_items

    nonterms = syntaxer.nonterms
    assert set(nonterms.keys()) == {"S", "expr", "eles"}

    terminals = syntaxer.terminals
    assert terminals == {"INT", "FLOAT", "LPAR", "RPAR",
                         "ADD", "SUB", "MUL", "DIV",
                         "COMMA", "LBPAR", "RBPAR"}

    syntaxer.find_nullables()
    nullables = syntaxer.nullables
    assert nullables == {"eles"}

    syntaxer.digraph_first()
    firsts = syntaxer.firsts
    for sym, first in firsts.items():
        if sym in terminals:
            assert first == {sym}
    assert firsts["expr"] == {"FLOAT", "INT", "LPAR", "LBPAR", "SUB"}
    assert firsts["eles"] == {"FLOAT", "INT", "LBPAR", "LPAR", "SUB"}

    syntaxer.digraph_follow()
    follows = syntaxer.follows
    assert len(follows) == len(lr_items)
    for lri_item, follow in follows.items():
        lp = lri_item.lp
        nsym = lri_item.nsym
        if lp == start and nsym is None:
            assert follow == {end}
        elif nsym == "expr":
            assert follow == {"FLOAT", "INT", "LPAR", "LBPAR", "SUB"}
        elif nsym == "eles":
            assert follow == {"FLOAT", "INT", "LPAR", "LBPAR", "RBPAR", "SUB"}


# %%
def test_Syntaxer_expr():
    syntaxer = Syntaxer(SYN_EXPR_PRODS)
    exprs = [
        ("{(2+4)*8+-1000*9+1-1.5, 9, {1, 3}, {}}",
         "{(2+4)*8+-1000*9+1-1.5, 9, frozenset({1, 3}), frozenset({})}"),
        "{(2+4)*8+-1000*9+1-1.5, 9}",
        "{(2+4)*8+-1000+9*22+-1000*9-22-1.5, 9}",
        "[1, 2, 3]",
        "[[], [3,4]]",
        "[1, 2, 3, [], [1,2+3,(2+4)*4]]",
        "1 in {2,3}",
        "1 in {1, 2}",
        "1+3 in {4, 5}",
        "[1, 2, 3][2]",
        "\"abc\" + \"bc\"",
        "\"1\" != 1",
        "[1,]",
        "[1]",
        "--9",
        "not 9",
        "not not 9",
    ]
    lexer = Lexer()
    for expr in exprs:
        if isinstance(expr, str):
            tokens = list(lexer.input(expr))
            assert syntaxer.parse_tokens(tokens) == eval(expr)
        else:
            tokens = list(lexer.input(expr[0]))
            assert syntaxer.parse_tokens(tokens) == eval(expr[1])


# %%
def test_Syntaxer_pprint():
    syntaxer = Syntaxer(SYN_EXPR_PRODS)
    gotodf, cfdf = syntaxer.pprint()
    assert gotodf.shape == cfdf.shape
