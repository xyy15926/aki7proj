#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_grammer.py
#   Author: xyy15926
#   Created: 2023-11-30 09:49:53
#   Updated: 2023-12-18 09:00:36
#   Description:
# ---------------------------------------------------------

# %%
from pytest import mark
if __name__ == "__main__":
    from importlib import reload
    from flagbear import lex, syntax, parser, patterns, graph
    reload(lex)
    reload(syntax)
    reload(parser)
    reload(patterns)
    reload(graph)

from flagbear.lex import Lexer
from flagbear.syntax import Production, LRItem, Syntaxer
from flagbear.patterns import SYN_EXPR_PRODS


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
    assert len(lris) == lri.len + lri2.len + 2
    assert lri.nsym == "expr"
    assert lri.len == 3
    assert lri.cur == 0
    assert lri2.start == 4
    assert next(lri2) == lri2.clone_next()
    assert repr(lri) == "expr -> . expr ADD term"


# %%
def test_Syntaxer_arith():
    syntaxer = Syntaxer()
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

    syntaxer.LALR_states_and_gotos()
    states = syntaxer.states
    gotos = syntaxer.gotos
    conflicts = syntaxer.conflicts
    for state in states:
        nsyms = []
        for lri in state:
            nsym = lri.nsym
            if nsym is not None:
                nsyms.append(nsym)
                dest = gotos[(state, nsym)]
                for nlri in syntaxer.LR0_closure(next(lri)):
                    assert nlri in dest
            else:
                for follow in follows[lri]:
                    dest = gotos[(state, follow)]
                    if follow in nsyms:
                        assert lri in conflicts[(state, follow)]
                    if isinstance(dest, LRItem):
                        assert dest is lri

    arith_expr = "(2+4)*8+-1000*9+1-1.5"
    lexer = Lexer()
    tokens = list(lexer.input(arith_expr))
    assert syntaxer.parse_tokens(tokens) == eval(arith_expr)


# %%
def test_Syntaxer_pprint():
    syntaxer = Syntaxer()
    syntaxer.init_gotos()
    pgotos, pconflicts = syntaxer.pprint()


# %%
def test_Syntaxer_nullable():
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

    syntaxer.LALR_states_and_gotos()
    states = syntaxer.states
    gotos = syntaxer.gotos
    conflicts = syntaxer.conflicts
    for state in states:
        nsyms = []      # next symbols of all LRItems in LRState
        for lri in state:
            nsym = lri.nsym
            if nsym is not None:            # Transition
                nsyms.append(nsym)
                dest = gotos[(state, nsym)]
                for nlri in syntaxer.LR0_closure(next(lri)):
                    assert nlri in dest
            else:                           # Reduction
                for follow in follows[lri]:
                    dest = gotos[(state, follow)]
                    if isinstance(dest, LRItem):
                        assert (dest is lri
                                or (dest in conflicts[(state, follow)]
                                    and lri in conflicts[(state, follow)]))

    arith_expr = "{(2+4)*8+-1000*9+1-1.5, 9}"
    lexer = Lexer()
    tokens = list(lexer.input(arith_expr))
    assert syntaxer.parse_tokens(tokens) == eval(arith_expr)
    arith_expr = "{(2+4)*8+-1000*9+1-1.5, 9, {1, 3}, {}}"
    lexer = Lexer()
    tokens = list(lexer.input(arith_expr))
    assert (syntaxer.parse_tokens(tokens)
            == eval("{(2+4)*8+-1000*9+1-1.5, 9, frozenset({1, 3}), frozenset({})}"))


# %%
def test_Syntaxer_expr():
    syntaxer = Syntaxer(SYN_EXPR_PRODS)
    syntaxer.init_gotos()
    exprs = [
        "{(2+4)*8+-1000*9+1-1.5, 9}",
        "[1, 2, 3]",
        "[[], [3,4]]",
        "[1, 2, 3, [], [1,2+3,(2+4)*4]]",
        "1 in {2,3}",
        "1 in {1, 2}",
        "1+3 in {4, 5}",
        "[1, 2, 3][2]",
        "\"abc\" + \"bc\"",
        "\"1\" != 1",
    ]
    lexer = Lexer()
    for expr in exprs:
        tokens = list(lexer.input(expr))
        assert syntaxer.parse_tokens(tokens) == eval(expr)
