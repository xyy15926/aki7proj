#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_grammer.py
#   Author: xyy15926
#   Created: 2023-11-30 09:49:53
#   Updated: 2023-12-04 20:44:55
#   Description:
# ---------------------------------------------------------

# %%
from flagbear import lex, syntax, parser, patterns
if __name__ == "__main__":
    from importlib import reload
    reload(lex)
    reload(syntax)
    reload(parser)
    reload(patterns)

from flagbear.lex import Lexer
from flagbear.syntax import Production, LRItem, Syntax
from flagbear.parser import LRParser


# %%
def test_lex():
    a = """
    var1 = 122
    var2 = var1 + 233
    if var1 > 10
        var2 = var2 + 100
    else
        var2 = Var2 + 10
    endif
    """
    lexer = Lexer()
    for tok in lexer.input(a):
        print(tok)


# %%
def test_LRItem():
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
    assert(len(lris) == lri.len + lri2.len + 2)
    assert(lri.nsym == "expr")
    assert(lri.len == 3)
    assert(lri.cur == 0)
    assert(lri2.start == 4)
    assert(next(lri2) == lri2.clone_next())


# %%
def test_Syntax():
    productions = [
        Production("S", ("expr", )),
        Production("expr", ("expr", "ADD", "expr")),
        Production("expr", ("expr", "SUB", "expr")),
        Production("expr", ("expr", "MUL", "expr")),
        Production("expr", ("expr", "DIV", "expr")),
        Production("expr", ("INTEGER", )),
        Production("expr", ("FLOAT", )),
        Production("expr", ("LPAR", "expr", "RPAR")),
        Production("expr", ("SUB", "INTEGER")),
    ]
    syntaxer = Syntax(productions)
    syntaxer.precedences = {
        "ADD": (1, "L"),
        "SUB": (1, "L"),
        "MUL": (2, "L"),
        "DIV": (2, "L"),
        productions[-1]: (3, "R"),
    }
    syntaxer.find_nullables()
    syntaxer.digraph_first()
    syntaxer.digraph_follow()
    syntaxer.lookahead()
    syntaxer.LALR_states_and_gotos()
    lex = Lexer()
    tokens = list(lex.input("(2+4)*8+-1000*9+1-1"))
    root = syntaxer.reduce_tokens(tokens)


# %%
def test_Parser():
    # productions = [
    #     Production("S", ("expr", ), lambda x: x[0]),
    #     Production("expr", ("expr", "ADD", "expr"), lambda x: x[0] + x[2]),
    #     Production("expr", ("expr", "SUB", "expr"), lambda x: x[0] - x[2]),
    #     Production("expr", ("expr", "MUL", "expr"), lambda x: x[0] * x[2]),
    #     Production("expr", ("expr", "DIV", "expr"), lambda x: x[0] / x[2]),
    #     Production("expr", ("INTEGER", ), lambda x: x[0]),
    #     Production("expr", ("FLOAT", ), lambda x: x[0]),
    #     Production("expr", ("LPAR", "expr", "RPAR"), lambda x: x[1]),
    #     Production("expr", ("SUB", "INTEGER"), lambda x: -x[1]),
    # ]
    parser = LRParser()
    assert(parser.parse("(2+4)*8+-1000*9+1-1.5") == -8952.5)

