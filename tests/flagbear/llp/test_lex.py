#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_lex.py
#   Author: xyy15926
#   Created: 2023-12-12 12:35:39
#   Updated: 2025-01-14 20:12:20
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear.llp import lex
    reload(lex)

from flagbear.llp.lex import Token, Lexer
from flagbear.const.tokens import LEX_ENDFLAG


# %%
def test_lexer():
    a = """
    var1 = 122
    var2 = var1 + "233xs熊"
    if var1 > 10
        var2 = var2 + 100
    else
        var2 = Var2 + 10
    endif
    """
    lexer = Lexer()
    tokens = list(lexer.input(a))
    assert tokens[-1] == Token(LEX_ENDFLAG, "", -1, -1)
    assert tokens[0] == Token("ID", "var1", 2, 4)
    assert tokens[7] == Token("STRING", "233xs熊", 3, 8)
    assert tokens[8] == Token("IF", "if", 4, 2)


# %%
def test_lexer_compile():
    lexer = Lexer()
    assert lexer.parse("2 + 3") == 5

    env = {
        "a": 2,
        "b": 3,
        "d": "adf",
        "e": [2, 3],
        "f": [4, 5],
    }
    assert lexer.bind_env(env).parse("a + 4") == 6
    assert lexer.bind_env(env).parse("d + \"a\"") == "adfa"
    assert len(lexer.parse("e")) == len(env["e"])
    assert lexer.parse("a @ e")
    assert not lexer.parse("a @ f")
