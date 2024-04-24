#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_lex.py
#   Author: xyy15926
#   Created: 2023-12-12 12:35:39
#   Updated: 2023-12-18 20:07:49
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear import lex
    from flagbear import patterns
    reload(lex)
    reload(patterns)

from flagbear.lex import Token, Lexer, LEX_ENDFLAG


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
