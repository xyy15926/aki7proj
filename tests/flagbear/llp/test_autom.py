#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_autom.py
#   Author: xyy15926
#   Created: 2024-08-12 11:43:52
#   Updated: 2024-11-11 11:43:28
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear.llp import autom
    reload(autom)

from flagbear.llp.autom import DFA


# %%
def test_DFA():
    dfa = DFA()
    words = ["cba", "cbaa", "caa"]
    for word in words:
        dfa.add(word)
    for word in words:
        assert dfa.check(word)

    dfa = DFA(words)
    for word in words:
        assert dfa.check(word) == word
