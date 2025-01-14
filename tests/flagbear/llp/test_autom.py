#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_autom.py
#   Author: xyy15926
#   Created: 2024-08-12 11:43:52
#   Updated: 2025-01-13 18:15:56
#   Description:
# ---------------------------------------------------------

# %%
import pytest
if __name__ == "__main__":
    from importlib import reload
    from flagbear.llp import autom
    reload(autom)

import logging
import numpy as np
import pandas as pd
from flagbear.llp.autom import AutomState, Automaton
from flagbear.llp.autom import StatesPDA


# %%
def test_Automaton(caplog):
    at = Automaton()
    sd = {}
    for ele in "AB":
        state = AutomState(ele)
        sd[ele] = state
        at.add_state(state)
    cst = at.add_state("C")
    sd["C"] = cst
    assert cst == AutomState("C")
    ccst = at.adddefault_state("C")
    assert ccst is cst
    ccst = at.adddefault_state(AutomState("C"))
    assert ccst is cst
    dst = at.adddefault_state("D")
    sd["D"] = dst
    assert dst == AutomState("D")

    # AutomState equality.
    for ele in "ABCD":
        assert AutomState(ele) == sd[ele]

    # Add automaton transition.
    tss = [
        ("A", "B", "C", "D"),
        ("B", "B", "C", "D"),
        ("C", "A", "C", "D"),
        ("D", "D", "B", "D"),
    ]
    for x, from_ in enumerate("ABCD"):
        for y, inp in enumerate("abcd"):
            at.add_transition(sd[from_], inp, sd[tss[x][y]])
    with caplog.at_level(logging.WARNING):
        at.add_transition(sd["A"], "b", sd["B"])
    at.start_state = sd["A"]
    at.end_states = {sd["D"]}

    assert at.start().cur == sd["A"]
    assert at.input("c").cur == sd["C"]
    assert at.input("b").cur == AutomState("A")
    assert not at.done()
    assert at.input("d").cur == sd["D"]
    assert at.done()

    tsdf = [[repr(sd[i]) for i in j] for j in tss]
    gotos = pd.DataFrame(tsdf, index=[repr(ele) for ele in sd.values()],
                         columns=list("abcd"))
    assert np.all(at.gotodf() == gotos)


# %%
def test_StatesPDA():
    pda = StatesPDA()
    sd = {}
    for ele in "ABCD":
        state = AutomState(ele)
        sd[ele] = state
        pda.add_state(state)
    # Add automaton transition.
    tss = [
        ("A", "B", "C", "D"),
        ("B", "B", "C", "D"),
        ("C", "A", "C", "D"),
        ("D", "D", "B", "D"),
    ]
    for x, from_ in enumerate("ABCD"):
        for y, inp in enumerate("abcd"):
            pda.add_transition(sd[from_], inp, sd[tss[x][y]])
    pda.start_state = sd["A"]
    pda.end_states = {sd["D"]}

    pda.start()
    assert len(pda.states_stack) == 1
    pda.input("a").input("b").input("c").input("b")
    assert len(pda.states_stack) == 5
    assert pda.cur == AutomState("A")
    assert not pda.done()
    tmp_state = pda.states_stack[-4]
    outp = pda.revert(3)
    assert outp[-1] == AutomState("A")
    assert pda.cur == tmp_state

    tsdf = [[repr(sd[i]) for i in j] for j in tss]
    gotos = pd.DataFrame(tsdf, index=[repr(ele) for ele in sd.values()],
                         columns=list("abcd"))
    assert np.all(pda.gotodf() == gotos)
