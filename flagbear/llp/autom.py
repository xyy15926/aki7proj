#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: autom.py
#   Author: xyy15926
#   Created: 2024-08-12 10:06:56
#   Updated: 2025-01-13 18:17:25
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, Any, List, TYPE_CHECKING
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from collections.abc import Iterator, Callable, Mapping, Hashable
if TYPE_CHECKING:
    import pandas as pd
# from IPython.core.debugger import set_trace

import copy
import logging
from collections import deque
from flagbear.const.tokens import LEX_ENDFLAG

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class AutomState:
    """Automaton State.

    Attrs:
    ---------------------------
    core: Hashable.
      Identity to identify the state int the Automaton.
      State with the same core will be treated as equal.
    _hashval: Int.
      Hash value of the state and hash value of the core will be used as
      default.
    autom: Automaton.
      The automaton the state belongs to.
    _regid: Int.
      Register id of the state in the Automaton.
    desc: Str.
      Description for repr.
    """
    def __init__(self, core: Hashable, desc: str = None):
        """Init.

        Params:
        -----------------------
        cores: Hashable to identify the state in Automaton.
        """
        self.core = core                # States with the same core are equal.
        self._hashval = hash(core)      # Hash value for hashable implementation.
        self.autom = None               # Automaton that the state belongs to.
        self._regid = None              # For repr and fast-comparion only.
        self.desc = (f"Unregesiterd State {self._hashval}" if desc is None
                     else desc)

    def __repr__(self):
        """Representation."""
        return self.desc

    def __str__(self):
        """String."""
        return f"AutomState: {self.core}."

    def __eq__(self, rhs: Self):
        """Equal comparison.

        `__eq__` is implemented to check if current state is replicated, along
        with `__hash__` is most cases.
        1. State with the same core will be treated as equal.
        2. But for time-efficiency, registered states will be compared with their
          automaton and register-id first.
        """
        return (
            isinstance(rhs, self.__class__)
            and (
                (self.autom is not None
                 and self.autom is rhs.autom
                 and self._regid == rhs._regid)
                or (self._hashval == rhs._hashval
                    and self.core == self.core)
            )
        )

    def __hash__(self):
        """Hash.

        1. `_hashval` stores the hash value of the state and will be returned
          directly for time-efficiency.
        """
        return self._hashval


# %%
class Automaton:
    """Automaton.

    1. Allow only one start-state but multiple end-states.
    2. A dict will be used to store the states for the convinience to get
      the existed state with only core-State.

    Attrs:
    --------------------------
    state_type: State Class.
      State type(class) of the automaton.
    states_store: Dict[AutomState, AutomState].
      Dict of AutomState to AutomState to store the states in the automaton.
    gotos: Dict[(AutomState, HashableInput), AutomState]
      Dict stores the transition of the Automaton.
    cur: AutomState.
      Current state.
    start_state: AutomState.
      Start state.
    end_state: Set of AutomState.
      End states.
    """
    def __init__(self, state_type: type = AutomState):
        """Init"""
        self.state_type = state_type        # Default state class to init implicitly.
        self.states_store = {}              # {AutomState: AutomState}
        self.states_list = []               # [AutomState]
        self.gotos = {}                     # {(AutomState, HashableInput): AutomState}
        self.cur = None                     # Current state.
        self.start_state = None             # Start state.
        self.end_states = None              # End state set.

    def add_state(self, state: Hashable) -> AutomState:
        """Add state in the automaton.

        Params:
        --------------------------
        state: AutomState or core to be added in the automaton.

        Raises:
        --------------------------
        ValueError

        Return:
        --------------------------
        Regsiter id of the added AutomState.
        """
        store = self.states_store
        slist = self.states_list
        InnerState = self.state_type
        if not isinstance(state, AutomState):
            state = InnerState(state)
        if state in store:
            raise ValueError(f"{state} already exists.")
        state._regid = len(store)
        state.desc = f"S[{state._regid:03}]"
        state.autom = self
        store[state] = state
        slist.append(state)
        return state

    def get_state(self, state: Hashable) -> AutomState | None:
        """Get state from the automaton with core-only state.

        Params:
        --------------------------
        state: Core or unregistered state.

        Return:
        --------------------------
        Registered AutomState in the automaton if core exists.
        Else None.
        """
        InnerState = self.state_type
        if not isinstance(state, AutomState):
            state = InnerState(state)
        return self.states_store.get(state)

    def adddefault_state(self, state: Hashable) -> AutomState:
        """Get state from or add state to the automaton.

        1. Return the state with the same core if existing.
        2. Else add the state to the automaton.

        Params:
        --------------------------
        state: Core or unregistered state.

        Return:
        --------------------------
        Registered AutomState.
        """
        inner = self.get_state(state)
        if inner is not None:
            return inner
        return self.add_state(state)

    def add_transition(self, from_: AutomState,
                       inp: Hashable,
                       to_: AutomState) -> Self:
        """Add transition in the automaton.

        Params:
        --------------------------
        from_: Source automaton state.
        inp: Any hashable input.
        from_: Target automaton state.

        Return:
        --------------------------
        Self will be return for chain operation.
        """
        gotos = self.gotos
        if from_.autom is not self:
            raise ValueError(f"State {from_} isn't registered in this automaton.")
        if to_.autom is not self:
            raise ValueError(f"State {to_} isn't registered in this automaton.")
        if (from_, inp) in gotos:
            logger.warning(f"Update existing transition {from_} with {inp}.")
        gotos[from_, inp] = to_
        return self

    def start(self, state: AutomState = None) -> Self:
        """Start the automaton.

        Params:
        --------------------------
        state: State the automaton starts with.
          Start state will be used as default.

        Raises:
        --------------------------
        ValueError

        Return:
        --------------------------
        Self will be return for chain operation.
        """
        if self.start_state is None and state is None:
            raise ValueError("Automaton can't start without start-state.")
        if not self.gotos:
            raise ValueError("Automaton can't start without transitions.")
        self.cur = self.start_state if state is None else state
        return self

    def input(self, inp: Hashable) -> Self:
        """Receive the input and transit.

        Params:
        --------------------------
        inp: Hashable input.

        Return:
        --------------------------
        Self will be return for chain operation.
        """
        self.cur = self.gotos[self.cur, inp]
        return self

    def done(self):
        """Check if current state is a end-state.

        Raises:
        --------------------------
        ValueError
        """
        if self.end_states is None:
            raise ValueError("No end-states are set.")
        return True if self.cur in self.end_states else False

    def gotodf(self) -> pd.DataFrame:
        """Generate a DataFrame representing the transitions.

        The index and columns of the DataFrame will be the states and valid
        input of the automaton along with responsible target states.
        """
        import pandas as pd

        gotos = self.gotos
        pdict = {}
        for (from_, inp), to_ in gotos.items():
            subd = pdict.setdefault(inp, {})
            subd[repr(from_)] = repr(to_)
        df = pd.DataFrame.from_dict(pdict).sort_index()

        return df


# %%
class StatesPDA(Automaton):
    """Pushdown DFA with stack stores the states.

    Attrs:
    -------------------------
    states_stack: List.
      History of the states of the automaton.
    """
    def __init__(self, state_type: type = AutomState):
        """Init."""
        super().__init__(state_type)
        self.states_stack = None

    def start(self, states: list[AutomState] = None):
        """Start the automaton.

        Params:
        --------------------------
        states: States the automaton starts with.
          Start state will be used as default.
          States passed in won't be check and will be pushed into states stack.

        Raises:
        --------------------------
        ValueError

        Return:
        --------------------------
        Self will be return for chain operation.
        """
        if self.start_state is None and states is None:
            raise ValueError("Automaton can't start without start-state.")
        if not self.gotos:
            raise ValueError("Automaton can't start without transitions.")
        if not states:
            self.cur = self.start_state
            self.states_stack = [self.cur, ]
        else:
            self.states_stack = list(states)
            self.cur = self.states_stack[-1]
        return self

    def input(self, inp: Hashable) -> Self:
        """Receive the input and transit.

        Params:
        --------------------------
        inp: Hashable input.

        Return:
        --------------------------
        Self will be return for chain operation.
        """
        self.cur = self.gotos[self.cur, inp]
        self.states_stack.append(self.cur)
        return self

    def revert(self, stepn: int = 1) -> list[AutomState]:
        """Revert the states of the automaton.

        Reverting automaton is to pop states from stack.

        Params:
        --------------------------
        stepn: The number of steps to revert.

        Return:
        --------------------------
        List of states poped out.
        """
        stack = self.states_stack
        outp = [None] * stepn
        for i in range(stepn):
            outp[stepn - i - 1] = stack.pop()
        self.cur = stack[-1]
        return outp
