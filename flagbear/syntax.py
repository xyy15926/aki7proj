#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: syntax.py
#   Author: xyy15926
#   Created: 2023-11-29 20:20:12
#   Updated: 2023-12-03 21:28:56
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations

import copy
from collections import deque, namedtuple
from sys import maxsize as MAXINT
from typing import Any, NamedTuple
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from flagbear.graph import backward_update_traverse
from flagbear.patterns import ENDFLAG

from IPython.core.debugger import set_trace


# %%
class Production(NamedTuple):
    lp: str
    rp: tuple


class ASTNode(NamedTuple):
    type: str | Production
    children: list


class LRItem:
    def __init__(self, production: Production, cur: int = 0):
        # Attention: never change public `expr` got from rules.
        self.production = production
        self.lp = production.lp
        self.rp = production.rp
        self.len = len(self.rp)

        # These attributes should be modified after `clone` in most time.
        self.cur = cur
        self.nsym = self._next_sym()

        # These attributes will be modified later according to the production
        # alone.
        self.store = None
        self.start = MAXINT
        self.index = MAXINT

    def clone_next(self):
        # Shallow copy works fine for most attributes.
        next_lr = copy.copy(self)
        next_lr.cur += 1
        if next_lr.cur > next_lr.len:
            raise Exception(f"End of the production {self}.")
        next_lr.index += 1
        next_lr.nsym = next_lr._next_sym()

        return next_lr

    def register_all(self, lr_list: list) -> None:
        # Only the first LRItem of the production can register the LRItems
        # derived from `lr_list`.
        assert self.cur == 0

        # Set the list storing all LRItems.
        self.start = self.index = len(lr_list)
        self.store = lr_list
        lr_list.append(self)

        # Register the following LRItems.
        for i in range(self.len):
            self = self.clone_next()
            lr_list.append(self)

    def _next_sym(self) -> str:
        if self.cur == self.len:
            return None
        return self.rp[self.cur]

    def __repr__(self):
        expr = " ".join([*self.rp[: self.cur], ".", *self.rp[self.cur :]])
        return f"{self.lp} -> {expr}"

    def __iter__(self):
        if self.store is None:
            raise StopIteration
        return iter(self.store[self.index : self.start + self.len + 1])

    def __next__(self):
        if self.store is None:
            return None
        if self.cur == self.len:
            return None
        return self.store[self.index + 1]

    def __eq__(self, rhs):
        return (
            isinstance(rhs, self.__class__)
            and self.store is rhs.store
            and self.index == rhs.index
        )

    # `__hash__`'s implements depends on the `store`, which helps reduce
    # the process to index `LRItem` a lot.
    def __hash__(self):
        if self.store is None:
            raise TypeError(f"unhashable type: unregistered {self.__class__}")
        return self.index


# %%
class LRState:
    def __init__(self, syntax: Syntax, cores: list):
        self.syntax = syntax
        self.cores = tuple(cores)
        self.store = None
        self.index = None
        self.closure = None

    # This `init` method be seperated from `__init__`, or `self`
    # will be append to `state_store` to early.
    def init(self):
        self.store = self.syntax.states
        self.store.append(self)
        self.index = len(self.store) - 1
        self.closure = self.syntax.LR0_closure(self.cores)

    def __repr__(self):
        if self.closure is not None:
            cs = "\n".join([repr(ele) for ele in self.closure])
            return f"I{self.index}\n{cs}"
        else:
            cs = "\n".join([repr(ele) for ele in self.cores])
            return f"Uninited State\n{cs}"

    def __iter__(self):
        if self.closure is None:
            raise StopIteration
        return iter(self.closure)

    def __eq__(self, rhs):
        return (
            isinstance(rhs, self.__class__)
            and self.syntax is rhs.syntax
            and self.cores == rhs.cores
        )

    def __hash__(self):
        # if self.store is None:
        #     raise TypeError(f"unhashable type: unregistered {self.__class__}")
        # return self.index
        return hash(self.cores)


# %%
class Syntax:
    def __init__(
        self, productions: list,
        precedences: str | None = None,
        start_sym: str = "S",
        end_flag: str = ENDFLAG,
    ):
        self.start = start_sym
        self.end = end_flag
        self.productions = []               # [Productions, ]
        self.lr_items = []                  # [LRItems, ]
        self.terminals = set()              # {terminal, }
        self.nonterms = {}                  # {nonterm: [LRItem for reduction, ]}
        self.symbols = {}                   # {symbol: [LRItem containing symbol, ]}
        self.init_productions(productions)
        self.precedences = {}               # {term | Production: (int, "R" | "L") }
        self.nullables = set()              # {nullable nonterm, }
        self.find_nullables()

        # These will be set properly with its methods later.
        self.firsts = {}            # {symbol: [term, ]}
        self.follows = {}           # {LRItem: [term, ]}
        self.lalrs = {}             # {symbol | LRItem: [term, ]}
        self.states = []            # [LRState, ]
        self.gotos = {}             # {(LRState, term): LRItem | LRState}
                                    # LRItem for reduction
                                    # LRState for transition
        self.conflicts = {}

    def init_productions(self, prod_L: list):
        productions = self.productions
        lr_items = self.lr_items
        terminals = self.terminals
        nonterms = self.nonterms
        symbols = self.symbols

        for idx, prod in enumerate(prod_L):
            lp = prod.lp
            lri_fst = LRItem(prod)
            # `copy.copy` is needed to keep this attribute.
            lri_fst.prod_idx = len(productions)
            productions.append(prod)
            # Register all LRItems derived from the same production in Syntax.
            lri_fst.register_all(lr_items)
            # Build index of symbols in LRItems.
            for lri in lri_fst:
                nsym = lri.nsym
                if nsym is None:
                    continue
                symbols.setdefault(nsym, []).append(lri)
            # Append first LRItem, as the reduction LRItem, to nonterms additionally.
            nonterms.setdefault(lp, []).append(lri_fst)
        # Get terminals
        terminals.update([key for key in symbols if key not in nonterms])

    # TODO
    def prec(self, lri: LRItem):
        precedences = self.precedences
        # Set LRItem's precedence with the last terminal in LRItem.
        # In fact, `lri.nsym` must be terminal in most time.
        for item in [lri.production, *lri.rp[:lri.cur % lri.len - lri.len - 1:-1]]:
            prec = precedences.get(item, None)
            if prec is not None:
                return prec
        # Shift first for `R`, known as rightmost, in `LR`.
        return -1, "R"

    def find_nullables(self):
        nullables = self.nullables
        nonterms = self.nonterms

        # Init direct-nullable non-terminals.
        for sym, lris in nonterms.items():
            for lri in lris:
                if lri.len == 0:
                    nullables.add(sym)
                    break

        new_nulls = True
        # Pop nullable from `nullables_newly`.
        while new_nulls:
            new_nulls = False
            for sym, lris in nonterms.items():
                for lri in lris:
                    if nullables.issuperset(lri.rp):
                        nullables.add(sym)
                        new_nulls = True
                        break

    def digraph_first(self):
        symbols = self.symbols
        firsts = self.firsts

        sym_ST = []
        mark_D = dict.fromkeys(symbols.keys(), 0)
        nonterms_L = list(symbols.keys())
        for sym in nonterms_L:
            if mark_D.get(sym, MAXINT) == 0:
                backward_update_traverse(
                    sym,
                    nonterms_L,
                    sym_ST,
                    mark_D,
                    self._first_nexts,
                    self._direct_first,
                    lambda x, y: x.update(y),
                    firsts,
                )

    def _direct_first(self, sym: str) -> set:
        terminals = self.terminals
        if sym in terminals:
            return {sym}
        nonterms = self.nonterms
        first = set()
        for lri in nonterms[sym]:
            nsym = lri.nsym
            if nsym in terminals:
                first.add(nsym)
        return first

    def _first_nexts(self, sym: str) -> set:
        nonterms = self.nonterms
        nullables = self.nullables
        nexts = set()
        if sym not in nonterms:
            return nexts
        for lri in nonterms[sym]:
            nsym = lri.nsym
            while lri.cur < lri.len and nsym in nullables:
                nexts.add(nsym)
                lri = next(lri)
                nsym = lri.nsym
            # In case getting to the last LRItem of the production.
            if nsym is not None:
                nexts.add(nsym)
        return nexts

    def digraph_follow(self):
        symbols = self.symbols
        lr_items = self.lr_items
        follows = self.follows

        sym_ST = []
        mark_D = {}
        for lri in lr_items:
            # Follows won't be set for reduction LRItems.
            if lri.cur < lri.len:
                mark_D[lri] = 0
        syms_L = list(symbols.keys())
        for lri in mark_D:
            if mark_D.get(lri, MAXINT) == 0:
                backward_update_traverse(
                    lri,
                    syms_L,
                    sym_ST,
                    mark_D,
                    self._follow_next,
                    self._follow_read,
                    lambda x, y: x.update(y),
                    follows,
                )

    def _follow_read(self, lri: LRItem) -> set:
        nullables = self.nullables
        firsts = self.firsts
        reads = set()

        # if lri.nsym in terminals:
        #     return reads

        if lri.lp == self.start and lri.cur + 1 == lri.len:
            reads.add(self.end)

        nsym = next(lri).nsym
        while lri.cur < lri.len and nsym in nullables:
            reads.update(firsts[nsym])
            lri = next(lri)
            nsym = lri.nsym
        if nsym is not None:
            reads.update(firsts[nsym])

        return reads

    def _follow_next(self, lri: str) -> set:
        symbols = self.symbols
        nullables = self.nullables
        nexts = set()

        if nullables.issuperset(lri.rp[lri.cur + 1 :]) and lri.lp != self.start:
            for lri in symbols[lri.lp]:
                nexts.add(lri)

        return nexts

    def lookahead(self):
        lr_items = self.lr_items
        follows = self.follows
        lalrs = self.lalrs

        for lri in lr_items:
            if lri.cur + 1 == lri.len:
                lalrs.setdefault(next(lri), set()).update(follows[lri])
        lalrs.update(self.firsts)

    # TODO
    def prefer(self, lri1, lri2):
        """
        Description:
        1. Prefer larger precedence.
        2. Prefer reduction if shift is left-associated.
        """
        prec1, assoc1 = self.prec(lri1)
        prec2, assoc2 = self.prec(lri2)
        if prec1 > prec2:
            return lri1
        elif prec2 > prec1:
            return lri2
        elif lri2.nsym is None and lri1.nsym is not None and assoc1 == "L":
            return lri2
        else:
            return lri1

    # TODO
    def LALR_states_and_gotos(self):
        """
        Description:
        Register all the states and construct gotos of all symbols, both
        temrinals and nonterms, for all the states.
        1. Loop on all the LRItems in 
        """
        nonterms = self.nonterms
        lalrs = self.lalrs
        gotos = self.gotos
        conficts = self.conflicts

        start_state = LRState(self, nonterms[self.start])
        start_state.init()
        self.start_state = start_state
        states_Q = deque([start_state,])
        states_D = {}       # {state: state}: for hashing comparing

        while states_Q:
            state = states_Q.popleft()
            closure = state.closure
            core_lris = {}  # {sym: effective lr-items in LR(0) transition}
            goto_lris = {}  # {term: effective lr-item in LALR(0) transition}
            # Check every LRItem in state.
            for lri in closure:
                nsym = lri.nsym
                # Add next LRItem to `cores_lris`.
                if nsym is not None:
                    core_lris.setdefault(nsym, []).append(next(lri))
                # Skip setting `goto_lris` for nonterms.
                if nsym in nonterms:
                    continue

                las = lalrs[lri] if nsym is None else lalrs[nsym]
                for term in las:
                    # But `goto_lri` stores LRItem in current state.
                    goto_lri = goto_lris.get(term, None)
                    if goto_lri is None:
                        goto_lris[term] = lri
                    # Record conflicts.
                    # According to the implementation, `self.prefer` the first LRItem.
                    else:
                        cf = conficts.setdefault((state, term), set())
                        cf.add(goto_lri)
                        cf.add(lri)
                        goto_lris[term] = self.prefer(lri, goto_lri)
            # Get existing states or add new states for goto-states.
            for sym, cores in core_lris.items():
                new_state = LRState(self, cores)
                if new_state in states_D:
                    new_state = states_D[new_state]
                # Add new state if not exists.
                else:
                    new_state.init()
                    states_Q.append(new_state)
                    states_D[new_state] = new_state
                core_lris[sym] = new_state
                # Set nonterms here.
                if sym in nonterms:
                    gotos[(state, sym)] = new_state
            # Set `gotos` for each state and terminal pair.
            for term, lri in goto_lris.items():
                nsym = lri.nsym
                if nsym is not None:
                    gotos[(state, term)] = core_lris[nsym]
                # For reduce
                else:
                    gotos[(state, term)] = lri

    def LR0_closure(self, lr_items: list) -> list:
        nonterms = self.nonterms
        closure = [*lr_items, ]
        nsym_Q = deque([lri.nsym for lri in closure])
        nsym_S = set()          # Record handled symbols

        while nsym_Q:
            next_sym = nsym_Q.popleft()
            if next_sym in nsym_S:
                continue
            nsym_S.add(next_sym)
            sym_lris = nonterms.get(next_sym, [])
            closure.extend(sym_lris)
            nsym_Q.extend([lri.nsym for lri in sym_lris])

        return closure

    def reduce_tokens(self, tokens: list, reduce_F: callable = ASTNode):
        """
        Description:
        Reduce tokens.

        Params:
        tokens:
        reudce_F: callable, accepts production and tokens to reduce.
        """
        gotos = self.gotos
        state_ST = [self.start_state]
        reduction = []
        # `latok.type` must exists.
        for latok in tokens:
            nlr = gotos[(state_ST[-1], latok.type)]
            while not isinstance(nlr, LRState):
                lp, rp = nlr.lp, nlr.rp
                children = []
                for _ in rp:
                    state_ST.pop()
                    children.append(reduction.pop())
                children = children[::-1]
                reduction.append(reduce_F(nlr.production, children))
                # Stop parsing.
                if lp == self.start:
                    return reduction[0]
                state_ST.append(gotos[(state_ST[-1], lp)])
                nlr = gotos[(state_ST[-1], latok.type)]
            state_ST.append(nlr)
            reduction.append(reduce_F("Token", latok))
