#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#   Name: syntax.py
#   Author: xyy15926
#   Created: 2023-11-29 20:20:12
#   Updated: 2025-01-04 19:17:18
#   Description:
#     PLY Ref: <https://github.com/dabeaz/ply>
#     Lookahead Caculation Ref: <https://dl.acm.org/doi/pdf/10.1145/69622.357187>
# -----------------------------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, Any
try:
    from typing import NamedTuple, Self
except ImportError:
    from typing_extensions import NamedTuple, Self
from collections.abc import Iterator, Callable

# from IPython.core.debugger import set_trace

import copy
import logging
from collections import deque
from sys import maxsize as MAXINT

from flagbear.tree.tree import GeTNode
from flagbear.llp.patterns import LEX_ENDFLAG, SYN_STARTSYM, SYN_ARITH_PRODS
from flagbear.llp.graph import backward_update_traverse

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

# Token should have following 2 attributes at least:
# `type`: The symbol in syntaxer's productions.
# `val` : The value for reduction passed to Production.
Token = TypeVar("Token")
Reduction = TypeVar("Reduction")


# %%
class Production(NamedTuple):
    lp: str                 # Left part
    rp: tuple[str]          # Right part
    reduce: Callable[[list | tuple], Any] | None = None
                            # Callable to produce `lp` with `rp`
    prec: int = 0           # Precedence, prefer the larger
    assoc: str = "R"        # Association, "L" or "R" for left or right first
                            # The order to compute items in `rp` when producing


class LRItem:
    """LR item derived from productions.

    LRItems derived from the same productions share the same production, and
    differ only in `cur` and other related attributes.
    1. LRItems should always be registered in `self.store` so to get other
      LRItems derived from the same productions easily.

    Attrs:
    -----------------------
    production: Production
      Production where the LRItem derives from.
    lp: str
      Left part, `Production.lp`.
    rp: tuple[str]
      Right part, `Production.rp`.
    cur: int
      Current position of the LRItem, representing the parse stage of the
      production.
      Ranging from 0 to `len`:
        0: No symbol in `rp` has been matched yet.
        len: All symbols in `rp` has been matched, namely reducible.
    nsym: str | None
      Next symbol in right part of production.
        1. `rp[cur]` if `cur < len`.
        2. `None` if `cur == len`.
    store: list
      Recording the current LRItem and other LRItem derived from the same
      production in the order of the `cur`.
    start: int
      The index of first LRItem, derived from the same production, in `store`.
    index: int.
      The index of current LRItem in `store`. And `start + cur == index`
      should always be true.
    """
    def __init__(self, production: Production, cur: int = 0):
        """Init LRItem with production.

        Initialize LRItem with production and current position. `cur` won't be
        passed but left default at most of time, since the left LRItem should
        be registered in `store` by calling `register_all`.

        Params:
        -----------------
        production: Production
          Production where the LRItem derives from.
        cur: int
          Current position of the LRItem, representing the parse stage of the
          production.
          Ranging from 0 to len:
            0: No symbol in rp has been matched yet.
            len: All symbols in rp has been matched.
        """
        # Attention: never change public `expr` got from rules.
        self.production = production
        self.lp = production.lp
        self.rp = production.rp

        # These attributes should be modified after `clone` in most time.
        self.cur = cur
        self.nsym = self._next_sym()

        # These attributes will be modified later according to the production
        # alone.
        self.store = None
        self.start = MAXINT
        self.index = MAXINT

    def clone_next(self) -> Self:
        """Clone self to get next LRItem derived from the same production.

        1. Only `cur` and related attributes will be updated, others will keep
          unchanged. So `copy.copy` doing the shallow copy works fine.

        Raise:
        -----------------
        Exception: End of the production.

        Return:
        -----------------
        Next LRItem
        """
        # Shallow copy works fine for most attributes.
        next_lr = copy.copy(self)
        next_lr.cur += 1
        if next_lr.cur > len(next_lr.rp):
            raise Exception(f"End of the production {self}.")
        next_lr.index += 1
        next_lr.nsym = next_lr._next_sym()

        return next_lr

    def register_all(self, lr_list: list[LRItem]) -> None:
        """Register all LRItems.

        Register all LRItems in `lr_list` so to get other LRItems derived from
        the same productions easily.
        1. `lr_list` will be record as the `store` attribute.
        2. Only the first LRItem of the production could register LRItems to
          avoid replication.

        Params:
        ------------------
        lr_list: list[LRItem]
          List will be set as `store` attribute storing all the LRItems.

        Raise:
        ------------------
        AssertError
        """
        # Only the first LRItem of the production can register the LRItems
        # derived from `lr_list`.
        assert self.cur == 0

        # Set the list storing all LRItems.
        self.start = self.index = len(lr_list)
        self.store = lr_list
        lr_list.append(self)

        # Register the following LRItems.
        for i in range(len(self.rp)):
            self = self.clone_next()
            lr_list.append(self)

    def _next_sym(self) -> str | None:
        """Next symbol of LRItem.

        Return:
        ------------------
        next symbol: str | None
          str: rp[cur] if cur < len
          None: if cur == len
        """
        if self.cur == len(self.rp):
            return None
        return self.rp[self.cur]

    def __repr__(self) -> str:
        """Represention."""
        expr = " ".join([*self.rp[: self.cur], ".", *self.rp[self.cur :]])
        return f"{self.lp} -> {expr}"

    def __iter__(self) -> Iterator:
        """Iterate rest LRItems from the current.

        Only registered LRItem is iterable and `store` will be used to
        iterate the rest LRItems derived from the same productions starting
        from current LRItem.
        """
        if self.store is None:
            raise StopIteration
        return iter(self.store[self.index : self.start + len(self.rp) + 1])

    def __next__(self) -> Self | None:
        """Get next LRItem.

        Only registered LRItem can get next LRItem and `store` will be used to
        get the next LRItem derived from the same productions. And None will
        be returned if self is the last one.

        Return:
        --------------------
        next LRItem: LRItem | None
          LRItem: cur < len
          None: cur == len or not registered
        """
        if self.store is None:
            return None
        if self.cur == len(self.rp):
            return None
        return self.store[self.index + 1]

    def __eq__(self, rhs:Self) -> bool:
        """Equal comparison.

        There are two different comparison logics for LRItems registered and
        unregistered.
        1. For registered LRItems, comparing `store` and `index` is enough and
          faster for the short-circuit evaluation.
        2. For unregistered LRItems, comparing `production` and `cur` is
          necessary.
        """
        return (
            isinstance(rhs, self.__class__)
            and ((self.store is not None
                  and self.store is rhs.store
                  and self.index == rhs.index)
                 or (self.production == self.production
                     and self.cur == self.cur))
        )

    def __hash__(self):
        """Hash.

        Hash endures LRItem to be key in dict so to simplify the code.
        1. `index` is used as the hash value as it can't be replicated.
        """
        if self.store is None:
            raise TypeError(f"Unhashable type: unregistered {self.__class__}")
        return self.index

    @staticmethod
    def prefer(lhs: Self, rhs:Self):
        """Get the prefered LRItem in conflict.

        1. Prefer larger precedence.
        2. Prefer reduction if production after shift is left-associated.

        Params:
        -----------------
        lhs, rhs: LRItems to be checked which one is prefered in GOTOs.

        Return:
        -----------------
        Prefered LRItem.
        """
        prec1, assoc1 = lhs.production.prec, lhs.production.assoc
        prec2, assoc2 = rhs.production.prec, rhs.production.assoc
        if prec1 > prec2:
            return lhs
        elif prec2 > prec1:
            return rhs
        # Reduction first when the preferences are the same and the production
        # after shift is left-associated.
        elif rhs.nsym is None and lhs.nsym is not None and assoc1 == "L":
            return rhs
        else:
            return lhs


# %%
class LRState:
    """LR state of LRItems.

    Attrs:
    -------------------
    cores: Tuple[LRItem]
      Tuple of core LRItems, from which all LRItems in LRState are derived.
      This must tuple instead of list to be hashable.
    ext_lris: Dict[str, List[LRItem]]
      Dict of the non-terminals and their first LR0_Item derived from their
      productions, namely extended LRItems.
      This will be used to calculate the LR0_closure for the `cores`.
    closure: List[LRItem]
      List of all LRItems derived from `cores`.
    store: List[LRState]
      Recording all the LRStates in Syntaxer.
    index: int
      The index of current LRState in `store`.
    """
    def __init__(self, cores: list[LRItem] | tuple[LRItem],
                 ext_lris: dict[str, list[LRItem]] = None):
        """Init LRState for only syntaxer and cores.

        Only necessary attributes are set here:
        1. LRState shouldn't be replicated but LRState do need to be
          instantized to check if already existing. So some attributes will
          be set later, especially those may lead to unrevertable impact.
        2. Replicated LRState instance will be dropped instantly, so the
          unnecessary attributes, `closure` for example, may also be set later
          to reduce overhead.

        Params:
        -----------------------
        cores: Tuple or list of core LRItems for current LRState.
        ext_lris: Dict of the non-terminals and their first LR0_Item derived
          from their productions, namely extended LRItems.
        """
        self.cores = tuple(cores)
        self.hash_value = hash(self.cores)
        self.ext_lris = ext_lris
        if self.ext_lris:
            self.closure = self.LR0_closure()
        self.index = None
        self.store = None

    # This `init` method be seperated from `__init__`, or `self`
    # will be append to `state_store` to early.
    def register_this(self, store: list[Self],
                      ext_lris: dict[str, list[LRItem]] = None):
        """Init LRState for rest of the attributes.

        Register this in `store` and set the rest of the attributs.
        1. Self will be add to `store`.
        2. `index`, `store` will be set responsibly.

        Params:
        -----------------------
        store: List to store all the states.
        """
        self.store = store
        if ext_lris is not None:
            self.ext_lris = ext_lris
            self.closure = self.LR0_closure()
        self.index = len(self.store)
        self.store.append(self)

    def LR0_closure(self) -> list[LRItem]:
        """Extend cores LRItems to a closure.

        If the extended LRItems is None, the closure can't be calculated and
        None will be return directly:
        1. Extended LRItems: The first LRItem derived from the productions
          producing the next symbol of LRItem.
        2. Closure: LRItems and all their extended LRItems.

        In process:
        1. A set is used to record symbols already processed.
        2. No check will be done directly on LRItem while appending new LRItem
          to closure, and LRItem will be replicated if called repeatedly.

        Return:
        -----------------------
        LR0 Closure for the cores of the LRState.
        """
        closure = list(self.cores)
        ext_lris = self.ext_lris
        if ext_lris is None:
            return None
        nsym_Q = deque([lri.nsym for lri in closure])
        nsym_recs = set()               # Record handled symbols

        while nsym_Q:
            next_sym = nsym_Q.popleft()
            if next_sym in nsym_recs:
                continue
            nsym_recs.add(next_sym)
            sym_lris = ext_lris.get(next_sym, [])
            closure.extend(sym_lris)
            nsym_Q.extend([lri.nsym for lri in sym_lris])

        return closure

    def __repr__(self):
        """Representation."""
        if self.closure is not None:
            cs = "\n".join([repr(ele) for ele in self.closure])
            return f"I{self.index}\n{cs}"
        else:
            cs = "\n".join([repr(ele) for ele in self.cores])
            return f"Uninited State\n{cs}"

    def __iter__(self) -> Iterator[LRItem]:
        """Iterate LRItem in closure.

        If self is not initialized fully, no LRItem will be iterated.

        Return:
        -----------------
        closure: Iterator[LRItem]
        """
        if self.closure is None:
            raise StopIteration
        return iter(self.closure)

    def __eq__(self, rhs:Self) -> bool:
        """Equal comparison.

        `__eq__` is implemented to check if current state is replicated, along
        with `__hash__`.
        There are two different comparison logics for LRStates fully inited
        and partially inited.
        1. For registered(fully-inited) LRStates, comparing `index` is enough
          and faster.
        2. For partially-inited LRStates, the whole `cores` must be compared.
        """
        return (
            isinstance(rhs, self.__class__)
            and ((self.store is not None
                  and self.store is rhs.store
                  and self.index == rhs.index)
                 or self.cores == rhs.cores)
        )

    def __hash__(self):
        """Hash function.

        `__hash__` is implemented to check if current state is replicated.
        ATTENTION: `cores` instead of `index` is used to calculated hash value
        because LRState do need to be checked if already existing before
        registered.
        """
        return self.hash_value


# %%
# TODO: Pretty print.
# TODO: Recall or reentry of its methods.
# TODO: Extended BNF for variable-length production.
class Syntaxer:
    """Syntaxer implemented based on LALR(1).

    Attrs:
    -------------------------
    start: str
      Start symbol.
    end: str
      End flag marking the end of input terminal stream.
    productions: list[Production | tuple]
      List of productions.
    lr_items: list[LRItem]
      List of LRItems derived from productions.
    terminals: set[str]
      Set of terminal symbols.
    nonterms: dict{str: [LRItem]}
      Dict of non-terminal symbols and their first LRItems of the productions
      producing the symbol.
    symbols: dict{str: [LRItem]}
      Dict of symbols and LRItems containing the symbol.
      So start symbol won't be in `symbols`.
    nullables: set[str]
      Set of nullable symbols.
    start_state: LRState
      Start state when parsing tokens.
    firsts: dict{str: [str]}
      Dict of symbols and list of possible terminals starting the symbol.
    follows: dict{LRItem: [str]}
      Dict of symbols and list of possible terminals following the symbol.
    lalrs: dict{str | LRItem: [str]}
      Dict of reducible LRItems and list of possible terminals following the
      LRItem(or the production).
    states: list[LRState]
      List of LRStates.
    gotos: dict{(LRState, term): LRItem | LRState}
      Dict of (LRState, terminal) and LRItem or LRState recording the
      transitions between LRStates. And LRItems represents reduction should
      be performed.
    conflicts: dict{(LRState, term): [LRItem | LRState, ]}
      Dict of (LRState, terminal) and list of LRItems or LRStates recording
      the conflicts in transitions between LRStates. Namely there are two or
      more transitions or reductions for given LRState and terminal.
    """
    def __init__(
        self, productions: list[Production | tuple] = SYN_ARITH_PRODS,
        start_sym: str = SYN_STARTSYM,
        end_flag: str = LEX_ENDFLAG,
    ):
        """Init Syntaxer with production and flags.

        1. Only productions, along with precedences, will be parsed here, along
          with which responsible attributes will be set.
        2. Also Start symbols and end flag will be set here.

        Params:
        -------------------
        productions: list[Production | tuple]
          List of productions.
        start_sym: str
          Start symbol.
        end_flag: str
          End flag marking the end of input terminal stream.
        """
        self.start = start_sym
        self.end = end_flag
        self.productions = []           # [Production, ]
        self.lr_items = []              # [LRItem, ]
        self.terminals = set()          # {terminal, }
        self.nonterms = {}              # {nonterm: [first LRItem of production, ]}
        self.symbols = {}               # {symbol: [LRItem containing symbol, ]}
        # Init the attributes above.
        self.init_productions(productions)

        # Following attributes will be inited in their responsible method in
        # `init_gotos`.
        self.nullables = set()          # {nullable nonterm, }
        self.start_state = None
        self.firsts = {}                # {symbol: [term, ]}
        self.follows = {}               # {LRItem: [term, ]}
        self.lalrs = {}                 # {LRItem: [term, ]}
        self.states = []                # [LRState, ]
        self.gotos = {}                 # {(LRState, term): LRItem | LRState}
                                        # LRItem for reduction LRState for transition.
        self.conflicts = {}             # {(LRState, term): [LRItem, ]}
                                        # Both Reduce-Reduce and Reduce-shift conflicts.

    def init_productions(self, prods: list[Production | tuple]):
        """Parse productions.

        `lr_items`、`terminals`、`nonterms`、`symbols` and `terminals` will
        be initialized here.
        1. `lr_items` will filled with LRItems by calling the
          `LRItem.register_all`.
        2. `symbols` will be filled while iterate the first LRItem of each
          production.

        Params:
        -------------------
        productions: list[Production | tuple]
          List of productions.
        """
        productions = self.productions
        lr_items = self.lr_items
        terminals = self.terminals
        nonterms = self.nonterms
        symbols = self.symbols

        for prod in prods:
            prod = Production(*prod)
            productions.append(prod)
            lp = prod.lp
            lri_fst = LRItem(prod)
            # Register all LRItems derived from the same production in Syntaxer.
            lri_fst.register_all(lr_items)
            # Build index of symbols in LRItems.
            for lri in lri_fst:
                nsym = lri.nsym
                if nsym is None:
                    continue
                symbols.setdefault(nsym, []).append(lri)
            # Append first LRItem, as the reduction LRItem, to nonterms additionally.
            nonterms.setdefault(lp, []).append(lri_fst)
        # Update terminals
        terminals.update([key for key in symbols if key not in nonterms])

    def init_gotos(self):
        """Init GOTOs.

        Init nullables, Firsts, Follows and GOTOs sequencely.
        """
        self.find_nullables()
        self.digraph_first()
        self.digraph_follow()
        self.LALR_states_and_gotos()

    def find_nullables(self):
        """Find nullable symbols.

        1. Get the nullables defined directly by its null production.
        2. Loop to check whether symbol could be produced totally by nullables.
        """
        nullables = self.nullables
        nonterms = self.nonterms

        # Init direct-nullable non-terminals.
        for sym, lris in nonterms.items():
            # set_trace()
            for lri in lris:
                if len(lri.rp) == 0:
                    nullables.add(sym)
                    break

        new_nulls = True
        # Loop until no more nullable symbols found.
        while new_nulls:
            # set_trace()
            new_nulls = False
            for sym, lris in nonterms.items():
                for lri in lris:
                    if nullables.issuperset(lri.rp) and sym not in nullables:
                        nullables.add(sym)
                        new_nulls = True
                        break

    def digraph_first(self):
        """Compute Firsts for symbols as directed graph vertices.

        First: The terminals that may start the symbols.
        1. First of terminal: Itself.
        2. First of nonterminal: Union of First of symbols starting it defined
          in its productions.

        So, the Firsts of symbols compose a digraph, and
        `backward_update_traverse` will be called to traverse the digraph to
        update the First of each symbol.
        """
        terminals = self.terminals
        nonterms = self.nonterms
        symbols = self.symbols
        nullables = self.nullables
        firsts = self.firsts

        def first_rel(sym):
            if sym in terminals:
                return []
            rel_syms = []
            for lri in nonterms[sym]:
                for nsym in lri.rp:
                    if nsym not in nullables:
                        break
                    rel_syms.append(nsym)
                # `rp` has to be checked if being empty, or `nsym` may not be
                # set.
                if lri.rp and nsym is not None:
                    rel_syms.append(nsym)
            return rel_syms

        sym_ST = []
        sym_L = list(symbols.keys())
        mark_D = dict.fromkeys(sym_L, 0)
        for sym in sym_L:
            if mark_D.get(sym, MAXINT) == 0:
                backward_update_traverse(
                    sym, sym_L, sym_ST, mark_D,
                    lambda x: {x} if x in terminals else set(),
                    first_rel,
                    lambda x, y: x.update(y),
                    firsts,
                )

    def digraph_follow(self):
        """Compute Follows for LRItem as directed graph vertices.

        Read: Terminals that may follow the LRItem according to productions
          that derives the LRItem. Namely the Firsts of the rest symbols of
          LRItem.
        Direct Read: The First of the symbol right after `LRItem.nsym`.
        Follow: Terminals that may follow the LRItem according to all the
          productions. Follow may differs from Read iff all the rest symbols
          in the LRItem are nullable. And the Follow of the LRItem could be
          updated with the Read of the LRItem whose next symbol if the `lp`
          of current LRItem.
        Lookahead: The Follow of reduction LRItem.
        Here, Follow and Read of LRItem could compose a digraph with Direct
        Read as the initial status and `backward_update_traverse` will be
        called to compute the Follows.

        But, does the Follow depend not only the LRItem but also LRState?
        """
        symbols = self.symbols
        nullables = self.nullables
        lr_items = self.lr_items
        firsts = self.firsts
        follows = self.follows

        # Build Follows digraph.
        follow_next_D = {}
        for lri in lr_items:
            nsym = lri.nsym
            # Follow of reduction LRItem is determined by LRItems whose last
            # symbols are reduction's `lp`.
            if nsym is None and lri.lp != self.start:
                follow_next_D[lri] = [next(lri) for lri in symbols[lri.lp]]
            # Follow of transition LRItem is determined by next LRItem iff
            # next symbol is nullable.
            # In fact, all the rest symbols in LRItems could be checked if
            # nullable and put responsible LRItem to `follow_next_D[lri]` to
            # reduce the length of circle in digraph.
            elif nsym in nullables:
                follow_next_D[lri] = [next(lri), ]

        # LRItem's Read as its initial Follow.
        def follow_read(lri):
            if lri.nsym is None:
                if lri.lp == self.start:
                    return {self.end}
                return set()
            return firsts[lri.nsym].copy()

        lri_ST = []
        mark_D = dict.fromkeys(lr_items, 0)
        for lri in lr_items:
            if mark_D.get(lri, MAXINT) == 0:
                backward_update_traverse(
                    lri, lr_items, lri_ST, mark_D,
                    follow_read,
                    lambda x: follow_next_D.get(x, []),
                    lambda x, y: x.update(y),
                    follows,
                )

    def LALR_states_and_gotos(self):
        """Init LALR(1) LRStates and GOTOs.

        1. Set the start LRState.
        2. Init other LRStates starting from start states with a dict to get
          LRStates with the same cores if already inited.
        3. Set GOTOs along with LRStates' initialization with Follows.
        """
        nonterms = self.nonterms
        follows = self.follows
        gotos = self.gotos
        conflicts = self.conflicts
        states = self.states

        start_state = LRState(nonterms[self.start], nonterms)
        start_state.register_this(states)
        self.start_state = start_state
        states_Q = deque([start_state,])
        states_D = {}       # {LRState: LRState}: Store all registered LRState
                            # to get the registered LRState with only cores.

        while states_Q:
            state = states_Q.popleft()
            closure = state.closure
            # This will be closed as the LRState build the LR(0) DFA.
            # {sym: [core LRItems of next LRState with `sym` shift-in]}
            core_lris = {}
            # This records the acceptable terminals and responsible most
            # prefered LRItem, specifying how to shift or reduce, when the
            # terminal is the next symbol
            # In fact, as the FOLLOW are calcuated for each LRItem seperately,
            # this `goto` are LR(1) GOTO in some way, which will be merged for
            # within each LRState.
            # {term: [LRItems of current LRState with `term` as one of their FOLLOW]}
            prefered_lris = {}

            # Check LRItems in state to get the pair of LRItems and terminals
            # so to determine the GOTOs.
            for lri in closure:
                # Prepare cores of next LRStates with `nsym` shift-in.
                nsym = lri.nsym
                if nsym is not None:
                    core_lris.setdefault(nsym, []).append(next(lri))

                # 1. Check and record reduce-reduce and ruduce-shift conficts.
                # 2. Calculate most-prefered LRItem for lookahead symbol,
                #   namemly next acceptable terminals.
                # ATTENTION: Reduction-LRItem and shift-LRItem will all be
                # checked here to calculate the most-prefered LRItem for
                # each lookahead terminal, but it's effective only when
                # reduction-LRItem is the most-prefered LRItem.
                # So it's doesn't matter that non-terminal will be set in
                # `prefered_lris`, as non-terminals are not in `follows`.
                las = [nsym] if nsym is not None else follows[lri]
                for term in las:
                    plri = prefered_lris.setdefault(term, lri)
                    if (plri.nsym != lri.nsym
                        or (plri.nsym is None and lri.nsym is None
                            and plri is not lri)):
                        cf = conflicts.setdefault((state, term), set())
                        cf.update([plri, lri])
                        # Precedences are taken into consideration to choose
                        # from the LRItems with any same terminals as a FOLLOW
                        # for shift or reduction so to resolve the conflicts.
                        prefered_lris[term] = LRItem.prefer(plri, lri)
                # TODO: TRUE conflicts
                # TODO: Pretty prints

            # Transit along the LR(0) DFA to init LRStates with cores from
            # `core_lris` to construct the LR(0) DFA.
            # Only shift transition will be set here in GOTO.
            for sym, cores in core_lris.items():
                new_state = LRState(cores)
                # `new_state` has been assigned with the inited states.
                if new_state in states_D:
                    new_state = states_D[new_state]
                # Add new state if not exists.
                else:
                    new_state.register_this(states, nonterms)
                    states_Q.append(new_state)
                    states_D[new_state] = new_state
                core_lris[sym] = new_state
                # Set GOTOs for symbols here.
                gotos[(state, sym)] = new_state

            # Update reduction transition in GOTO if the reduction-LRItem
            # is prefered for the terminal.
            for term, lri in prefered_lris.items():
                if lri.nsym is None:
                    gotos[(state, term)] = lri

    # TODO: Error handlers for invalid token stream.
    def reduce_tokens(
        self, tokens: list[Token],
        reduce_F: Callable[[Production | Token, Reduction | None], Reduction] = GeTNode,
    ) -> Reduction:
        """Reduce tokens.

        Params:
        ------------------------
        tokens: Tokens to be reduced.
          Tokens should have `type` attribute.
        reudce_F: Callabe to reduce tokens of syntax-sub-tree.
          The default ASTNode will reduce the tokens into a AST.

        Raise:
        ------------------------
        ValueError: Invalid tokens.

        Return:
        ------------------------
        Reduction result.
        """
        gotos = self.gotos
        state_ST = [self.start_state]
        reductions = []
        # `latok.type` must exists.
        for latok in tokens:
            nlr = gotos[(state_ST[-1], latok.type)]
            # Reduction.
            while not isinstance(nlr, LRState):
                lp, rp = nlr.lp, nlr.rp
                children = []

                # Pop states and reductions out according to the production.
                for _ in rp:
                    state_ST.pop()
                    children.append(reductions.pop())
                children = children[::-1]

                reductions.append(reduce_F(nlr.production, children))
                # Stop parsing.
                if lp == self.start:
                    return reductions[0]
                state_ST.append(gotos[(state_ST[-1], lp)])
                nlr = gotos[(state_ST[-1], latok.type)]
            state_ST.append(nlr)
            reductions.append(reduce_F(latok, None))
        raise ValueError(f"Invalid tokens {tokens}.")

    def parse_tokens(self, tokens: list[Token]) -> Reduction:
        """Reduce tokens by calling reduce of productions.

        All `reduce`s of the productions should be set properly.

        Params:
        ------------------------
        tokens: Tokens to be parsed.
          Tokens should have `type` and `val` attribute.

        Return:
        ------------------------
        Reduction result.
        """
        return self.reduce_tokens(tokens,
                                  lambda x, y: x.reduce(y)
                                  if isinstance(x, Production)
                                  else x.val)

    def pprint(self):
        """Print GOTOs and conflicts.

        Return:
        --------------------
        pgoto: dict[from_states, dict[terminal, to_states or reduction]]
          Dict represent the GOTOs matrix of the formation:
            from_state and to_state: "I" + LRState index
            reduction: "R" + production index
        pconflicts: dict[from_states, dict[terminal, productions of conflicts]]
        """
        # Build a dict of LRItem start index and production index.
        prod_idx = 0
        lr_items = self.lr_items
        production_index_D = {}
        for lri in lr_items:
            if lri.cur == 0:
                production_index_D[lri.index] = prod_idx
                prod_idx += 1

        gotos = self.gotos
        pgotos = {}
        for (state, sym), dest in gotos.items():
            from_ = f"I{state.index}"
            to_ = (f"I{dest.index}" if isinstance(dest, LRState)
                   else f"R{production_index_D[dest.start]}")
            pgotos.setdefault(from_, {})[sym] = to_

        pconflicts = {}
        conflicts = self.conflicts
        for (state, sym), dests in conflicts.items():
            from_ = f"I{state.index}"
            to_ = ",".join([f"R{production_index_D[dest.start]}"
                            for dest in dests])
            pconflicts.setdefault(from_, {})[sym] = to_

        return pgotos, pconflicts
