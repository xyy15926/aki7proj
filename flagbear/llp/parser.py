#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: parser.py
#   Author: xyy15926
#   Created: 2023-12-02 21:04:21
#   Updated: 2025-01-14 09:47:57
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, Any
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from collections import ChainMap
from collections.abc import Callable, Mapping
from IPython.core.debugger import set_trace

import logging
from collections import namedtuple
import copy

from flagbear.tree.tree import GeTNode
from flagbear.llp.lex import Token, Lexer
from flagbear.llp.syntax import Production, LRItem, LRState, Syntaxer
from flagbear.llp.patterns import (
    LEX_TOKEN_SPECS,
    LEX_SKIPS,
    LEX_RESERVEDS,
    LEX_ENDFLAG,
    SYN_STARTSYM,
    SYN_EXPR_PRODS,
    CALLABLE_ENV
)

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")

Reduction = TypeVar("Reduction")


# %%
class EnvParser:
    """Parser with environment to get value of ID.

    Attrs:
    --------------------------
    lexer: Lexer.
      Lexer to split words into tokens.
    syntaxer: Syntaxer.
      Syntaxer to compile tokens into ASTree.
    history: Dict[str, list[Tokens].
      Record the history to accelerate.
    default_env: Dict[ID-str, Any].
      The default environment to get value of ID from, which will be searched
      only after `env` fails.
    env: Dict[ID-str, Any] mostly.
      The customed environment to get the value of ID from.
      This could be updated easily so to change the execution environment,
      which differs from the `default_env`.
    """
    def __init__(
        self, default_env: dict[str, Callable] = CALLABLE_ENV,
        token_specs: dict = LEX_TOKEN_SPECS,
        reserveds: dict = LEX_RESERVEDS,
        skips: set = LEX_SKIPS,
        productions: list[Production | tuple] = SYN_EXPR_PRODS,
        start_sym: str = SYN_STARTSYM,
        end_flag: str = LEX_ENDFLAG,
    ):
        """Init LRParser.

        1. Update the token ID's reduce so to search the ID in the
          `default_env` and `env`.

        Params:
        --------------------------
        default_env: Dict[ID-str, Any].
          The default environment to get value of ID from, which will be searched
          only after `env` fails.
        token_specs: dict of {TOKEN_TYPE: REGEX_PATTERN}
          Token type names and their regex patterns.
          Atttention: Token patterns could override each others, but preceding
            patterns will be detected first for the `re.finditer` feature. So
            do remember to order the tokens carefully.
          Attention: Master regular are constructed with the advantage of named
            group feature in python's re. So only no-captureing version of
            parentheses are allowed in tokens' regexs.
        reserveds: dict of {TOKEN_VALUE: TOKEN_TYPE}
          Special tokens share the same token regex pattern but should be
          treated differently. Tokens detected will be checked and converted to
          reserveds if true.
        skpis: set of {TOKEN_TYPE}
          Special tokens to be skiped.
        productions: list[Production | tuple]
          List of productions or compatiable tuple.
        start_sym: str
          Start symbol.
        end_flag: str
          End flag marking the end of input terminal stream.
          Namely the speical token type name to mark the end of a token stream.
        """
        self.default_env = default_env
        self.env = {}
        # Replace the production of `expr := ID` so to update the reduce to
        # bind environment.
        for idx, prod in enumerate(productions):
            if prod[1] == ("ID",):
                break
        productions = copy.copy(productions)
        productions[idx] = (*prod[:2], self._getID, *prod[3:])
        self.lexer = Lexer(token_specs, reserveds, skips, end_flag)
        self.syntaxer = Syntaxer(productions, start_sym, end_flag)
        self.history = {}

    def bind_env(self, env: dict[str, Any]) -> Self:
        """Bind environment.

        Replace the callable of the token ID with a lambda to get the value
        from new `env`.
        1. `_` represent the `env` itself.
        2. Chain the default env and bound env with ChainMap.

        Params:
        ----------------------
        env: The dict to get the value of ID from.

        Return:
        ----------------------
        self
        """
        self.env = env
        return self

    def _getID(self, id_: str):
        """Get value from inner environment."""
        id_ = id_[0]
        if id_ == "_":
            return self.env
        # Any `env` with `get` method works fine, not only Mapping.
        if hasattr(self.env, "get"):
            ret = self.env.get(id_)
            if ret is None:
                ret = self.default_env.get(id_)
        else:
            ret = self.default_env.get(id_)
        return ret

    def compile(self, words: str) -> list[GeTNode]:
        """Compile input words into post-ordered AST nodes list.

        1. ASTree could be got by calling `syntaxer.reduce_tokens`, and the
          value of node is Token or production.
        2. ASTree is good and fast enough for further executing. But as
          executing is done along with traversing the AST in post-order,
          post-ordered AST with its nodes in a list will be more convenient.

        Params:
        --------------------
        words: Input words.

        Return:
        --------------------
        List of nodes from AST in post-order.
        """
        lexer = self.lexer
        syntaxer = self.syntaxer
        history = self.history
        if words in history:
            return history[words]

        ast = syntaxer.reduce_tokens(lexer.input(words), GeTNode)
        history[words] = ast.post_order()
        return history[words]

    def exec(self, nodes: list[GeTNode]) -> Any:
        """Execute the nodes from post-ordered traversing.

        1. Append value of token node to temperary stack directly.
        2. Reduce part of the values in temperary stack for production node.
          The number of values to pop out is the length of the right part of
          the production, namely the number of children of the node.

        Params:
        --------------------
        nodes: Nodes from AST in post-order.

        Return:
        --------------------
        Execution result.
        """
        tmp_ret = []
        for node in nodes:
            chn = node.chn
            val = node.val
            # Token node -> append directly.
            if isinstance(val, Token):
                tmp_ret.append(val.val)
            # Production node -> reduce then append.
            elif chn == 0:
                tmp_ret.append(val.reduce(None))
            else:
                cret = val.reduce(tmp_ret[-chn:])
                tmp_ret = [*tmp_ret[:-chn], cret]
        return tmp_ret[0]

    def parse(self, words: str):
        """Compile and execute the words.

        Params:
        --------------------
        words: Input words.

        Return:
        --------------------
        Execution result.
        """
        nodes = self.compile(words)
        return self.exec(nodes)
