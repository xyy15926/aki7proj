#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: parser.py
#   Author: xyy15926
#   Created: 2023-12-02 21:04:21
#   Updated: 2024-01-24 10:00:42
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
from collections.abc import Callable
from IPython.core.debugger import set_trace

import logging
from collections import namedtuple

from flagbear.lex import Token, Lexer
from flagbear.syntax import Production, LRItem, LRState, Syntaxer
from flagbear.tree import GeTNode
from flagbear.patterns import (LEX_TOKEN_SPECS, LEX_SKIPS,
                               LEX_RESERVEDS, LEX_ENDFLAG,
                               REGEX_TOKEN_SPECS,
                               SYN_STARTSYM, SYN_ARITH_PRODS,
                               SYN_EXPR_PRODS,
                               CALLABLE_ENV)

Reduction = TypeVar("Reduction")

logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
class LRParser:
    """Parse and execute words.

    Attrs:
    --------------------------
    lexer: Lexer to split words into tokens.
    syntaxer: Syntaxer to compile tokens into ASTree.
    history: Dict to record the history to accelerate.
    prod_reduces: Dict of Production and Callable to provide extra reduction for
      Production.
    """
    def __init__(
        self, token_specs: dict = LEX_TOKEN_SPECS,
        reserveds: dict = LEX_RESERVEDS,
        skips: set = LEX_SKIPS,
        productions: list[Production | tuple] = SYN_ARITH_PRODS,
        start_sym: str = SYN_STARTSYM,
        end_flag: str = LEX_ENDFLAG,
        prod_reduces: dict[Production, Callable[[list | tuple], Any]] | None = None,
    ):
        """Init LRParser.

        Params:
        --------------------------
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
          List of productions.
        start_sym: str
          Start symbol.
        end_flag: str
          End flag marking the end of input terminal stream.
          Namely the speical token type name to mark the end of a token stream.
        prod_reduces: dict[Production, Callable[[list | tuple], Any]] | None
          Dict of Production and Callable to provide extra reduction for
          Production.
        """
        self.lexer = Lexer(token_specs, reserveds, skips, end_flag)
        self.syntaxer = Syntaxer(productions, start_sym, end_flag)
        self.syntaxer.init_gotos()
        self.history = {}
        self.prod_reduces = {} if prod_reduces is None else prod_reduces

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
        prod_reduces = self.prod_reduces
        ret_ST = []
        for node in nodes:
            chn = node.chn
            val = node.val
            # Token node -> append directly.
            if isinstance(val, Token):
                ret_ST.append(val.val)
            # Production node -> reduce then append.
            elif chn == 0:
                reduce = prod_reduces.get(val, val.reduce)
                ret_ST.append(reduce(None))
            else:
                reduce = prod_reduces.get(val, val.reduce)
                ret = reduce(ret_ST[-chn:])
                ret_ST = [*ret_ST[:-chn], ret]
        return ret_ST[0]

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


# %%
class EnvParser(LRParser):
    """Parser with environment to get value of ID.

    Attrs:
    ----------------------
    env: dict[str of variable name, Any]
      The dict to get the value of ID from.
    id_prod: Production
      Hook for changing the behavior of how to get the value of token `ID` in
      order to bind `env` to `_`.
    """
    def __init__(self, default_env: dict[str, Callable] = CALLABLE_ENV):
        """Init parser with environment.

        Replace the token ID's specification with a list as a hook, so to
        change the value of token ID easily.
        """
        super().__init__(LEX_TOKEN_SPECS, LEX_RESERVEDS, LEX_SKIPS,
                         SYN_EXPR_PRODS, SYN_STARTSYM, LEX_ENDFLAG)
        self.default_env = default_env
        self.env = None
        for prod in self.syntaxer.productions:
            if prod[1] == ("ID",):
                break
        self.id_prod = prod
        self.bind_env(None)

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
        if env is not self.env:
            self.env = env
            if isinstance(env, dict):
                envs = ChainMap(env, self.default_env)
            else:
                envs = self.default_env
            self.prod_reduces[self.id_prod] = (lambda x: env if x[0] == "_"
                                               else envs.get(x[0]))
        return self
