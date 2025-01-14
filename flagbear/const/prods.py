#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: prods.py
#   Author: xyy15926
#   Created: 2025-01-14 20:08:15
#   Updated: 2025-01-14 20:08:15
#   Description:
# ---------------------------------------------------------

# %%
SYN_STARTSYM = "S"

SYN_ARITH_PRODS = [
    ("S"        , ("expr", )                    , lambda x: x[0]            , 0     , "R"),
    ("expr"     , ("FLOAT", )                   , lambda x: x[0]            , 0     , "R"),
    ("expr"     , ("SUB", "FLOAT")              , lambda x: -x[1]           , 3     , "R"),
    ("expr"     , ("INT", )                     , lambda x: x[0]            , 0     , "R"),
    ("expr"     , ("SUB", "INT")                , lambda x: -x[1]           , 3     , "R"),
    ("expr"     , ("expr", "ADD", "expr")       , lambda x: x[0] + x[2]     , 1     , "L"),
    ("expr"     , ("expr", "SUB", "expr")       , lambda x: x[0] - x[2]     , 1     , "L"),
    ("expr"     , ("expr", "MUL", "expr")       , lambda x: x[0] * x[2]     , 2     , "L"),
    ("expr"     , ("expr", "DIV", "expr")       , lambda x: x[0] / x[2]     , 2     , "L"),
    ("expr"     , ("LPAR", "expr", "RPAR")      , lambda x: x[1]            , 0     , "R"),
]

# %%
# TODO
SYN_EXPR_PRODS = [
    ("S"        , ("expr", )                    , lambda x: x[0]                    , 0     , "R"),
    ("expr"     , ("ID", )                      , None                              , 0     , "R"),
    ("expr"     , ("FLOAT", )                   , lambda x: x[0]                    , 0     , "R"),
    ("expr"     , ("INT", )                     , lambda x: x[0]                    , 0     , "R"),
    ("expr"     , ("STRING", )                  , lambda x: x[0]                    , 0     , "R"),
    ("expr"     , ("SUB", "expr")               , lambda x: -x[1]                   , 8     , "R"),
    ("expr"     , ("expr", "ADD", "expr")       , lambda x: x[0] + x[2]             , 3     , "L"),
    ("expr"     , ("expr", "SUB", "expr")       , lambda x: x[0] - x[2]             , 3     , "L"),
    ("expr"     , ("expr", "MUL", "expr")       , lambda x: x[0] * x[2]             , 4     , "L"),
    ("expr"     , ("expr", "DIV", "expr")       , lambda x: x[0] / x[2]             , 4     , "L"),
    ("expr"     , ("LPAR", "expr", "RPAR")      , lambda x: x[1]                    , 0     , "L"),
    ("eles"     , ()                            , lambda x: []                      , 0     , "L"),
    ("eles"     , ("expr", )                    , lambda x: [x[0]]                  , 0     , "L"),
    ("eles"     , ("expr", "COMMA", "expr")     , lambda x: [x[0], x[2]]            , 0     , "L"),
    ("eles"     , ("expr", "COMMA", "eles")     , lambda x: [x[0], *x[2]]           , 0     , "L"),
    ("expr"     , ("LBPAR", "eles", "RBPAR")    , lambda x: frozenset(x[1])         , 0     , "L"),
    ("expr"     , ("LSPAR", "eles", "RSPAR")    , lambda x: x[1]                    , 0     , "L"),
    ("expr"     , ("expr", "EQ", "expr")        , lambda x: x[0] == x[2]            , 2     , "L"),
    ("expr"     , ("expr", "NEQ", "expr")       , lambda x: x[0] != x[2]            , 2     , "L"),
    ("expr"     , ("expr", "LT", "expr")        , lambda x: x[0] < x[2]             , 2     , "L"),
    ("expr"     , ("expr", "GT", "expr")        , lambda x: x[0] > x[2]             , 2     , "L"),
    ("expr"     , ("expr", "LE", "expr")        , lambda x: x[0] <= x[2]            , 2     , "L"),
    ("expr"     , ("expr", "GE", "expr")        , lambda x: x[0] >= x[2]            , 2     , "L"),
    ("expr"     , ("expr", "OR", "expr")        , lambda x: x[0] or x[2]            , 1     , "L"),
    ("expr"     , ("expr", "BOR", "expr")       , lambda x: x[0].__or__(x[2])       , 1     , "L"),
    ("expr"     , ("expr", "AND", "expr")       , lambda x: x[0] and x[2]           , 1     , "L"),
    ("expr"     , ("expr", "BAND", "expr")      , lambda x: x[0].__and__(x[2])      , 1     , "L"),
    ("expr"     , ("NOT", "expr")               , lambda x: not x[1]                , 1     , "R"),
    ("expr"     , ("expr", "IN", "expr")        , lambda x: x[0] in x[2]            , 1     , "L"),
    ("expr"     , ("expr", "LSPAR", "expr", "RSPAR")    , lambda x: x[0][x[2]]      , 9     , "L"),
    ("expr"     , ("expr", "LPAR", "expr", "RPAR")      , lambda x: x[0](x[2])      , 9     , "L"),
    ("expr"     , ("expr", "LPAR", "eles", "RPAR")      , lambda x: x[0](*x[2])     , 9     , "L"),
]

CALLABLE_ENV = {
    "count"             : len,
    "sum"               : sum,
    "max"               : max,
    "min"               : min,
    "nnfilter"          : lambda x: [i for i in x if i is not None],
    "nncount"           : lambda x: len([i for i in x if i is not None]),
}
