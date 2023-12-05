#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: patterns.py
#   Author: xyy15926
#   Created: 2023-12-03 21:05:51
#   Updated: 2023-12-03 22:19:22
#   Description:
# ---------------------------------------------------------

# %%
ENDFLAG = "$END"
TOKEN_SPECS = {
    "ID"            : r"[A-Za-z_]+[A-Za-z_0-9]*",
    "FLOAT"         : (r"\d*\.(\d)+", float),
    "INTEGER"       : (r"\d+", int),
    "ADD"           : r"\+",
    "SUB"           : r"-",
    "MUL"           : r"\*",
    "DIV"           : r"/",
    "SEP"           : r",",
    "EQ"            : r"==",
    "NEQ"           : r"!=",
    "GE"            : r"\>=",
    "GT"            : r"\>",
    "LE"            : r"\<=",
    "LT"            : r"\<",
    "ASG"           : r"=",
    "AND"           : r"&&",
    "OR"            : r"\|\|",
    "LPAR"          : r"\(",
    "RPAR"          : r"\)",
    "LSPAR"         : r"\[",
    "RSPAR"         : r"\]",
    "LBPAR"         : r"\{",
    "RBPAR"         : r"\}",
    "LITERAL"       : r"\$\$",
    "COMMA"         : r",",
    "DOT"           : r"\.",
    "SKIP_NL"       : r"\n+",
    "SKIP_SEMI"     : r";",
    "SKIP_BLANK"    : r"[ \t]+",
}
RESERVEDS = {
    "if"            : "IF",
    "else"          : "ELSE",
    "or"            : "OR",
    "and"           : "AND",
    "endif"         : "ENDIF"
}
# TODO
ARITH_PRODS = [
    ("S -> expr"                    , lambda x: x[0]            ,(0, "R")),
    ("expr -> FLOAT"                , lambda x: x[0]            ,(0, "R")),
    ("expr -> SUB FLOAT"            , lambda x: -x[1]           ,(3, "R")),
    ("expr -> INTEGER"              , lambda x: x[0]            ,(0, "R")),
    ("expr -> SUB INTEGER"          , lambda x: -x[1]           ,(3, "R")),
    ("expr -> expr MUL expr"        , lambda x: x[0] * x[2]     ,(2, "L")),
    ("expr -> expr DIV expr"        , lambda x: x[0] / x[2]     ,(2, "L")),
    ("expr -> expr ADD expr"        , lambda x: x[0] + x[2]     ,(1, "L")),
    ("expr -> expr SUB expr"        , lambda x: x[0] - x[2]     ,(1, "L")),
    ("expr -> LPAR expr RPAR"       , lambda x: x[1]            ,(0, "R")),
]
