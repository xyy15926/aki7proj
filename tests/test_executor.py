#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_executor.py
#   Author: xyy15926
#   Created: 2023-07-24 15:52:26
#   Updated: 2023-08-16 15:21:15
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import string
import pandas as pd
from ringbear.executor import (
    exec_expstr,
    aggregate_with_key,
)
from ringbear.unifier import (
    extract_field,
)


# %%
def test_str_executor() -> None:
    assert(exec_expstr("1$ + 2$ * 3$ * 4$ + (5$ + 6$) * 7$") == 102)
    assert(exec_expstr("1$ + 2$ * 3$ *4$ +(5$ + 6$)*7$") == 102)
    assert(exec_expstr("1$+2$*3$*4$+(5$+6$)*7$") == 102)
    assert(exec_expstr("1$ +2$ *3$ *4$ < (5$ +6$)*7$"))
    assert(exec_expstr("1$ +2$ *3$ *4$ <= (5$ +6$)*7$"))
    nest = {i: int(i) for i in string.digits}
    assert(exec_expstr("1+2*3*4 <= (5+6)*7", nest))
    nest = {"a": 1}
    assert(exec_expstr("a+2$ *3$ *4$ +(5$ +6$)*7$", nest) == 102)
    a = [(1,2,2), (1,2,4), (2,3,4)]
    assert aggregate_with_key(a, key=0, agg_how=["first", sum, "list"])[0] == [1, 4, [2, 4]]
    assert aggregate_with_key(a, key=0, agg_how=["first", "list", "last"])[0] == [1, [2, 2], 4]

def test_str_executor_dataframe() -> None:
    df = pd.DataFrame(
        [[1, 2, 3, 4],
         [2, 3, 4, 5],
         [1, 3, 4, 5],
         [2, 2, 3, 4]],
        columns=list("abcd"))
    rules = pd.DataFrame(
        [["a>1$", "R0001", "R0001"],
         ["b<3$", "R0002", "R0002"],
         ["b<3$&c>2$", "R0003", "R0003"]],
        columns=["rule", "code", "desc"])
    assert exec_expstr(rules.iloc[0, 0], df).sum() == 2
    assert exec_expstr(rules.iloc[2, 0], df).sum() == 2


# %%
def test_dict_extractor() -> None:
    test_cases = {
        "AntiFraud": [
            {"riskInfo": [{"riskCode": 1101}, {"riskCode": 1105}], "riskScore": 62},
        ],
    }
    assert(extract_field(test_cases["AntiFraud"][0], "riskInfo:count") == 2)
    assert(extract_field(test_cases["AntiFraud"][0], "riskInfo:[0]:riskCode&&riskCode==1101$") == 1101)
    assert(extract_field(test_cases["AntiFraud"][0], "riskInfo:[0]:riskCode&&riskCode==1107$") is None)
    assert(extract_field(test_cases["AntiFraud"][0], "riskInfo:max:riskCode&&riskCode==1107$") is None)
    assert(extract_field(test_cases["AntiFraud"][0], "riskInfo:max:riskCode&&riskCode==1105$") == 1105)
    assert(extract_field(test_cases["AntiFraud"][0], "riskInfo:max:riskCode") == 1105)
    assert(1105 in extract_field(test_cases["AntiFraud"][0], "riskInfo:[]:riskCode&&riskCode@{1105,1107}$"))


