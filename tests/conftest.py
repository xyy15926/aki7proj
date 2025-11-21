#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: conftest.py
#   Author: xyy15926
#   Created: 2024-09-20 17:42:52
#   Updated: 2024-09-20 21:15:12
#   Description:
# ---------------------------------------------------------

# %%
import pytest
import time

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# %%
# session, module, class, function(default)
@pytest.fixture(scope="session", autouse=False)
def time_session_scope():
    start = time.time()
    print("\nStart: {}".format(time.strftime(DATE_FORMAT, time.localtime(start))))
    yield
    end = time.time()
    print("\nEnd: {}".format(time.strftime(DATE_FORMAT, time.localtime(end))))
    print("Total time cost: {:0.3f}s".format(end - start))


@pytest.fixture(scope="function", autouse=False)
def time_function_scope():
    start = time.time()
    yield
    end = time.time()
    print("\nTime cost: {:0.3f}s".format(end - start))
