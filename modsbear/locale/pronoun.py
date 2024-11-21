#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: pronoun.py
#   Author: xyy15926
#   Created: 2024-11-21 10:06:38
#   Updated: 2024-11-21 20:22:23
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, Any

import logging
import numpy as np
import string

from modsbear.locale.govreg import get_chn_govrs
from flagbear.slp.finer import get_assets_path

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def init_short_alts():
    fname = get_assets_path() / "userdict/jieba_dict_small.txt"
    name_alts = []
    for line in open(fname, "r"):
        zh, w, zht = line.strip().split(" ")
        if zht in ["j"] and len(zh) <= 2:
            name_alts.append(zh)
    return name_alts


def init_firstname_alts():
    fname = get_assets_path() / "userdict/jieba_dict_small.txt"
    name_alts = []
    for line in open(fname, "r"):
        zh, w, zht = line.strip().split(" ")
        if zht in ["nr"] and len(zh) <= 2:
            name_alts.append(zh)
    return name_alts


def init_orgno_ws():
    ws = [0] * 17
    for i in range(1, 18):
        ws[i - 1] = 3 ** (i - 1) % 31
    return ws


FIRSTNAME_ALTS = init_firstname_alts()
LASTNAME_ALTS = [
    "李", "王", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴",
    "徐", "孙", "朱", "马", "胡", "郭", "林", "何", "高", "梁",
    "郑", "罗", "宋", "谢", "唐", "韩", "曹", "许", "邓", "萧",
    "冯", "曾", "程", "蔡", "彭", "潘", "袁", "於", "董", "余",
    "苏", "叶", "吕", "魏", "蒋", "田", "杜", "丁", "沈", "姜",
    "范", "江", "傅", "钟", "卢", "汪", "戴", "崔", "任", "陆",
    "廖", "姚", "方", "金", "邱", "夏", "谭", "韦", "贾", "邹",
    "石", "熊", "孟", "秦", "阎", "薛", "侯", "雷", "白", "龙",
    "段", "郝", "孔", "邵", "史", "毛", "常", "万", "顾", "赖",
    "武", "康", "贺", "严", "尹", "钱", "施", "牛", "洪", "龚",
]
SHORT_ALTS = init_short_alts()
ORGNO_WS = init_orgno_ws()


# %%
def certno_parity(certno_p: str) -> str:
    """Generate the parity bit of the certno.

    Params:
    -----------------------
    certno_p: Certno without last parity bit.

    Return:
    -----------------------
    Valid certno parity bit char.
    """
    ws = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    wss = 0
    for w, b in zip(ws, certno_p):
        wss += w * int(b)
    rem = (11 - ((wss - 1) % 11)) % 11
    return "X" if rem == 10 else str(rem)


# %%
def certno_check(certno: str):
    """Check if certno is valid.

    Params:
    -----------------------
    certno_p: Certno without last parity bit.

    Return:
    -----------------------
    If the certno is valid.
    """
    ws = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2, 1]
    wss = 0
    for w, b in zip(ws, certno):
        b = 10 if b.upper() == "X" else int(b)
        wss += w * b
    rem = wss % 11
    return rem == 1


# %%
def rand_certno(
    govrid: str = None,
    age: int | tuple = None,
    gender: int = None,
) -> str:
    """Generate random certno.

    Params:
    -----------------------
    govrid: Predefed government region.
    age: Predefed age.
    gender: Predefed gender, 1 for male and 0 for female.

    Return:
    -----------------------
    Valid certno.
    """
    # Random government region id.
    if govrid is None:
        govrid = str(np.random.choice(get_chn_govrs(3)["id"].values))

    # Random birthday.
    if age is None:
        age = (20, 50)
    elif isinstance(age, int):
        age = (age, age)
    ty = np.datetime64("today", "Y")
    ed = ((ty - age[0] + 1).astype("M8[D]") - 1).astype(int)
    sd = (ty - age[1]).astype("M8[D]").astype(int)
    rd = (str(np.array(np.random.randint(sd, ed)).astype("M8[D]"))
          .replace("-", ""))

    # Random regsiter id.
    regnum = np.random.randint(1, 327)
    if gender is not None and gender != regnum % 2:
        regnum += 1

    certno_p = f"{govrid}{rd}{regnum:03}"
    parity = certno_parity(certno_p)
    certno = f"{certno_p}{parity}"

    return certno


# %%
# TODO: Enable SIP selection.
def rand_mobile() -> str:
    """Generate random cell phone number.
    """
    TELE = [133, 149, 153, 173, 177, 180, 181,
            189, 190, 191, 193, 199]
    UNICOM = [130, 131, 132, 145, 155, 156, 166, 167, 171,
              175, 176, 185, 186, 196]
    MOBILE = [134,
              135, 136, 137, 138, 139, 1440, 147, 148, 150,
              151, 152, 157, 158, 159, 172, 178, 182, 183,
              184, 187, 188, 195, 197, 198]
    VTELE = [1700, 1701, 1702, 162]
    VUNICOM = [1704, 1707, 1708, 1709, 171, 167]
    VMOBILE = [1703, 1705, 1706, 165]

    f3s = TELE + UNICOM + MOBILE
    f3sv = VTELE + VUNICOM + VMOBILE
    f3 = np.random.choice(f3s)
    tail = np.random.randint(0, 9, 8)
    tail = "".join([str(ele) for ele in tail])

    return f"{f3}{tail}"[:11]


# %%
def rand_nname() -> str:
    """Generate random name.
    """
    # Init global firstname alternatives if necessary.
    global FIRSTNAME_ALTS
    if FIRSTNAME_ALTS is None:
        FIRSTNAME_ALTS = init_firstname_alts()

    name = np.random.choice(LASTNAME_ALTS) + np.random.choice(FIRSTNAME_ALTS)

    return name


# %%
def rand_email(dom: str = "163.com") -> str:
    """Generate random email address.
    """
    alts = list(string.digits + string.ascii_lowercase + "_")
    accl = np.random.randint(6, 20)
    return "".join(np.random.choice(alts, accl)) + "@" + dom


# %%
def rand_orgname() -> str:
    """Generate random orgnization name.
    """
    # Init global shortname alternatives if necessary.
    global SHORT_ALTS
    if SHORT_ALTS is None:
        SHORT_ALTS = init_short_alts()

    name = ""
    while len(name) < 5:
        name += np.random.choice(SHORT_ALTS)
    name += "有限公司"

    return name


# %%
def orgno_parity(orgno_p: str) -> str:
    """Generate the parity bit of the orgno.

    Params:
    -----------------------
    orgno_p: Orgno without last parity bit.

    Return:
    -----------------------
    Valid orgno parity bit char.
    """
    # Init value map.
    alts = list(string.digits + "ABCDEFGHJKLMNPQRTUWXY")
    alts_M = {c: i for i, c in enumerate(alts)}

    wss = 0
    for w, b in zip(ORGNO_WS, orgno_p):
        wss += w * alts_M[b]
    rem = (31 - (wss % 31)) % 31

    return alts[rem]


# %%
def check_orgno(orgno: str) -> bool:
    """Check if orgno is valid.

    Params:
    -----------------------
    certno_p: Orgno without last parity bit.

    Return:
    -----------------------
    If the orgno is valid.
    """
    orgno = orgno.replace(" ", "")
    return orgno_parity(orgno[:-1]) == orgno[-1]


# %%
def rand_orgno() -> str:
    """Generate random unified orgnization code.
    """
    f12 = np.random.choice(["91", "92", "93"])
    f38 = str(np.random.choice(get_chn_govrs(3)["id"].values))
    alts = list(string.digits + "ABCDEFGHJKLMNPQRTUWXY")
    f917 = "".join(np.random.choice(alts, 9))
    orgno_p = f"{f12}{f38}{f917}"
    orgno = orgno_p + orgno_parity(orgno_p)

    return orgno


# %%
def rand_addr(
    govrid: str = None
) -> str:
    """Generate random address.

    Params:
    -----------------------
    govrid: Predefed government region.

    Return:
    -----------------------
    Random address.
    """
    govrs = get_chn_govrs(None).set_index("id")

    if govrid is None:
        govrid = np.random.choice(govrs[govrs["deep"] == 2].index)
    else:
        govrid = int(govrid)

    # Determine the goverment region.
    govr = govrs.loc[govrid]
    cityr = govrs.loc[govr["pid"]]
    provr = govrs.loc[cityr["pid"]]

    name = ""
    while len(name) < 5:
        name += np.random.choice(SHORT_ALTS)
    area = np.random.choice(["小区", "村", "农场", "大队"])

    addr = (f'{provr["ext_name"]}{cityr["ext_name"]}{govr["ext_name"]}'
            f'{name}{area}')

    return addr
