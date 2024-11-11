#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: test_geoenc.py
#   Author: xyy15926
#   Created: 2024-07-24 21:13:00
#   Updated: 2024-11-11 14:44:09
#   Description:
# ---------------------------------------------------------

# %%
import pytest
from pytest import mark

if __name__ == "__main__":
    from importlib import reload
    from modsbear.locale import geoenc
    reload(geoenc)

from modsbear.locale.geoenc import CHNGovEncoder


# %%
def test_get_gregion_toker():
    sent = "中国北京市深圳市开平区熊风扬"
    toker = CHNGovEncoder.get_gregion_toker()

    # Precise mode.
    tok_gen = toker.cut(sent)
    # Return-all mode, where the tokens could overlap with others.
    tok_gen_all = toker.cut(sent, cut_all=True)
    toks = toker.lcut(sent)
    toks_all = toker.lcut(sent, cut_all=True)

    assert list(tok_gen) == toks
    assert list(tok_gen_all) == toks_all
    assert len(toks) < len(toks_all)

    # For search engine as search-key, like return-all mode.
    tok_gen_se = toker.cut_for_search(sent)
    tok_se = toker.lcut_for_search(sent)

    assert list(tok_gen_se) == tok_se


# %%
def test_geo_encode():
    geoenc = CHNGovEncoder()
    # ptoker = geoenc.ptoker
    addr = "广东深圳福田莲花福中三路国银金融中心大厦2003号"
    with_all = geoenc.encode(addr)
    addr = "深圳福田莲花福中三路国银金融中心大厦2003号"
    miss_prov = geoenc.encode(addr)
    addr = "广东福田莲花福中三路国银金融中心大厦2003号"
    miss_city = geoenc.encode(addr)

    assert with_all == miss_prov
    assert with_all == miss_city
