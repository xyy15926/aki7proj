#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: geoenc.py
#   Author: xyy15926
#   Created: 2024-07-25 14:01:53
#   Updated: 2025-01-20 16:31:20
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
import logging
# from IPython.core.debugger import set_trace

from collections import ChainMap
import pandas as pd
import jieba
from jieba import posseg

from pathlib import Path
from importlib_resources import files
from ubears.flagbear.slp.finer import get_data, get_tmp_path

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")

GOVERN_REGION_LV4 = get_data() / "govern_region/govern_region_level4.csv"


# %%
class CHNGovEncoder:
    """Encode address in ZH-CN into structured informations.

    Attrs:
    ------------------------
    ptoker: jieba.posseg.POSToknizer
      Toknizer to cut address string.
    reg_df: pd.DataFrame
      DataFrame with region-id as index for search the region infomation.
    reg_name_map: ChainMap[region-name, List[region-id]]
      Mapper to get the possible region-id of given region-name.
    """
    def __init__(self):
        self.ptoker = posseg.POSTokenizer(self.get_gregion_toker())
        self._init_region_mapper()

    @staticmethod
    def get_gregion_toker() -> jieba.Tokenizer:
        """Get Jieba Tokenizer with region names dict loaded.
        """
        greg_dict = get_tmp_path() / "govern_region_names.txt"
        if not greg_dict.is_file():
            reg_df = pd.read_csv(GOVERN_REGION_LV4)
            reg_df["pos"] = "ns"

            # Get region names and set word freq.
            # The region with higher level will be set with larger word freq.
            reg_names = (reg_df[["name", "deep", "pos"]]
                         .drop_duplicates("name")
                         .copy())
            reg_names["deep"] = 10000 // (reg_names["deep"] + 1) ** 2

            # Get region ext-names and set word freqs.
            reg_exts = (reg_df[["ext_name", "deep", "pos"]]
                        .drop_duplicates("ext_name")
                        .copy())
            reg_exts["deep"] = 150000000 // (reg_exts["deep"] + 1) ** 2
            reg_exts.set_axis(["name", "deep", "pos"], axis=1, inplace=True)

            # Write regions name to word dict file.
            reg_names = (pd.concat([reg_names, reg_exts])
                         .to_csv(greg_dict,
                                 sep=" ",
                                 index=None,
                                 header=None))

        toker = jieba.Tokenizer(greg_dict)
        # Load user dict.
        toker.add_word("熊风扬", tag="nr")

        return toker

    def _init_region_mapper(self):
        """Init `reg_df` and `reg_name_map` with the config of regions.

        The config file should contain region-id, region-name, region-extname.
        """
        reg_df = pd.read_csv(GOVERN_REGION_LV4)
        reg_df["id"] = reg_df["id"].astype(str)
        name_map = reg_df.groupby("name")["id"].agg(lambda x: x.to_list())
        ext_map = reg_df.groupby("ext_name")["id"].agg(lambda x: x.to_list())

        self.reg_df = reg_df.set_index("id")
        self.reg_name_map = ChainMap(ext_map, name_map)

    def encode(self, addr: str):
        """Encode given address into structured infomation.

        Params:
        ----------------------
        addr: Address to be structured.

        Return:
        ----------------------
        {
            "province": str,
            "city": str,
            "county": str,
            "district": str,
            "province_id": str,
            "city_id": str,
            "county_id": str,
            "district_id": str,
            "detail": str,
            "detail_core": str,
        }
        """
        ptoker = self.ptoker
        reg_name_map = self.reg_name_map
        reg_df = self.reg_df

        # Cut address into tokens of address and others.
        addr_lv_ids = {}        # {id-len: [], }
        addr_toks = []          # [(idx, field, flag), ]
        tokens = ptoker.lcut(addr)
        for idx, (field, flag) in enumerate(tokens):
            # Atmost 4 address parts are allowed.
            if flag == "ns" and len(addr_toks) < 4:
                # Get address ids from the mapper.
                ids = reg_name_map.get(field)
                if ids is not None:
                    for ele in ids:
                        addr_lv_ids.setdefault(len(ele), []).append(ele)
                    addr_toks.append((idx, field, flag))

        addr_stop_mapper = {
            2: "province",
            4: "city",
            6: "county",
            9: "district",
            # 12: "final",
        }
        addr_ids = [None] * len(addr_stop_mapper)
        check_prefix = {}
        addr_stops = list(addr_stop_mapper.keys())
        # Cut the ids to different levels seperately.
        for stop in addr_stops:
            aids = addr_lv_ids.get(stop)
            # Cut all ids into different levels so to get the intersection as
            # the id for the region of corresponsible level.
            # So that some missed level could be infered by its' child and
            # some noises of the filling-ups could be canceled.
            if aids is not None:
                for cut_stop in addr_stops:
                    if cut_stop > stop:
                        break
                    cut_stop_prefix = check_prefix.setdefault(cut_stop, [])
                    cut_stop_prefix.append([ele[:cut_stop] for ele in aids])

        # Intersect to get the ids.
        last_id = None
        for idx, stop in enumerate(addr_stops):
            prefix = check_prefix.get(stop)
            if prefix is None:
                continue
            intersec = set(prefix[0]).intersection(*prefix[1:])

            # Filter the child ids with pid.
            if last_id is not None:
                intersec = set([ele for ele in intersec if ele.startswith(last_id)])

            # Pop the intersection as the id.
            if len(intersec) == 0:
                continue
            elif len(intersec) > 1:
                logger.warning("Multiple possible address.")
            last_id = intersec.pop()

            addr_ids[idx] = last_id

        # Recover the upper level region-id with lower level region-id.
        for idx in range(len(addr_ids) - 2, -1, -1):
            upper_rid = addr_ids[idx]
            lower_rid = addr_ids[idx - 1]
            if upper_rid is None and lower_rid is not None:
                addr_ids[idx] = lower_rid[:addr_stops[idx]]

        # Map to get the precise region-names.
        addr_exts = [reg_df.loc[rid, "ext_name"]
                     if rid is not None else None
                     for rid in addr_ids]

        # Drop tokens before the tokens for structure infomation.
        end_addr_ext = None
        for addr_ext in reversed(addr_exts):
            if addr_ext is not None:
                end_addr_ext = addr_ext
                break
        if end_addr_ext is not None:
            # Traverse to get the pos of the last region in address tokens.
            for idx, field, flag in addr_toks:
                if end_addr_ext.startswith(field):
                    break
        else:
            idx = 0
            logger.warning("No structured region extracted.")

        # Construct return dict.
        rets = {}
        for lv_name, rid, rname in zip(addr_stop_mapper.values(),
                                       addr_ids, addr_exts):
            rets[f"{lv_name}_id"] = rid
            rets[f"{lv_name}"] = rname

        rets["detail"] = "".join([field for field, flag in tokens[idx + 1:]])
        # Filter the not-noun tokens which is considered to be less necessary
        # so to get the core infomations of the detail.
        rets["detail_core"] = "".join([field for field, flag
                                       in tokens[idx + 1:]
                                       if flag.startswith("n")])

        return rets
