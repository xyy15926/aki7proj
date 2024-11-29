#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: display.py
#   Author: xyy15926
#   Created: 2024-11-29 12:13:36
#   Updated: 2024-11-29 17:22:42
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING

import logging
import os
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Kline, Line, Bar, Grid

from flagbear.slp.finer import tmp_file, get_assets_path

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def kline(
    oris: pd.DataFrame,
    mas: pd.DataFrame = None,
    fname: str = "stock/kline.html",
) -> Kline:
    """
    """
    xticks = oris["date"].tolist()
    bars = oris[["open_", "close", "high", "low"]].values.tolist()
    updn = (oris["close"] > oris["open_"]).astype(int) * 2 - 1
    volumes = list(zip(range(len(oris)), oris["volume"].values.tolist(), updn))

    # Basic Kline.
    kline = (
        Kline()
        .add_xaxis(xticks)
        .add_yaxis(
            series_name="kline",
            y_axis=bars,
        ).set_global_opts(
            title_opts=opts.TitleOpts(title="KLine"),
            legend_opts=opts.LegendOpts(
                is_show=False,
                pos_bottom=10,
                pos_left="center"
            ),
            datazoom_opts=[
                # Set both two DataZoom wigets for xaxis:
                # 1. inside: scoll to zoom in or out
                # 2. slider: slider bar
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",
                    # Control index 0 and 1.
                    xaxis_index=[0, 1],
                    range_start=98,
                    range_end=100,
                ),
                opts.DataZoomOpts(
                    # Bug: Two slide bars show.
                    is_show=True,
                    xaxis_index=[0, 1],
                    type_="slider",
                    pos_top="85%",
                    range_start=98,
                    range_end=100,
                ),
            ],
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True,
                    areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(245, 245, 245, 0.8)",
                border_width=1,
                border_color="#ccc",
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            ),
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True,
                link=[{"xAxisIndex": "all"}],
                label=opts.LabelOpts(background_color="#777"),
            ),
            brush_opts=opts.BrushOpts(
                x_axis_index="all",
                brush_link="all",
                out_of_brush={"colorAlpha": 0.1},
                brush_type="lineX",
            ),
        )
    )

    # Add MA Lines.
    line = Line().add_xaxis(xaxis_data=xticks)
    mas = pd.DataFrame() if mas is None else mas
    for ma in mas:
        line.add_yaxis(
            series_name=ma,
            y_axis=mas[ma].tolist(),
            is_smooth=True,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
        )
    overlaped = kline.overlap(line)

    # Volume Bar.
    bar = (
        Bar()
        .add_xaxis(xaxis_data=xticks)
        .add_yaxis(
            series_name="Volume",
            y_axis=volumes,
            xaxis_index=1,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
        ).set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                grid_index=1,
                boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=False),
                split_number=20,
                min_="dataMin",
                max_="dataMax",
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                is_scale=True,
                split_number=2,
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(
                is_show=False,
                dimension=2,
                # Index among the whole data series.
                series_index=mas.shape[1] + 1,
                is_piecewise=True,
                pieces=[
                    {"value": 1, "color": "#00da3c"},
                    {"value": -1, "color": "#ec0000"},
                ],
            ),
        )
    )

    grid_chart = Grid(
        init_opts=opts.InitOpts(
            width="1000px",
            height="800px",
            animation_opts=opts.AnimationOpts(animation=False)
        )
    )
    grid_chart.add(
        overlaped,
        grid_opts=opts.GridOpts(
            pos_left="10%",
            pos_right="8%",
            height="50%",
        )
    )
    grid_chart.add(
        bar,
        grid_opts=opts.GridOpts(
            pos_left="10%",
            pos_right="8%",
            pos_top="63%",
            height="16%",
        )
    )

    if fname is not None:
        ghtml = grid_chart.render(tmp_file(fname, incr=0).with_suffix(".html"))
        logger.info(f"Graph saved at {ghtml}.")

    return kline


# %%
if __name__ == "__main__":
    import json

    dfile = get_assets_path() / "stock/stock_jdi.json"
    data = json.load(open(dfile))
    oris = pd.DataFrame.from_records(
        data,
        columns=["date", "open_", "close", "low", "high", "volume"])
    kline(oris)
