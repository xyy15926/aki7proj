#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: display.py
#   Author: xyy15926
#   Created: 2024-11-29 12:13:36
#   Updated: 2025-01-21 12:04:58
#   Description:
# ---------------------------------------------------------

# %%
from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING

import logging
import os
import pandas as pd
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.charts import Kline, Line, Bar, Grid

from ubears.flagbear.slp.finer import tmp_file, get_assets_path

# %%
logging.basicConfig(
    format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"))
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
def draw_kline(
    data: pd.DataFrame,
    MAs: pd.DataFrame = None,
    PTNs: pd.DataFrame = None,
    fname: str = None,
) -> Grid:
    """Draw KLine with volume bars, trend lines.

    Params:
    ------------------------
    data: Price data with columns
      date: Time tag.
      open_: Open price.
      high: High price.
      low: Low price.
      close: Close price.
      volume: Volume.
    MAs: MA trend line.
      Each column of `MAs` represents one type of MAs line and will be drawn
      seperately with Line.
    PTNs: Candlestick pattern.
      Each column of `PTNs` represents one type of pattern recognition with
      value of 100, 0, -100 and will be drawn seperately with Line.
    fname: Filename for saving html render

    Return:
    ------------------------
    Grid Chart
    """
    xticks = data["date"].tolist()
    bars = data[["open_", "close", "high", "low"]].values.tolist()
    updn = (data["close"] > data["open_"]).astype(int) * 2 - 1
    # `Series.values.tolist()` is necessary or unsupported `np.int64` will
    # be passed.
    volumes = list(zip(range(len(data)),
                       data["volume"].values.tolist(),
                       updn))

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
                is_show=True,
                pos_top=10,
                pos_left="center"
            ),
            # Set both two DataZoom wigets for xaxis:
            # 1. inside: scoll to zoom in or out
            # 2. slider: slider bar
            # But bugs: two pair of datazoom wigets will be rendered if
            #   a list is passed. So two datazooms are set for different
            #   chart.
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=False,
                    type_="inside",
                    # Control index 0 and 1.
                    xaxis_index=[0, 1],
                    range_start=98,
                    range_end=100,
                    min_value_span=30,
                ),
                opts.DataZoomOpts(
                    is_show=True,
                    xaxis_index=[0, 1],
                    type_="slider",
                    pos_top="90%",
                    range_start=98,
                    range_end=100,
                    min_value_span=30,
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
    MAs = pd.DataFrame() if MAs is None else MAs
    line = Line() .add_xaxis(xaxis_data=xticks)
    for matype in MAs:
        line.add_yaxis(
            series_name=matype,
            y_axis=MAs[matype].tolist(),
            yaxis_index=0,
            is_smooth=True,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
        )
    # Add candlstick patterns.
    PTNs = pd.DataFrame() if PTNs is None else PTNs
    for ptntype in PTNs:
        line.add_yaxis(
            series_name=ptntype,
            y_axis=PTNs[ptntype].tolist(),
            yaxis_index=1,
            is_smooth=True,
            is_hover_animation=False,
            linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
        )
    overlaped = kline.overlap(line)
    # The bottom axes will be overrided, so the `extend_axis` on the overlapped.
    overlaped.extend_axis(
        yaxis=opts.AxisOpts(type_="value", position="right")
    )

    # Volume Bar.
    bar = (
        Bar()
        .add_xaxis(xaxis_data=xticks)
        .add_yaxis(
            series_name="Volume",
            y_axis=volumes,
            xaxis_index=1,
            yaxis_index=2,
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
                series_index=MAs.shape[1] + PTNs.shape[1] + 1,
                is_piecewise=True,
                pieces=[
                    {"value": 1, "color": "#ec0000"},
                    {"value": -1, "color": "#00da3c"},
                ],
            ),
        )
    )

    grid_chart = Grid(
        init_opts=opts.InitOpts(
            width="1000px",
            height="800px",
            animation_opts=opts.AnimationOpts(animation=False),
            theme=ThemeType.INFOGRAPHIC,
        )
    )
    grid_chart.add(
        overlaped,
        grid_opts=opts.GridOpts(
            pos_left="5%",
            pos_right="5%",
            height="60%",
        ),
        grid_index=0,
        # Control axis by self, so the axis index won't be overrided by the
        # Grid with `grid_index`.
        is_control_axis_index=True,
    )
    grid_chart.add(
        bar,
        grid_opts=opts.GridOpts(
            pos_left="5%",
            pos_right="5%",
            pos_top="75%",
            height="15%",
        ),
        grid_index=1,
        is_control_axis_index=True,
    )
    grid_chart.options["dataZoom"] = grid_chart.options["dataZoom"][:-1]

    if fname is not None:
        ghtml = grid_chart.render(fname)
        logger.info(f"Graph saved at {ghtml}.")

    return grid_chart
