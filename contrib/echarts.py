#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: echarts.py
#   Author: xyy15926
#   Created: 2024-09-21 16:48:48
#   Updated: 2024-09-22 21:38:20
#   Description:
#   Ref: <https://gallery.pyecharts.org/>
# ---------------------------------------------------------

# %%
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.globals import ThemeType, SymbolType
from pyecharts.faker import Faker
from pyecharts.commons.utils import JsCode
# from pyecharts.render import make_snapshot

from suitbear.finer import get_assets_path, get_tmp_path

fname = get_tmp_path() / "pyecharts.html"

logging.basicConfig(
    # format="%(module)s: %(asctime)s: %(levelname)s: %(message)s",
    format="%(message)s",
    level=logging.INFO,
    force=(__name__ == "__main__"),
)
logger = logging.getLogger()
logger.info("Logging Start.")


# %%
# Ref: <https://gallery.pyecharts.org/#/Calendar/README>
def charts_calendar_heatmap():
    from pyecharts.charts import Calendar

    begin = np.datetime64("2023-09-21")
    end = np.datetime64("2024-09-21")
    dates = np.arange(begin, end, 1).astype(str)
    vals = np.random.randint(100, 500, dates.shape).tolist()
    data = list(zip(dates, vals))

    (
        Calendar()
        .add(
            series_name="",
            yaxis_data=data,
            calendar_opts=opts.CalendarOpts(
                pos_top="120",
                pos_left="30",
                pos_right="30",
                range_=["2023-09", "2024-09"],
                yearlabel_opts=opts.CalendarYearLabelOpts(is_show=True),
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(pos_top="30",
                                      pos_left="center",
                                      title="2023年步数情况"),
            visualmap_opts=opts.VisualMapOpts(
                max_=500, min_=100,
                orient="horizontal", is_piecewise=False
            ),
        )
        .render(fname)
    )


# %%
def chart_funnel():
    from pyecharts.charts import Funnel

    (
        Funnel()
        .add(
            "商品",
            [list(z) for z in zip(Faker.choose(), Faker.values())],
            sort_="ascending",
            label_opts=opts.LabelOpts(position="inside"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Funnel-Sort（ascending）")
        )
        .render(fname)
    )


# %%
def chart_gauge():
    from pyecharts.charts import Gauge

    (
        Gauge()
        .add(
            "业务指标",
            [("完成率", 55.5)],
            radius="100%",
            split_number=5,
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(
                    color=[(0.3, "#67e0e3"), (0.7, "#37a2da"), (1, "#fd666d")],
                    width=30
                )
            ),
            detail_label_opts=opts.LabelOpts(formatter="{value}"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Gauge-分割段数-Label"),
            legend_opts=opts.LegendOpts(is_show=False),
        )
        .render(fname)
    )


# %%
# 1. `category` 被设置时，未被 `add` 中 `categories` 参数显式指明类别不展示
def chart_graph():
    from pyecharts.charts import Graph

    nodes_data = [
        opts.GraphNode(name="Node1", symbol_size=10, category="Cat1"),
        opts.GraphNode(name="Node2", symbol_size=20, category="Cat1"),
        opts.GraphNode(name="Node3", symbol_size=30, category="Cat2"),
        opts.GraphNode(name="Node4", symbol_size=40, category="Cat1"),
        opts.GraphNode(name="Node5", symbol_size=50, category="Cat2"),
        opts.GraphNode(name="Node6", symbol_size=60, category="Cat2"),
    ]
    links_data = [
        opts.GraphLink(source="Node1", target="Node2", value=2),
        opts.GraphLink(source="Node2", target="Node3", value=3),
        opts.GraphLink(source="Node3", target="Node4", value=4),
        opts.GraphLink(source="Node4", target="Node5", value=5),
        opts.GraphLink(source="Node5", target="Node6", value=6),
        opts.GraphLink(source="Node6", target="Node1", value=7),
    ]
    categories = [
        opts.GraphCategory("Cat1"),
        opts.GraphCategory("Cat2"),
    ]
    (
        Graph()
        .add(
            "",
            nodes=nodes_data,
            links=links_data,
            categories=categories,
            repulsion=4000,
            edge_label=opts.LabelOpts(
                is_show=True, position="middle", formatter="{b} 的数据 {c}"
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Graph-GraphNode-GraphLink-WithEdgeLabel")
        )
        .render(fname)
    )


# %%
# 1. 多重波浪：`data` 设置为列表
def chart_liquid():
    from pyecharts.charts import Liquid

    (
        Liquid()
        .add(series_name="lq",
             data=[0.7, 0.5, 0.3],
             is_animation=True,
             is_outline_show=False,
             shape=SymbolType.DIAMOND)
        .set_global_opts(title_opts=opts.TitleOpts(title="Liquid-Shape-Diamond"))
        .render(fname)
    )


# %%
# 1. 多类别：`add` 多组数据
def chart_parallel():
    from pyecharts.charts import Parallel

    data = [
        [1, 91, 45, 125, 0.82, 34, 23, "良"],
        [2, 65, 27, 78, 0.86, 45, 29, "良"],
        [3, 83, 60, 84, 1.09, 73, 27, "良"],
        [4, 109, 81, 121, 1.28, 68, 51, "轻度污染"],
        [5, 106, 77, 114, 1.07, 55, 51, "轻度污染"],
        [6, 109, 81, 121, 1.28, 68, 51, "轻度污染"],
        [7, 106, 77, 114, 1.07, 55, 51, "轻度污染"],
        [8, 89, 65, 78, 0.86, 51, 26, "良"],
        [9, 53, 33, 47, 0.64, 50, 17, "良"],
        [10, 80, 55, 80, 1.01, 75, 24, "良"],
        [11, 117, 81, 124, 1.03, 45, 24, "轻度污染"],
        [12, 99, 71, 142, 1.1, 62, 42, "良"],
        [13, 95, 69, 130, 1.28, 74, 50, "良"],
        [14, 116, 87, 131, 1.47, 84, 40, "轻度污染"],
    ]
    data2 = [[a, b - 10, c + 5, *gg] for a, b, c, *gg in data]
    (
        Parallel()
        .add_schema(
            [
                opts.ParallelAxisOpts(dim=0, name="data"),
                opts.ParallelAxisOpts(dim=1, name="AQI"),
                opts.ParallelAxisOpts(dim=2, name="PM2.5"),
                opts.ParallelAxisOpts(dim=3, name="PM10"),
                opts.ParallelAxisOpts(dim=4, name="CO"),
                opts.ParallelAxisOpts(dim=5, name="NO2"),
                opts.ParallelAxisOpts(dim=6, name="CO2"),
                opts.ParallelAxisOpts(
                    dim=7,
                    name="等级",
                    type_="category",
                    data=["优", "良", "轻度污染", "中度污染", "重度污染", "严重污染"],
                ),
            ]
        )
        .add(series_name="cat1", data=data)
        .add(series_name="cat2", data=data2)
        .set_global_opts(title_opts=opts.TitleOpts(title="Parallel-Category"))
        .render(fname)
    )


# %%
# 1. 环形：设置 `radius` 半径范围
# 2. 多饼：添加 `center` 不同多组数据
# 3. 嵌套：添加 `radius` 半径范围不同多组数据
# 4. 玫瑰：指定 `rosetype` 参数
def chart_pie():
    from pyecharts.charts import Pie

    inner_x_data = ["直达", "营销广告", "搜索引擎"]
    inner_y_data = [335, 679, 1548]
    inner_data_pair = [list(z) for z in zip(inner_x_data, inner_y_data)]

    outer_x_data = ["直达", "营销广告", "搜索引擎", "邮件营销",
                    "联盟广告", "视频广告", "百度", "谷歌", "必应", "其他"]
    outer_y_data = [335, 310, 234, 135, 1048, 251, 147, 102]
    outer_data_pair = [list(z) for z in zip(outer_x_data, outer_y_data)]

    (
        Pie()
        .add(
            series_name="访问来源",
            data_pair=inner_data_pair,
            radius=[0, "30%"],
            rosetype="area",
            label_opts=opts.LabelOpts(position="inner"),
        )
        .add(
            series_name="访问来源",
            radius=["40%", "55%"],
            data_pair=outer_data_pair,
            rosetype="radius",
            label_opts=opts.LabelOpts(
                position="outside",
                formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
                background_color="#eee",
                border_color="#aaa",
                border_width=1,
                border_radius=4,
                rich={
                    "a": {"color": "#999", "lineHeight": 22, "align": "center"},
                    "abg": {
                        "backgroundColor": "#e3e3e3",
                        "width": "100%",
                        "align": "right",
                        "height": 22,
                        "borderRadius": [4, 4, 0, 0],
                    },
                    "hr": {
                        "borderColor": "#aaa",
                        "width": "100%",
                        "borderWidth": 0.5,
                        "height": 0,
                    },
                    "b": {"fontSize": 16, "lineHeight": 33},
                    "per": {
                        "color": "#eee",
                        "backgroundColor": "#334455",
                        "padding": [2, 4],
                        "borderRadius": 2,
                    },
                },
            ),
        )
        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_left="left", orient="vertical")
        )
        .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
            )
        )
        .render(fname)
    )


# %%
def chart_polar():
    from pyecharts.charts import Polar

    pass


# %%
def chart_radar():
    from pyecharts.charts import Radar

    value_bj = [
        [55, 9, 56, 0.46, 18, 6, 1],
        [25, 11, 21, 0.65, 34, 9, 2],
        [56, 7, 63, 0.3, 14, 5, 3],
        [33, 7, 29, 0.33, 16, 6, 4],
        [42, 24, 44, 0.76, 40, 16, 5],
        [82, 58, 90, 1.77, 68, 33, 6],
        [74, 49, 77, 1.46, 48, 27, 7],
        [78, 55, 80, 1.29, 59, 29, 8],
        [267, 216, 280, 4.8, 108, 64, 9],
        [185, 127, 216, 2.52, 61, 27, 10],
        [39, 19, 38, 0.57, 31, 15, 11],
        [41, 11, 40, 0.43, 21, 7, 12],
    ]
    value_sh = [
        [91, 45, 125, 0.82, 34, 23, 1],
        [65, 27, 78, 0.86, 45, 29, 2],
        [83, 60, 84, 1.09, 73, 27, 3],
        [109, 81, 121, 1.28, 68, 51, 4],
        [106, 77, 114, 1.07, 55, 51, 5],
        [109, 81, 121, 1.28, 68, 51, 6],
        [106, 77, 114, 1.07, 55, 51, 7],
        [89, 65, 78, 0.86, 51, 26, 8],
        [53, 33, 47, 0.64, 50, 17, 9],
        [80, 55, 80, 1.01, 75, 24, 10],
        [117, 81, 124, 1.03, 45, 24, 11],
        [99, 71, 142, 1.1, 62, 42, 12],
    ]
    c_schema = [
        {"name": "AQI", "max": 300, "min": 5},
        {"name": "PM2.5", "max": 250, "min": 20},
        {"name": "PM10", "max": 300, "min": 5},
        {"name": "CO", "max": 5},
        {"name": "NO2", "max": 200},
        {"name": "SO2", "max": 100},
    ]
    (
        Radar()
        .add_schema(schema=c_schema, shape="circle")
        .add("北京", value_bj, color="#f9713c")
        .add("上海", value_sh, color="#b3e4a1")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            legend_opts=opts.LegendOpts(selected_mode="single"),
            title_opts=opts.TitleOpts(title="Radar-空气质量"),
        )
        .render(fname)
    )


# %%
def charts_sankey():
    from pyecharts.charts import Sankey

    nodes = [
        {"name": "cat1"},
        {"name": "cat2"},
        {"name": "cat3"},
        {"name": "cat4"},
        {"name": "cat5"},
        {"name": "cat6"},
    ]
    links = [
        {"source": "cat1", "target": "cat2", "value": 10},
        {"source": "cat2", "target": "cat3", "value": 15},
        {"source": "cat3", "target": "cat4", "value": 20},
        {"source": "cat5", "target": "cat6", "value": 25},
        {"source": "cat2", "target": "cat6", "value": 25},
    ]
    (
        Sankey()
        .add(
            series_name="sankey",
            nodes=nodes,
            links=links,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
            label_opts=opts.LabelOpts(position="right"),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="Sankey-基本示例"))
        .render(fname)
    )


# %%
def charts_sunburst():
    from pyecharts.charts import Sunburst

    data = [
        opts.SunburstItem(
            name="Grandpa",
            children=[
                opts.SunburstItem(
                    name="Uncle Leo",
                    value=15,
                    children=[
                        opts.SunburstItem(name="Cousin Jack", value=2),
                        opts.SunburstItem(
                            name="Cousin Mary",
                            value=5,
                            children=[opts.SunburstItem(name="Jackson", value=2)],
                        ),
                        opts.SunburstItem(name="Cousin Ben", value=4),
                    ],
                ),
                opts.SunburstItem(
                    name="Father",
                    value=10,
                    children=[
                        opts.SunburstItem(name="Me", value=5),
                        opts.SunburstItem(name="Brother Peter", value=1),
                    ],
                ),
            ],
        ),
        opts.SunburstItem(
            name="Nancy",
            children=[
                opts.SunburstItem(
                    name="Uncle Nike",
                    children=[
                        opts.SunburstItem(name="Cousin Betty", value=1),
                        opts.SunburstItem(name="Cousin Jenny", value=2),
                    ],
                )
            ],
        ),
    ]

    (
        Sunburst(init_opts=opts.InitOpts(width="1000px", height="600px"))
        .add(series_name="", data_pair=data, radius=[0, "90%"])
        .set_global_opts(title_opts=opts.TitleOpts(title="Sunburst-基本示例"))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}"))
        .render(fname)
    )


# %%
def charts_river():
    from pyecharts.charts import ThemeRiver

    x_data = ["DQ", "TY", "SS", "QG", "SY", "DD"]
    y_data = [
        ["2015/11/08", 10, "DQ"],
        ["2015/11/09", 15, "DQ"],
        ["2015/11/10", 35, "DQ"],
        ["2015/11/11", 38, "DQ"],
        ["2015/11/12", 22, "DQ"],
        ["2015/11/13", 16, "DQ"],
        ["2015/11/14", 7, "DQ"],
        ["2015/11/15", 2, "DQ"],
        ["2015/11/16", 17, "DQ"],
        ["2015/11/17", 33, "DQ"],
        ["2015/11/18", 40, "DQ"],
        ["2015/11/19", 32, "DQ"],
        ["2015/11/20", 26, "DQ"],
        ["2015/11/21", 35, "DQ"],
        ["2015/11/22", 40, "DQ"],
        ["2015/11/23", 32, "DQ"],
        ["2015/11/24", 26, "DQ"],
        ["2015/11/25", 22, "DQ"],
        ["2015/11/26", 16, "DQ"],
        ["2015/11/27", 22, "DQ"],
        ["2015/11/28", 10, "DQ"],
        ["2015/12/08", 35, "TY"],
        ["2015/12/09", 36, "TY"],
        ["2015/12/10", 37, "TY"],
        ["2015/12/11", 22, "TY"],
        ["2015/12/12", 24, "TY"],
        ["2015/12/13", 26, "TY"],
        ["2015/12/14", 34, "TY"],
        ["2015/12/15", 21, "TY"],
        ["2015/12/16", 18, "TY"],
        ["2015/12/17", 45, "TY"],
        ["2015/12/18", 32, "TY"],
        ["2015/12/19", 35, "TY"],
        ["2015/12/20", 30, "TY"],
        ["2015/12/21", 28, "TY"],
        ["2015/12/22", 27, "TY"],
        ["2015/12/23", 26, "TY"],
        ["2015/12/24", 15, "TY"],
        ["2015/12/25", 30, "TY"],
        ["2015/12/26", 35, "TY"],
        ["2015/12/27", 42, "TY"],
        ["2015/12/28", 42, "TY"],
        ["2015/11/08", 21, "SS"],
        ["2015/11/09", 25, "SS"],
        ["2015/11/10", 27, "SS"],
        ["2015/11/11", 23, "SS"],
        ["2015/11/12", 24, "SS"],
        ["2015/11/13", 21, "SS"],
        ["2015/11/14", 35, "SS"],
        ["2015/11/15", 39, "SS"],
        ["2015/11/16", 40, "SS"],
        ["2015/11/17", 36, "SS"],
        ["2015/11/18", 33, "SS"],
        ["2015/11/19", 43, "SS"],
        ["2015/11/20", 40, "SS"],
        ["2015/11/21", 34, "SS"],
        ["2015/11/22", 28, "SS"],
        ["2015/11/23", 26, "SS"],
        ["2015/11/24", 37, "SS"],
        ["2015/11/25", 41, "SS"],
        ["2015/11/26", 46, "SS"],
        ["2015/11/27", 47, "SS"],
        ["2015/11/28", 41, "SS"],
        ["2015/11/08", 10, "QG"],
        ["2015/11/09", 15, "QG"],
        ["2015/11/10", 35, "QG"],
        ["2015/11/11", 38, "QG"],
        ["2015/11/12", 22, "QG"],
        ["2015/11/13", 16, "QG"],
        ["2015/11/14", 7, "QG"],
        ["2015/11/15", 2, "QG"],
        ["2015/11/16", 17, "QG"],
        ["2015/11/17", 33, "QG"],
        ["2015/11/18", 40, "QG"],
        ["2015/11/19", 32, "QG"],
        ["2015/11/20", 26, "QG"],
        ["2015/11/21", 35, "QG"],
        ["2015/11/22", 40, "QG"],
        ["2015/11/23", 32, "QG"],
        ["2015/11/24", 26, "QG"],
        ["2015/11/25", 22, "QG"],
        ["2015/11/26", 16, "QG"],
        ["2015/11/27", 22, "QG"],
        ["2015/11/28", 10, "QG"],
        ["2015/11/08", 10, "SY"],
        ["2015/11/09", 15, "SY"],
        ["2015/11/10", 35, "SY"],
        ["2015/11/11", 38, "SY"],
        ["2015/11/12", 22, "SY"],
        ["2015/11/13", 16, "SY"],
        ["2015/11/14", 7, "SY"],
        ["2015/11/15", 2, "SY"],
        ["2015/11/16", 17, "SY"],
        ["2015/11/17", 33, "SY"],
        ["2015/11/18", 40, "SY"],
        ["2015/11/19", 32, "SY"],
        ["2015/11/20", 26, "SY"],
        ["2015/11/21", 35, "SY"],
        ["2015/11/22", 4, "SY"],
        ["2015/11/23", 32, "SY"],
        ["2015/11/24", 26, "SY"],
        ["2015/11/25", 22, "SY"],
        ["2015/11/26", 16, "SY"],
        ["2015/11/27", 22, "SY"],
        ["2015/11/28", 10, "SY"],
        ["2015/11/08", 10, "DD"],
        ["2015/11/09", 15, "DD"],
        ["2015/11/10", 35, "DD"],
        ["2015/11/11", 38, "DD"],
        ["2015/11/12", 22, "DD"],
        ["2015/11/13", 16, "DD"],
        ["2015/11/14", 7, "DD"],
        ["2015/11/15", 2, "DD"],
        ["2015/11/16", 17, "DD"],
        ["2015/11/17", 33, "DD"],
        ["2015/11/18", 4, "DD"],
        ["2015/11/19", 32, "DD"],
        ["2015/11/20", 26, "DD"],
        ["2015/11/21", 35, "DD"],
        ["2015/11/22", 40, "DD"],
        ["2015/11/23", 32, "DD"],
        ["2015/11/24", 26, "DD"],
        ["2015/11/25", 22, "DD"],
        ["2015/11/26", 16, "DD"],
        ["2015/11/27", 22, "DD"],
        ["2015/11/28", 10, "DD"],
    ]

    (
        ThemeRiver()
        .add(
            series_name=x_data,
            data=y_data,
            singleaxis_opts=opts.SingleAxisOpts(
                pos_top="50", pos_bottom="50", type_="time"
            ),
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line")
        )
        .render(fname)
    )


# %%
def gen_vals():
    labels = Faker.cars + Faker.country + Faker.dogs + Faker.provinces
    xlabels = labels[:20]
    val1 = np.random.randint(10, 100, 20)
    val2 = np.random.randint(10, 100, 20)
    percent1 = val1 / (val1 + val2)
    percent2 = val2 / (val1 + val2)

    list1 = [{"value": int(val), "percent": float(per)}
             for val, per in zip(val1, percent1)]
    list2 = [{"value": int(val), "percent": float(per)}
             for val, per in zip(val2, percent2)]
    sums = (val1 + val2 + 20).tolist()

    return xlabels, list1, list2, sums


# %%
# 1. 自定义数据标签：`add` 添加带辅助值数据，`opts.LabelOpts` 设置标签取值逻辑
# 2. 堆叠：`stack` 指定堆叠逻辑
def rect_bar():
    from pyecharts.charts import Bar

    colors = ["#5793f3", "#d14a61", "#675bba"]
    xlabels, list1, list2, sums = gen_vals()
    bar = (
        Bar(init_opts=opts.InitOpts(
            theme=ThemeType.LIGHT,
            width="900px",
            height="600px",
        ))
        .add_xaxis(xlabels)
        # `stack` 指定堆叠逻辑
        .add_yaxis("Ser1", list1, stack="stack1", category_gap="50%")
        .add_yaxis("Ser2", list2, stack="stack1", category_gap="50%")
        # .extend_axis(
        #     yaxis=opts.AxisOpts(
        #         name="Y1",
        #         type_="value",
        #         min_=10,
        #         max_=100,
        #         position="right",
        #         axisline_opts=opts.AxisLineOpts(
        #             linestyle_opts=opts.LineStyleOpts(color=colors[0])
        #         ),
        #         axislabel_opts=opts.LabelOpts(formatter="{value} Unit"),
        #     )
        # )
        # .extend_axis(
        #     yaxis=opts.AxisOpts(
        #         name="Y2",
        #         type_="value",
        #         min_=10,
        #         max_=100,
        #         position="right",
        #         offset=70,
        #         axisline_opts=opts.AxisLineOpts(
        #             linestyle_opts=opts.LineStyleOpts(color=colors[2])
        #         ),
        #         axislabel_opts=opts.LabelOpts(formatter="{value} Unit"),
        #     )
        # )
        .set_global_opts(
            # 轴设置：轴名、旋转标签
            xaxis_opts=opts.AxisOpts(name="X-轴",
                                     axislabel_opts=opts.LabelOpts(rotate=-45)),
            title_opts=opts.TitleOpts(title="Bar-旋转X轴标签",
                                      subtitle="解决标签名字过长的问题"),
            # 添加笔刷
            brush_opts=opts.BrushOpts(),
            # 添加聚焦轴
            # datazoom_opts=opts.DataZoomOpts(),
            # 添加工具箱
            toolbox_opts=opts.ToolboxOpts(),
            # 添加指示器
            tooltip_opts=opts.TooltipOpts(trigger="axis",
                                          axis_pointer_type="cross"),
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(
                position="right",
                # 手动设置标签格式、取值逻辑
                formatter=JsCode(
                    "function(x){return Number(x.data.percent * 100).toFixed() + '%';}"
                ),
            )
        )
    )
    bar.render(fname)

    return bar


# %%
def rect_line():
    from pyecharts.charts import Line

    xlabels, list1, list2, sums = gen_vals()
    line = (
        Line()
        .add_xaxis(xaxis_data=xlabels)
        .add_yaxis(
            series_name="Sums",
            y_axis=sums,
            # 平滑曲线
            is_smooth=False,
            # 阶梯折线
            is_step=False,
            # 标记点
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="最大值"),
                    opts.MarkPointItem(type_="min", name="最小值"),
                ]
            ),
            # 标记线
            markline_opts=opts.MarkLineOpts(
                data=[
                    opts.MarkLineItem(type_="average", name="平均值"),
                    opts.MarkLineItem(symbol="none", x="90%", y="max"),
                    opts.MarkLineItem(symbol="circle", type_="max", name="最高点"),
                ]
            ),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(name="X-轴",
                                     axislabel_opts=opts.LabelOpts(rotate=-45)),
        )
        .set_series_opts(
            # 面积图
            areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
        )
    )
    line.render(fname)

    return line


# %%
def grid_bar_line():
    from pyecharts.charts import Grid

    bar = rect_bar()
    line = rect_line()
    # 添加 Y 轴后将渲染失败
    (
        Grid(init_opts=opts.InitOpts(width="1024px", height="768px"))
        .add(
            chart=bar,
            grid_opts=opts.GridOpts(pos_left=50, pos_right=50, height="35%")
        )
        .add(
            chart=line,
            grid_opts=opts.GridOpts(pos_left=50, pos_right=50,
                                    pos_top="55%", height="35%")
        )
        .render(fname)
    )


# %%
def timeline_bar_line():
    from pyecharts.charts import Timeline

    bar = rect_bar()
    line = rect_line()

    tl = Timeline()
    tl.add(bar, "2023年")
    tl.add(line, "2024年")
    tl.render(fname)
