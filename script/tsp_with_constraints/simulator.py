import tsp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


def define_problem(
    nodes_num: int = 10,
    seed: int = 0,
    time_windows_constraints=False,
    precedence_constraints=False,
) -> nx.Graph:
    random.seed(seed)
    g = tsp.simulator.define_problem(nodes_num, seed)

    # 時間枠制約と順序制約のデフォルト値の設定（スタートは0）
    for node in g.nodes:
        g.nodes[node]["time_window"] = None
    g.nodes[0]["time_window"] = (0, 0)
    g.graph["precedence_pairs"] = []

    if time_windows_constraints:
        time_window_num = round(nodes_num / 10) * 3
        tour = tsp.solver.fi.solve(g)
        obj_value = tsp.simulator.obj_func(g, tour)
        time_window_width = obj_value / time_window_num
        for i, node in enumerate(random.sample(list(g.nodes), time_window_num)):
            g.nodes[node]["time_window"] = (
                i * time_window_width,
                (i + 1) * time_window_width,
            )

    if precedence_constraints:
        precedence_pair_num = round(nodes_num * 3 / 10)
        selected_nodes = random.sample(list(g.nodes), precedence_pair_num * 2)
        g.graph["precedence_pairs"] = [
            {"before": selected_nodes[i], "after": selected_nodes[i + 1]}
            for i in range(0, precedence_pair_num * 2, 2)
        ]

    return g


def is_valid_tour(g: nx.Graph, tour: list[int]) -> tuple[bool, str]:
    """ツアーが有効であるかを判定します。"""
    # 決定変数のチェック
    n = len(g.nodes)
    if len(tour) != n:
        return False, "ツアーの長さが不正です。"
    if set(range(n)) != set(tour):
        return False, "ツアーの訪問ノードに重複があり不正です。"

    # 時間枠制約のチェック
    total_time = 0
    for i in g.nodes:
        if i != 0:
            total_time += g[tour[i - 1]][tour[i]]["weight"]
        time_window = g.nodes[tour[i]]["time_window"]
        if time_window is not None:
            start, end = time_window
            if start <= total_time <= end:
                pass
            else:
                return False, f"ノード {tour[i]} の時間枠を超過しています。"

    # 順序制約のチェック
    for pair in g.graph["precedence_pairs"]:
        before_idx = tour.index(pair["before"])
        after_idx = tour.index(pair["after"])
        if before_idx > after_idx:
            return (
                False,
                f"ノード {pair['before']} はノード {pair['after']} より先に訪問される必要があります。",
            )

    return True, "このツアーは有効です。"


def obj_func(g: nx.Graph, tour: list[int]) -> float:
    try:
        assert is_valid_tour(g, tour)
    except:
        return np.nan

    total_time = 0.0
    for i in range(len(tour) - 1):
        u = tour[i]
        v = tour[i + 1]
        total_time += g[u][v]["weight"]
    # 最後の頂点から最初の頂点への移動距離を加算
    total_time += g[tour[-1]][tour[0]]["weight"]
    return total_time


def vizualize(g, tour, path=None):
    tsp.simulator.vizualize(g, tour, path)
    return None
