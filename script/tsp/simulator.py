import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


def define_problem(nodes_num: int = 10, seed: int = 0) -> nx.Graph:
    """
    指定された数の頂点を持つ完全グラフを作成し、ランダムな座標を割り当て、
    エッジにユークリッド距離を重みとして設定します。

    Args:
        nodes_num (int): グラフの頂点数
        seed (int): 乱数シード

    Returns:
        nx.Graph: 座標とエッジ重みが設定されたグラフ
    """
    random.seed(seed)
    g: nx.Graph = nx.complete_graph(nodes_num)

    # ランダムな2次元座標の設定
    for n in g.nodes:
        g.nodes[n]["x"] = random.random()
        g.nodes[n]["y"] = random.random()

    # エッジ重みの設定（ユークリッド距離）
    for u, v in g.edges:
        g[u][v]["weight"] = (
            (g.nodes[u]["x"] - g.nodes[v]["x"]) ** 2
            + (g.nodes[u]["y"] - g.nodes[v]["y"]) ** 2
        ) ** (1 / 2)

    # スタート地点の設定
    g.graph["start"] = 0

    return g


def is_valid_tour(g: nx.Graph, tour: list[int]) -> tuple[bool, str]:
    """ツアーが有効であるかを判定します。"""
    n = len(g.nodes)
    if len(tour) != n:
        return False, "ツアーの長さが不正です。"
    if set(range(n)) != set(tour):
        return False, "ツアーの訪問ノードに重複があり不正です。"
    return True, "このツアーは有効です。"


def obj_func(g: nx.Graph, tour: list[int]) -> float:
    """
    グラフとツアー（頂点の訪問順序）を基に合計移動距離を計算して返します。

    Args:
        g (nx.Graph): エッジ重みが設定されたグラフ
        tour (list[int]): 訪問する頂点のリスト

    Returns:
        float: ツアーの総移動距離
    """
    try:
        assert is_valid_tour(g, tour)
    except:
        return np.nan

    total_distance = 0.0
    for i in range(len(tour) - 1):
        u = tour[i]
        v = tour[i + 1]
        total_distance += g[u][v]["weight"]
    # 最後の頂点から最初の頂点への移動距離を加算
    total_distance += g[tour[-1]][tour[0]]["weight"]
    return total_distance


def get_pos(g: nx.Graph) -> dict[int, tuple[float, float]]:
    """
    グラフに割り当てられた頂点座標を辞書形式で返します。

    Args:
        g (nx.Graph): 頂点座標を含むグラフ

    Returns:
        dict[int, tuple[float, float]]: 頂点番号をキーに持つ (x, y) 座標の辞書
    """
    return {n: (g.nodes[n]["x"], g.nodes[n]["y"]) for n in g.nodes}


def vizualize(g: nx.Graph, tour: list[int] = None, path: str = "tsp.png") -> None:
    """
    グラフを可視化し、与えられたツアーを赤色の線で描画します。

    Args:
        g (nx.Graph): 可視化するグラフ
        tour (list[int], optional): 描画するツアーの頂点リスト
        path (str, optional): 出力先ファイルパス
    """
    # 求まったツアーを描画(ルートを赤線で描画)
    plt.figure(figsize=(10, 10), tight_layout=True)
    pos = get_pos(g)
    nx.draw_networkx_nodes(g, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_labels(g, pos, font_size=12, font_weight="bold")

    # 全エッジ(灰色)を描画
    # nx.draw_networkx_edges(g, pos, alpha=0.5, edge_color="gray")
    if tour is not None:
        # 求まったツアーをエッジリストに変換
        tour_edges = list(zip(tour[:-1], tour[1:])) + [(tour[-1], tour[0])]
        # ツアーエッジを赤色で上書き描画
        nx.draw_networkx_edges(g, pos, edgelist=tour_edges, edge_color="red", width=2)

    plt.axis("on")
    plt.tick_params(labelbottom=True, labelleft=True, bottom=True, left=True)
    plt.savefig(path)
    plt.close()
    return None
