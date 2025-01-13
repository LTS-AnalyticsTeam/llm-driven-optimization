import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


class Simulator:

    def __init__(self, nodes_num: int = 10, seed: int = 0):
        random.seed(seed)
        self.seed = seed
        self.nodes_num = nodes_num
        self._define_problem()
        self.opt_tour = None

    def _define_problem(self) -> None:
        """
        指定された数の頂点を持つ完全グラフを作成し、ランダムな座標を割り当て、
        エッジにユークリッド距離を重みとして設定します。

        Args:
            nodes_num (int): グラフの頂点数
            seed (int): 乱数シード

        Returns:
            nx.Graph: 座標とエッジ重みが設定されたグラフ
        """
        g: nx.Graph = nx.complete_graph(self.nodes_num)
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

        self.g = g
        return None

    def _get_pos(self) -> dict[int, tuple[float, float]]:
        """
        グラフに割り当てられた頂点座標を辞書形式で返します。

        Args:
            g (nx.Graph): 頂点座標を含むグラフ

        Returns:
            dict[int, tuple[float, float]]: 頂点番号をキーに持つ (x, y) 座標の辞書
        """
        return {n: (self.g.nodes[n]["x"], self.g.nodes[n]["y"]) for n in self.g.nodes}

    def is_valid_tour(self, tour: list[int]) -> tuple[bool, str]:
        """ツアーが有効であるかを判定します。"""
        is_valid, message_list = True, []
        # 決定変数のチェック
        n = len(self.g.nodes)
        if len(tour) != n:
            is_valid = False
            message_list.append("ツアーの長さが不正です。")

        if set(range(n)) != set(tour):
            is_valid = False
            message_list.append("ツアーの訪問ノードに重複があり不正です。")

        return is_valid, ",".join(message_list)

    def obj_func(self, tour: list[int]) -> float:
        """
        グラフとツアー（頂点の訪問順序）を基に合計移動距離を計算して返します。

        Args:
            g (nx.Graph): エッジ重みが設定されたグラフ
            tour (list[int]): 訪問する頂点のリスト

        Returns:
            float: ツアーの総移動距離
        """
        try:
            assert self.is_valid_tour(tour)
        except:
            return np.nan

        total_distance = 0.0
        for u, v in zip(tour[:-1], tour[1:]):
            total_distance += self.g[u][v]["weight"]
        # 最後の頂点から最初の頂点への移動距離を加算
        total_distance += self.g[tour[-1]][tour[0]]["weight"]
        return total_distance

    def vizualize(self, tour: list[int] = None, path: str = "tsp.png") -> None:
        """
        グラフを可視化し、与えられたツアーを赤色の線で描画します。

        Args:
            g (nx.Graph): 可視化するグラフ
            tour (list[int], optional): 描画するツアーの頂点リスト
            path (str, optional): 出力先ファイルパス
        """
        # 求まったツアーを描画(ルートを赤線で描画)
        plt.figure(figsize=(10, 10), tight_layout=True)
        pos = self._get_pos()
        nx.draw_networkx_nodes(self.g, pos, node_size=500, node_color="lightblue")
        nx.draw_networkx_labels(self.g, pos, font_size=12, font_weight="bold")

        # 全エッジ(灰色)を描画
        # nx.draw_networkx_edges(g, pos, alpha=0.5, edge_color="gray")
        if tour is not None:
            # 求まったツアーをエッジリストに変換
            tour_edges = list(zip(tour[:-1], tour[1:])) + [(tour[-1], tour[0])]
            # ツアーエッジを赤色で上書き描画
            nx.draw_networkx_edges(
                self.g, pos, edgelist=tour_edges, edge_color="red", width=2
            )

        plt.axis("on")
        plt.tick_params(labelbottom=True, labelleft=True, bottom=True, left=True)
        plt.savefig(path)
        plt.close()
        return None

    def vizualize_nodes(self, path: str) -> None:
        self.vizualize(None, path)
        return None
