import tsp
from tsp.simulator import Simulator
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


class SimulatorExp(Simulator):

    time_windows_constraints: bool = False
    precedence_constraints: bool = False

    def __init__(
        self,
        nodes_num: int = 10,
        seed: int = 0,
    ) -> None:
        random.seed(seed)
        self.seed = seed
        self.nodes_num = nodes_num
        self._define_problem()

    def _define_problem(
        self,
        nodes_num: int = 10,
        seed: int = 0,
    ) -> None:
        super()._define_problem()

        # スタート地点の設定
        self.g.graph["start"] = 0

        # 時間枠制約と順序制約のデフォルト値の設定
        for node in self.g.nodes:
            self.g.nodes[node]["time_window"] = None
        self.g.graph["precedence_pairs"] = []

        if self.time_windows_constraints:
            time_window_num = round(nodes_num / 10) * 3
            tour = tsp.solver.fi.solve(self.g)
            obj_value = self.obj_func(tour)
            time_window_width = obj_value / time_window_num
            optional_nodes = list(self.g.nodes)[1:]  # スタート地点0は除く
            selected_nodes = random.sample(optional_nodes, time_window_num)
            for i, node in enumerate(selected_nodes):
                self.g.nodes[node]["time_window"] = (
                    0,
                    (i + 1) * time_window_width,
                )

        if self.precedence_constraints:
            precedence_pair_num = round(nodes_num * 3 / 10)
            optional_nodes = list(self.g.nodes)[1:]  # スタート地点0は除く
            selected_nodes = random.sample(optional_nodes, precedence_pair_num * 2)
            self.g.graph["precedence_pairs"] = [
                {"before": selected_nodes[i], "after": selected_nodes[i + 1]}
                for i in range(0, precedence_pair_num * 2, 2)
            ]

        return None

    def is_valid_tour(self, tour: list[int], log=False) -> tuple[bool, str]:
        """ツアーが有効であるかを判定します。"""
        # 決定変数のチェック
        n = len(self.g.nodes)
        if len(tour) != n:
            return False, "ツアーの長さが不正です。"
        if set(range(n)) != set(tour):
            return False, "ツアーの訪問ノードに重複があり不正です。"

        # スタート地点のチェック
        if tour[0] != self.g.graph["start"]:
            return False, "スタート地点が不正です。"

        # 時間枠制約のチェック
        if log:
            print(">>> time_windowのチェック")
        total_time = 0
        for u, v in zip(tour[:-1], tour[1:]):
            total_time += self.g[u][v]["weight"]
            time_window = self.g.nodes[v]["time_window"]
            if time_window is not None:
                start, end = time_window
                if start <= total_time <= end:
                    if log:
                        print(f"{v}: {start} <= {total_time} <= {end}")
                else:
                    return False, f"ノード {tour[v]} の時間枠を超過しています。"

        # 順序制約のチェック
        if log:
            print(">>> precedence_pairsのチェック")
        for pair in self.g.graph["precedence_pairs"]:
            before_idx = tour.index(pair["before"])
            after_idx = tour.index(pair["after"])
            if before_idx < after_idx:
                if log:
                    print(
                        f"node[{pair['before']}](=index: {before_idx}) -> node[{pair['after']}](=index: {after_idx})"
                    )
            else:
                return (
                    False,
                    f"ノード {pair['before']} はノード {pair['after']} より先に訪問される必要があります。",
                )

        return True, "このツアーは有効です。"
