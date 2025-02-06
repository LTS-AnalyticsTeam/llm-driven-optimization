from tsp.simulator import Simulator
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import random
import os
from matplotlib.patches import Rectangle


class SimulatorImage(Simulator):

    area_partition = False
    restricted_area = False

    def _define_problem(self) -> None:
        g: nx.Graph = nx.complete_graph(self.nodes_num, create_using=nx.DiGraph)
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
        tmp_dir = Path(__file__).parent / "__tmp__"
        tmp_dir.mkdir(exist_ok=True, parents=True)
        self.vizualize_only_nodes(tmp_dir / "only_nodes.png")

        for u, v in self.g.edges:
            self.g.edges[u, v]["is_allowed"] = True
            self.g.edges[u, v]["reason"] = ""

        if self.area_partition:
            print("エリア分割を行います。分割する中心座標を入力してください。")
            self.ap_x_center = float(input("x:"))
            self.ap_y_center = float(input("y:"))
            for u in self.g.nodes:
                x = self.g.nodes[u]["x"]
                y = self.g.nodes[u]["y"]
                if x > self.ap_x_center:
                    if y > self.ap_y_center:
                        self.g.nodes[u]["area"] = "A"
                    else:
                        self.g.nodes[u]["area"] = "B"
                else:
                    if y < self.ap_y_center:
                        self.g.nodes[u]["area"] = "C"
                    else:
                        self.g.nodes[u]["area"] = "D"

            allowed_paths = [
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("D", "A"),
                ("A", "A"),
                ("B", "B"),
                ("C", "C"),
                ("D", "D"),
            ]
            for u, v in self.g.edges:
                u_area = self.g.nodes[u]["area"]
                v_area = self.g.nodes[v]["area"]
                if (u_area, v_area) not in allowed_paths:
                    self.g.edges[u, v]["is_allowed"] = False
                    self.g.edges[u, v]["reason"] = ",".join(
                        [self.g.edges[u, v]["reason"], "エリア移動順序の制約"]
                    )

        if self.restricted_area:

            def is_in_restricted_area(x, y, mergine=0.01):
                return (
                    self.ra_x - mergine <= x <= self.ra_x + self.ra_w + mergine
                ) and (self.ra_y - mergine <= y <= self.ra_y + self.ra_h + mergine)

            def is_intermidiate(
                dot_1: tuple[float, float],
                dot_2: tuple[float, float],
                dot_in: tuple[float, float],
            ):
                return min(dot_1[0], dot_2[0]) <= dot_in[0] <= max(
                    dot_1[0], dot_2[0]
                ) and min(dot_1[1], dot_2[1]) <= dot_in[1] <= max(dot_1[1], dot_2[1])

            while True:
                print(
                    "制約エリアを設定します。左下の座標と幅、高さを入力してください。"
                )
                self.ra_x = float(input("x:"))
                self.ra_y = float(input("y:"))
                self.ra_w = float(input("w:"))
                self.ra_h = float(input("h:"))
                is_out_list = []
                for u in self.g.nodes:
                    x = self.g.nodes[u]["x"]
                    y = self.g.nodes[u]["y"]
                    if is_in_restricted_area(x, y):
                        is_out_list.append(False)
                    else:
                        is_out_list.append(True)
                if all(is_out_list):
                    # 制約エリアに全ての頂点が含まれていない場合終了
                    break

            ra_points = (
                (self.ra_x, self.ra_y),
                (self.ra_x + self.ra_w, self.ra_y),
                (self.ra_x, self.ra_y + self.ra_h),
                (self.ra_x + self.ra_w, self.ra_y + self.ra_h),
            )

            ra_points_pair = (
                (ra_points[0], ra_points[1]),
                (ra_points[1], ra_points[3]),
                (ra_points[3], ra_points[2]),
                (ra_points[2], ra_points[0]),
            )

            for u, v in self.g.edges:
                u_point = self.g.nodes[u]["x"], self.g.nodes[u]["y"]
                v_point = self.g.nodes[v]["x"], self.g.nodes[v]["y"]
                for ra_point1, ra_point2 in ra_points_pair:
                    intersect = line_intersection(
                        *u_point, *v_point, *ra_point1, *ra_point2
                    )
                    # 制約エリアとの交点がある場合
                    if is_in_restricted_area(*intersect):
                        # 交点がエッジの内側にある場合
                        if is_intermidiate(u_point, v_point, intersect):
                            self.g.edges[u, v]["is_allowed"] = False
                            self.g.edges[u, v]["reason"] = ",".join(
                                [self.g.edges[u, v]["reason"], "立入禁止エリアの制約"]
                            )

        self.vizualize_nodes(tmp_dir / "nodes.png")
        return None

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

        if self.area_partition:
            ax = plt.gca()
            # 右上: A
            a_rect = Rectangle(
                (self.ap_x_center, self.ap_y_center),
                1 - self.ap_x_center,
                1 - self.ap_y_center,
                facecolor="lightblue",
                alpha=0.3,
            )
            ax.add_patch(a_rect)
            plt.text(
                a_rect.get_x() + a_rect.get_width() / 2,
                a_rect.get_y() + a_rect.get_height() / 2,
                "A",
                fontsize=24,
                fontweight="bold",
                color="blue",
                horizontalalignment="center",
                verticalalignment="center",
            )

            # 右下: B
            b_rect = Rectangle(
                (self.ap_x_center, 0.0),
                1 - self.ap_x_center,
                self.ap_y_center,
                facecolor="lightgreen",
                alpha=0.3,
            )
            ax.add_patch(b_rect)
            plt.text(
                b_rect.get_x() + b_rect.get_width() / 2,
                b_rect.get_y() + b_rect.get_height() / 2,
                "B",
                fontsize=24,
                fontweight="bold",
                color="green",
                horizontalalignment="center",
                verticalalignment="center",
            )
            # 左下: C
            c_rect = Rectangle(
                (0.0, 0.0),
                self.ap_x_center,
                self.ap_y_center,
                facecolor="lightyellow",
                alpha=0.3,
            )

            ax.add_patch(c_rect)
            plt.text(
                c_rect.get_x() + c_rect.get_width() / 2,
                c_rect.get_y() + c_rect.get_height() / 2,
                "C",
                fontsize=24,
                fontweight="bold",
                color="orange",
                horizontalalignment="center",
                verticalalignment="center",
            )
            # 左上: D
            d_rect = Rectangle(
                (0.0, self.ap_y_center),
                self.ap_x_center,
                1 - self.ap_y_center,
                facecolor="lightpink",
                alpha=0.3,
            )

            ax.add_patch(d_rect)
            plt.text(
                d_rect.get_x() + d_rect.get_width() / 2,
                d_rect.get_y() + d_rect.get_height() / 2,
                "D",
                fontsize=24,
                fontweight="bold",
                color="red",
                horizontalalignment="center",
                verticalalignment="center",
            )

        if self.restricted_area:
            ax = plt.gca()
            rect = Rectangle(
                (self.ra_x, self.ra_y),
                self.ra_w,
                self.ra_h,
                facecolor="yellow",
                edgecolor="black",
                hatch="//",
                alpha=1.0,
            )
            # Axes にパッチを追加
            ax.add_patch(rect)

        plt.xlim(*self.x_range)
        plt.ylim(*self.y_range)
        plt.axis("on")
        plt.tick_params(labelbottom=True, labelleft=True, bottom=True, left=True)
        plt.savefig(path)
        plt.close()
        return None

    def vizualize_only_nodes(self, path: str) -> None:
        super().vizualize(None, path)
        return None

    def is_valid_tour(self, tour: list[int], log=False) -> tuple[bool, str]:
        """ツアーが有効であるかを判定します。"""
        is_valid, messages = super().is_valid_tour(tour)
        message_list = messages.split(",") if messages != "" else []
        try:
            for u, v in zip(tour, tour[1:] + [tour[0]]):

                edge = self.g.edges[u, v]
                if not edge["is_allowed"]:
                    is_valid = False
                    message_list.append(
                        f"エッジ {(u, v)} は移動許可されていません。理由: {edge['reason']}"
                    )
        except:
            is_valid = False
            message_list.append("予期せぬエラーが発生しました。")

        return is_valid, ",".join(message_list)


def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    2つの直線（A, B）が交わる点を返す関数。
    直線 A は (x1, y1), (x2, y2)
    直線 B は (x3, y3), (x4, y4)

    戻り値:
        (x, y): 交点をタプルで返す
        None: 平行 or 同一直線で交点を一意に定められない場合
    """

    # 方向ベクトル
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3

    # 外積(= 2つのベクトルの平行判定に使用)
    # これが 0 の場合、平行または同一直線
    cross = dx1 * dy2 - dy1 * dx2

    if abs(cross) < 1e-12:
        # 平行 or 同一直線の可能性

        # 例えば (x1, y1) ともう片方の直線ベクトル (dx2, dy2) の外積をとって
        # 同一直線かどうかを確認する
        cross_collinear = (x3 - x1) * dy1 - (y3 - y1) * dx1

        if abs(cross_collinear) < 1e-12:
            # 同一直線なので、交点は一意に定まらない
            return None
        else:
            # 単に平行なだけなので交点は存在しない
            return None

    # ここからは cross != 0 なので交点を求められる
    # パラメータ t を求める
    # (x3 - x1, y3 - y1) と (dx2, dy2) の外積 / cross
    t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / cross

    # t を直線 A のパラメータに代入して交点 (x, y) を得る
    x = x1 + t * dx1
    y = y1 + t * dy1

    return (x, y)
