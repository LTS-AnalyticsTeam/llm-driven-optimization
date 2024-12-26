import networkx as nx


def solve(g: nx.Graph, start=0) -> list[int]:
    """
    Nearest Neighbor (NN) を用いて TSP の訪問順序を求める関数。

    Parameters:
        g (nx.Graph): TSP 用のグラフ
        start (int): 開始ノード

    Returns:
        list[int]: 訪問ノードの順序を表すリスト
    """
    visited = [start]
    current = start
    nodes = list(g.nodes())
    unvisited = set(nodes) - {start}

    while unvisited:
        # 現在ノードcurrentから一番近い未訪問ノードを探す
        next_node = min(unvisited, key=lambda node: g[current][node]["weight"])
        visited.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    return visited
