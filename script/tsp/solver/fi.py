import networkx as nx


def solve(g: nx.Graph) -> list[int]:
    """
    Farthest Insertion (FI) を用いて TSP の訪問順序を求める関数。

    手順:
    1. 最も離れている2点を初期ツアーとして開始
    2. まだツアーに含まれていない全頂点のうち、
       部分ツアーへの「最小挿入コスト」が最も大きい頂点を選ぶ
    3. 選んだ頂点を、部分ツアー内における最小挿入コストの場所に挿入
    4. 全頂点が挿入されるまで繰り返す

    Args:
        g (nx.Graph): エッジに 'weight' が設定されたグラフ

    Returns:
        list[int]: FI によって得られた頂点の訪問順序
    """

    # 距離関数を定義
    def dist(u, v) -> float:
        return g[u][v]["weight"]

    # ノード一覧を取得
    nodes = list(g.nodes)

    # =======================
    # 1. 初期ツアーの構築
    # =======================
    # まず最も距離が離れている2ノードを見つける
    max_d = -1.0
    first, second = None, None
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d_ij = dist(nodes[i], nodes[j])
            if d_ij > max_d:
                max_d = d_ij
                first, second = nodes[i], nodes[j]

    # 部分ツアー (partial_tour) を初期化し、訪問済み集合も用意
    partial_tour = [first, second]
    visited = {first, second}

    # =========================================
    # 2. & 3. 未訪問頂点を順次ツアーに挿入
    # =========================================
    # 全頂点がツアーに含まれるまで繰り返す
    while len(visited) < len(nodes):
        # まだ訪問していないノードを1つ選ぶ
        # 選び方: "最小挿入コスト" が最大になるノードを選ぶ
        farthest_cost = -1.0
        node_to_insert = None

        for node in nodes:
            if node not in visited:
                # この node を部分ツアー内に挿入するときの
                # 「最小挿入コスト」を計算する
                min_insertion_cost = float("inf")

                # 現在の部分ツアーは「サイクル」として扱う
                # （(i, i+1), (最終ノード, 先頭ノード) などを候補とする）
                for i in range(len(partial_tour)):
                    j = (i + 1) % len(partial_tour)  # サイクルを考慮
                    insert_cost = (
                        dist(partial_tour[i], node)
                        + dist(node, partial_tour[j])
                        - dist(partial_tour[i], partial_tour[j])
                    )
                    if insert_cost < min_insertion_cost:
                        min_insertion_cost = insert_cost

                # Farthest Insertion: この最小挿入コストが最も大きいノードを選ぶ
                if min_insertion_cost > farthest_cost:
                    farthest_cost = min_insertion_cost
                    node_to_insert = node

        # node_to_insert を、最小挿入コストの場所に挿入する
        best_increase = float("inf")
        best_pos = 0
        for i in range(len(partial_tour)):
            j = (i + 1) % len(partial_tour)
            cost_ij = (
                dist(partial_tour[i], node_to_insert)
                + dist(node_to_insert, partial_tour[j])
                - dist(partial_tour[i], partial_tour[j])
            )
            if cost_ij < best_increase:
                best_increase = cost_ij
                best_pos = j

        partial_tour.insert(best_pos, node_to_insert)
        visited.add(node_to_insert)

    # 全ノードを挿入したら完成
    return partial_tour
