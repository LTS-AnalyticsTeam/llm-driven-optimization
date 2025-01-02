from tsp_with_constraints.simulator import (
    define_problem,
    is_valid_tour,
    obj_func,
    vizualize,
)
import pytest
from pathlib import Path


N = 20
OUTPUT_DIR = Path(__file__).parent / "__output__" / "simulator"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def test_simulator():
    g = define_problem(
        nodes_num=10, seed=0, time_windows_constraints=True, precedence_constraints=True
    )
    vizualize(g, None, path=f"{OUTPUT_DIR}/tsp_problem.png")
    tour = list(g.nodes)
    vizualize(g, tour, path=f"{OUTPUT_DIR}/tsp_solution.png")

    print("制約条件の確認")
    for node in g.nodes:
        print(f'{node}: {g.nodes[node]["time_window"]}')
    print(g.graph["precedence_pairs"])


@pytest.mark.parametrize("nodes_num", [10, 20, 50])
def test_is_valid_tour(nodes_num):
    g = define_problem(nodes_num=nodes_num, seed=0)
    tour = list(g.nodes)
    assert is_valid_tour(g, tour)[0] == True

    # ツアーの長さが不正
    tour = list(g.nodes)[:-1]
    assert is_valid_tour(g, tour)[0] == False

    # ツアーの訪問ノードに重複がある
    tour = list(g.nodes) + [0]
    assert is_valid_tour(g, tour)[0] == False

    # 時間枠制約を超過
    g.nodes[1]["time_window"] = (0, 0)
    g[0][1]["weight"] = 100
    tour = list(g.nodes)
    assert is_valid_tour(g, tour)[0] == False

    # 時間枠制約を満たす
    g.nodes[1]["time_window"] = (0, 100)
    tour = list(g.nodes)
    assert is_valid_tour(g, tour)[0] == True

    # 順序制約を追加し、不正ツアーを検証
    g.graph["precedence_pairs"] = [{"before": 2, "after": 3}]
    invalid_tour = list(g.nodes)
    invalid_tour[2], invalid_tour[3] = invalid_tour[3], invalid_tour[2]
    assert is_valid_tour(g, invalid_tour)[0] == False

    # 正しい順序のツアーを検証
    valid_tour = list(g.nodes)
    assert is_valid_tour(g, valid_tour)[0] == True


def test_obj_func():
    g = define_problem(nodes_num=4, seed=0)
    # 任意の重み設定（すべての辺を1.0に設定）
    for u in g.nodes:
        for v in g.nodes:
            if u != v:
                g[u][v]["weight"] = 1.0

    # 期待距離: ノード数4 → (4辺×1.0) = 4.0
    tour = [0, 1, 2, 3]
    assert obj_func(g, tour) == 4.0
