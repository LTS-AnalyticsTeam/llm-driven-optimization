from tsp_with_constraints.simulator import SimulatorExp
import pytest
from pathlib import Path


N = 20
OUTPUT_DIR = Path(__file__).parent / "__output__" / "simulator"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def test_simulator():
    SimulatorExp.time_windows_constraints = True
    SimulatorExp.precedence_constraints = True
    sim = SimulatorExp(nodes_num=10, seed=0)
    sim.vizualize_nodes(path=f"{OUTPUT_DIR}/tsp_problem.png")
    tour = list(sim.g.nodes)
    sim.vizualize(tour, path=f"{OUTPUT_DIR}/tsp_solution.png")

    print("制約条件の確認")
    for node in sim.g.nodes:
        print(f'{node}: {sim.g.nodes[node]["time_window"]}')
    print(sim.g.graph["precedence_pairs"])


@pytest.mark.parametrize("nodes_num", [10, 20, 50])
def test_is_valid_tour(nodes_num):
    SimulatorExp.time_windows_constraints = False
    SimulatorExp.precedence_constraints = False
    sim = SimulatorExp(
        nodes_num=nodes_num,
        seed=0,
    )
    tour = list(sim.g.nodes)
    assert sim.is_valid_tour(tour)[0] == True

    # ツアーの長さが不正
    tour = list(sim.g.nodes)[:-1]
    assert sim.is_valid_tour(tour)[0] == False

    # ツアーの訪問ノードに重複がある
    tour = list(sim.g.nodes) + [0]
    assert sim.is_valid_tour(tour)[0] == False

    # 時間枠制約を超過
    sim.g.nodes[1]["time_window"] = (0, 0)
    sim.g[0][1]["weight"] = 100
    tour = list(sim.g.nodes)
    assert sim.is_valid_tour(tour)[0] == False

    # 時間枠制約を満たす
    sim.g.nodes[1]["time_window"] = (0, 100)
    tour = list(sim.g.nodes)
    assert sim.is_valid_tour(tour)[0] == True

    # 順序制約を追加し、不正ツアーを検証
    sim.g.graph["precedence_pairs"] = [{"before": 2, "after": 3}]
    invalid_tour = list(sim.g.nodes)
    invalid_tour[2], invalid_tour[3] = invalid_tour[3], invalid_tour[2]
    assert sim.is_valid_tour(invalid_tour)[0] == False

    # 正しい順序のツアーを検証
    valid_tour = list(sim.g.nodes)
    assert sim.is_valid_tour(valid_tour)[0] == True


def test_obj_func():
    SimulatorExp.time_windows_constraints = True
    SimulatorExp.precedence_constraints = True
    sim = SimulatorExp(nodes_num=7, seed=0)
    # 任意の重み設定（すべての辺を1.0に設定）
    for u in sim.g.nodes:
        for v in sim.g.nodes:
            if u != v:
                sim.g[u][v]["weight"] = 1.0

    # 期待距離: ノード数4 → (4辺×1.0) = 4.0
    tour = [0, 1, 2, 3, 4, 5, 6]
    assert sim.obj_func(tour) == 7.0
