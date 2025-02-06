from tsp_image.simulator import SimulatorImage, line_intersection
from pathlib import Path
import pytest
from unittest.mock import patch
from io import StringIO

N = 30
SEED = 20
OUTPUT_DIR = Path(__file__).parent / "__output__" / "simulator"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


@patch("builtins.input", side_effect=["0.4", "0.525", "0.45", "0.2", "0.05", "0.2"])
@patch("sys.stdout", new_callable=StringIO)
def test_simulator(mock_stdout, mock_input):
    SimulatorImage.area_partition = True
    SimulatorImage.restricted_area = True
    sim = SimulatorImage(nodes_num=N, seed=SEED)
    sim.vizualize_nodes(path=f"{OUTPUT_DIR}/tsp_problem.png")
    tour = list(sim.g.nodes)
    sim.vizualize(tour, path=f"{OUTPUT_DIR}/tsp_solution.png")

    print("制約条件の確認")
    for u, v in sim.g.edges:
        print(f'{(u, v)}: {sim.g.edges[u, v]["is_allowed"]}')


def test_is_valid_tour():
    sim = SimulatorImage(nodes_num=N, seed=SEED)
    # Test a valid tour
    valid_tour = list(sim.g.nodes)
    is_valid, messages = sim.is_valid_tour(valid_tour)
    assert is_valid, f"Expected valid tour but got {messages}"
    # Test an invalid tour (repeated node)
    invalid_tour = [0, 1, 2, 3, 2]
    is_valid, messages = sim.is_valid_tour(invalid_tour)
    assert not is_valid, "Expected invalid tour, but got a valid result"
    # Test empty tour
    empty_tour = []
    is_valid, messages = sim.is_valid_tour(empty_tour)
    assert not is_valid, "Expected invalid empty tour"
    # Test partial set of nodes
    partial_tour = list(sim.g.nodes)[:-1]
    is_valid, messages = sim.is_valid_tour(partial_tour)
    assert not is_valid, "Expected invalid partial tour"
    # Test out-of-range node
    out_of_range_tour = list(sim.g.nodes) + [9999]
    is_valid, messages = sim.is_valid_tour(out_of_range_tour)
    assert not is_valid, "Expected invalid out-of-range tour"


@pytest.mark.parametrize(
    "x1, y1, x2, y2, x3, y3, x4, y4, expected",
    [
        # 1. 典型例: 2本の直線が明確に交わる(対角線の交点)
        (0, 0, 1, 1, 0, 1, 1, 0, (0.5, 0.5)),
        # 2. 平行(または同一直線)な例
        #    ここでは完全に平行（d=0）の場合
        (0, 0, 2, 2, 1, 1, 3, 3, None),
        # 3. X軸と垂直線の交点
        #    y=0とx=5が交わる交点: (5, 0)
        (0, 0, 10, 0, 5, -5, 5, 5, (5.0, 0.0)),
        # 4. 同一直線上にある場合（完全に重なる）
        #    A: (0,0)~(2,2), B: (1,1)~(3,3)
        #    この場合は交点が無限にあるため None を期待
        (0, 0, 2, 2, 1, 1, 3, 3, None),
        # 5. 負の座標を含む: 直線A=(-1, -1)~(1, 1), 直線B=(-1, 1)~(1, -1)
        #    ちょうど原点(0,0)で交わる
        (-1, -1, 1, 1, -1, 1, 1, -1, (0.0, 0.0)),
        # 6. どちらか一方が1点のみ(退化した直線) - (1,1)~(1,1) など
        #    この場合は「実質点と直線」とみなす。
        #    ここでは結果が「点が他線上にあればその座標、なければNone」に
        #    なってもよいが、今回の実装では d=0 になるため None
        (1, 1, 1, 1, 0, 0, 2, 2, None),
        # 7. 近いけれど完全に平行ではない(交点が大きく外側に発生する)
        #    A: y = x, B: y = x + 0.00001
        #    ほぼ平行だが、厳密には一点で交わる
        #    交点は非常に遠い場所で起こる可能性がある
        #    実際に計算すると d ≠ 0 のため、一点が返る
        (0, 0, 100000, 100000, 0, 0.00001, 100000, 100000.00001, None),
        # 8. 延長させると交点があるが、短い部分だけでは交点がない
        (0, 1, 0, 2, 1, 0, -1, 0, (0, 0)),
    ],
)
def test_line_intersection(x1, y1, x2, y2, x3, y3, x4, y4, expected):
    """
    line_intersection 関数のテスト。
    expected が None の場合は結果が None であることを確認し、
    タプルの場合は x, y ともに pytest.approx で近似比較を行う。
    """
    result = line_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
    if expected is None:
        assert result is None, f"Expected None but got {result}"
    else:
        assert result is not None, "Expected a tuple but got None"
        assert result[0] == pytest.approx(
            expected[0]
        ), f"X座標が期待値と異なります: {result[0]} != {expected[0]}"
        assert result[1] == pytest.approx(
            expected[1]
        ), f"Y座標が期待値と異なります: {result[1]} != {expected[1]}"
