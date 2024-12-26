from pathlib import Path
from tsp.simulator import define_problem, vizualize

N = 10
OUTPUT_DIR = Path(__file__).parent / "__output__" / "simulator"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def test_simulator():
    g = define_problem()
    vizualize(g, None, path=f"{OUTPUT_DIR}/tsp_problem.png")
    tour = list(g.nodes)
    vizualize(g, tour, path=f"{OUTPUT_DIR}/tsp_solution.png")
