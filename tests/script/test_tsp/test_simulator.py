from pathlib import Path
from tsp.simulator import Simulator

N = 20
OUTPUT_DIR = Path(__file__).parent / "__output__" / "simulator"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def test_simulator():
    sim = Simulator(N, seed=0)
    sim.vizualize_nodes(path=f"{OUTPUT_DIR}/tsp_problem.png")
    tour = list(sim.g.nodes)
    sim.vizualize(tour, path=f"{OUTPUT_DIR}/tsp_solution.png")
