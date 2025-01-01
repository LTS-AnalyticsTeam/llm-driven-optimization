import pytest
from pathlib import Path
from tsp.simulator import define_problem, obj_func, vizualize, is_valid_tour
from tsp.solver import nn, fi, milp, llm


N = 30
OUTPUT_DIR = Path(__file__).parent / "__output__" / "solver"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def show_result(g, tour, save_path):
    print("tour", tour)
    print("obj_func:", obj_func(g, tour))
    vizualize(g, tour, path=save_path)


def test_nn():
    g = define_problem(N)
    tour = nn.solve(g)
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_nn_solution.png")
    assert is_valid_tour(g, tour)


def test_fi():
    g = define_problem(N)
    tour = fi.solve(g)
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_fi_solution.png")
    assert is_valid_tour(g, tour)


def test_milp():
    g = define_problem(N)
    model = milp.TSP(g)
    tour = model.solve(
        solver_name="cplex",
        logfile=f"{OUTPUT_DIR}/tsp_milp_cplex.log",
        timelimit=60,
        mipgap=0.0,
        tee=True,
    )
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_milp_solution.png")
    assert is_valid_tour(g, tour)


def test_llm():
    g = define_problem(N)
    tour = llm.solve(g)
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_llm_solution.png")
    assert is_valid_tour(g, tour)
