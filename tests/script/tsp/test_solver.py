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
    is_valid, message = is_valid_tour(g, tour)
    assert is_valid, message


def test_fi():
    g = define_problem(N)
    tour = fi.solve(g)
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_fi_solution.png")
    is_valid, message = is_valid_tour(g, tour)
    assert is_valid, message


def test_milp():
    milp_logs_dir = OUTPUT_DIR / "milp_logs"
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    g = define_problem(N)
    model = milp.TSP(g)
    tour = model.solve(
        log_dir=milp_logs_dir,
        timelimit=60,
        mipgap=0.0,
        tee=True,
    )
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_milp_solution.png")
    is_valid, message = is_valid_tour(g, tour)
    assert is_valid, message


def test_llm():
    g = define_problem(N)
    tour = llm.LLMSolver.solve(g, iter_num=1, llm_model="gpt-4o")
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_llm_solution.png")
    is_valid, message = is_valid_tour(g, tour)
    assert is_valid, message
