import pytest
from pathlib import Path
from tsp_with_constraints.simulator import (
    define_problem,
    obj_func,
    vizualize,
    is_valid_tour,
)
from tsp_with_constraints.solver import milp, llm


N = 10
OUTPUT_DIR = Path(__file__).parent / "__output__" / "solver"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def show_result(g, tour, save_path):
    print("tour", tour)
    print("obj_func:", obj_func(g, tour))
    vizualize(g, tour, path=save_path)


def test_milp():
    milp_logs_dir = OUTPUT_DIR / "milp_logs"
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    g = define_problem(
        N, seed=3, time_windows_constraints=True, precedence_constraints=True
    )
    model = milp.TSP(g)
    tour = model.solve(
        log_dir=milp_logs_dir,
        timelimit=60,
        mipgap=0.0,
        tee=True,
    )
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_milp_solution.png")
    is_valid, message = is_valid_tour(g, tour, log=True)
    assert is_valid, message


def test_milp_time_windows_constraints():
    milp_logs_dir = OUTPUT_DIR / "milp_time_windows_constraints_logs"
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    g = define_problem(N, time_windows_constraints=True, precedence_constraints=False)
    model = milp.TSP(g)
    tour = model.solve(
        log_dir=milp_logs_dir,
        timelimit=60,
        mipgap=0.0,
        tee=True,
    )
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_milp_solution.png")
    is_valid, message = is_valid_tour(g, tour, log=True)
    assert is_valid, message


def test_milp_precedence_constraints():
    milp_logs_dir = OUTPUT_DIR / "milp_precedence_constraints_logs"
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    g = define_problem(N, time_windows_constraints=False, precedence_constraints=True)
    model = milp.TSP(g)
    tour = model.solve(
        log_dir=milp_logs_dir,
        timelimit=60,
        mipgap=0.0,
        tee=True,
    )
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_milp_solution.png")
    is_valid, message = is_valid_tour(g, tour, log=True)
    assert is_valid, message


def test_llm():
    g = define_problem(N)
    tour = llm.solve(g)
    show_result(g, tour, f"{OUTPUT_DIR}/tsp_llm_solution.png")
    is_valid, message = is_valid_tour(g, tour, log=True)
    assert is_valid, message
