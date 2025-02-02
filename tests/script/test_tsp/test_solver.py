import pytest
import traceback
from pathlib import Path
from tsp.simulator import Simulator
from tsp.solver import nn, fi, milp, llm


N = 50
OUTPUT_DIR = Path(__file__).parent / "__output__" / "solver"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def show_result(sim: Simulator, tour, save_path):
    print("tour", tour)
    print("obj_func:", sim.obj_func(tour))
    sim.vizualize(tour, path=save_path)


def test_nn():
    sim = Simulator(nodes_num=N, seed=0)
    tour = nn.solve(sim.g)
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_nn_solution.png")
    is_valid, message = sim.is_valid_tour(tour)
    assert is_valid, message


def test_fi():
    sim = Simulator(nodes_num=N, seed=0)
    tour = fi.solve(sim.g)
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_fi_solution.png")
    is_valid, message = sim.is_valid_tour(tour)
    assert is_valid, message


def test_milp():
    milp_logs_dir = OUTPUT_DIR / "milp_logs"
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    sim = Simulator(nodes_num=N, seed=0)
    model = milp.TSP(sim.g)
    tour = model.solve(
        log_dir=milp_logs_dir,
        timelimit=60,
        mipgap=0.0,
        tee=True,
    )
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_milp_solution.png")
    is_valid, message = sim.is_valid_tour(tour)
    assert is_valid, message


def test_milp_analyze_conflict():
    milp_logs_dir = OUTPUT_DIR / "milp_logs"
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    sim = Simulator(nodes_num=N, seed=0)
    model = milp.TSP(sim.g)
    try:
        model.analyze_conflict(milp_logs_dir)
    except Exception as e:
        traceback.print_exc()


def test_llm_gpt4o():
    sim = Simulator(nodes_num=N, seed=0)
    tour = llm.LLMSolver.solve(sim, iter_num=0, llm_model="gpt-4o")
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_llm_gpt4o_solution.png")
    is_valid, message = sim.is_valid_tour(tour)
    assert is_valid, message


def test_llm_o1():
    sim = Simulator(nodes_num=N, seed=0)
    tour = llm.LLMSolver.solve(sim, iter_num=0, llm_model="o1")
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_llm_o1_solution.png")
    is_valid, message = sim.is_valid_tour(tour)
    assert is_valid, message
