import pytest
from pathlib import Path
from tsp_with_constraints.simulator import SimulatorExp
from tsp_with_constraints.solver import milp, llm


N = 10
OUTPUT_DIR = Path(__file__).parent / "__output__" / "solver"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def show_result(sim: SimulatorExp, tour, save_path):
    print("tour", tour)
    print("obj_func:", sim.obj_func(tour))
    sim.vizualize(tour, path=save_path)


def test_milp():
    milp_logs_dir = OUTPUT_DIR / "milp_logs"
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    sim = SimulatorExp(
        N, seed=3, time_windows_constraints=True, precedence_constraints=True
    )
    model = milp.TSPExp(sim.g)
    tour = model.solve(
        log_dir=milp_logs_dir,
        timelimit=60,
        mipgap=0.0,
        tee=True,
    )
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_milp_solution.png")
    is_valid, message = sim.is_valid_tour(tour, log=True)
    assert is_valid, message


def test_milp_time_windows_constraints():
    milp_logs_dir = OUTPUT_DIR / "milp_time_windows_constraints_logs"
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    sim = SimulatorExp(
        N, seed=3, time_windows_constraints=True, precedence_constraints=False
    )

    model = milp.TSPExp(sim.g)
    tour = model.solve(
        log_dir=milp_logs_dir,
        timelimit=60,
        mipgap=0.0,
        tee=True,
    )
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_milp_solution.png")
    is_valid, message = sim.is_valid_tour(tour, log=True)
    assert is_valid, message


def test_milp_precedence_constraints():
    milp_logs_dir = OUTPUT_DIR / "milp_precedence_constraints_logs"
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    sim = SimulatorExp(
        N, seed=0, time_windows_constraints=False, precedence_constraints=True
    )
    model = milp.TSPExp(sim.g)
    tour = model.solve(
        log_dir=milp_logs_dir,
        timelimit=60,
        mipgap=0.0,
        tee=True,
    )
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_milp_solution.png")
    is_valid, message = sim.is_valid_tour(tour, log=True)
    assert is_valid, message


def test_llm():
    sim = SimulatorExp(
        N, seed=3, time_windows_constraints=True, precedence_constraints=True
    )
    tour = llm.LLMSolver.solve(sim, iter_num=1, llm_model="gpt-4o")
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_llm_solution.png")
    is_valid, message = sim.is_valid_tour(tour, log=True)
    print(is_valid, message)  # エラーの可能性があるため、asset文はなし


def test_llm_time_windows_constraints():
    sim = SimulatorExp(N, time_windows_constraints=True, precedence_constraints=False)
    tour = llm.LLMSolver.solve(sim, iter_num=1, llm_model="gpt-4o")
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_llm_solution.png")
    is_valid, message = sim.is_valid_tour(tour, log=True)
    print(is_valid, message)  # エラーの可能性があるため、asset文はなし


def test_llm_precedence_constraints():
    sim = SimulatorExp(N, time_windows_constraints=False, precedence_constraints=True)
    tour = llm.LLMSolver.solve(sim, iter_num=1, llm_model="gpt-4o")
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_llm_solution.png")
    is_valid, message = sim.is_valid_tour(tour, log=True)
    print(is_valid, message)  # エラーの可能性があるため、asset文はなし
