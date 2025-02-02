import pytest
from pathlib import Path
from tsp_exp.simulator import SimulatorExp
from tsp_exp.solver import milp, llm

OUTPUT_DIR = Path(__file__).parent / "__output__" / "solver"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def show_result(sim: SimulatorExp, tour, save_path):
    print("tour", tour)
    print("obj_func:", sim.obj_func(tour))
    sim.vizualize(tour, path=save_path)


def execute_milp(N: int, seed: int, experiment_name: str):
    milp_logs_dir = OUTPUT_DIR / experiment_name
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    sim = SimulatorExp(N, seed=seed)
    model = milp.TSPExp(sim.g)
    tour = model.solve(
        log_dir=milp_logs_dir,
        timelimit=60,
        mipgap=0.0,
        tee=True,
    )
    show_result(sim, tour, f"{milp_logs_dir}/tsp_milp_solution.png")
    is_valid, message = sim.is_valid_tour(tour, log=True)
    assert is_valid, message


def execute_llm(N: int, seed: int, iter_num: int, llm_model: str, experiment_name: str):
    sim = SimulatorExp(N, seed=seed)
    tour = llm.LLMSolverExp.solve(sim, iter_num=iter_num, llm_model=llm_model)
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_llm_solution_{experiment_name}.png")
    is_valid, message = sim.is_valid_tour(tour, log=True)
    print(is_valid, message)  # エラーの可能性があるため、asset文はなし


def test_milp():
    SimulatorExp.time_windows_num = 3
    SimulatorExp.precedence_pair_num = 3
    execute_milp(N=30, seed=3, experiment_name="milp_all_constraints")


def test_milp_time_windows():
    SimulatorExp.time_windows_num = 3
    SimulatorExp.precedence_pair_num = 0
    execute_milp(N=30, seed=0, experiment_name="milp_time_windows")


def test_milp_precedence_pair():
    SimulatorExp.time_windows_num = 0
    SimulatorExp.precedence_pair_num = 3
    execute_milp(N=30, seed=0, experiment_name="milp_precedence_pair")


def test_gpt4o():
    SimulatorExp.time_windows_num = 3
    SimulatorExp.precedence_pair_num = 3
    execute_llm(
        N=30,
        seed=3,
        iter_num=0,
        llm_model="gpt-4o",
        experiment_name="gpt4o_all_constraints",
    )


def test_gpt4o_time_windows():
    SimulatorExp.time_windows_num = 3
    SimulatorExp.precedence_pair_num = 0
    execute_llm(
        N=30,
        seed=0,
        iter_num=0,
        llm_model="gpt-4o",
        experiment_name="gpt4o_time_windows",
    )


def test_gpt4o_precedence_pair():
    SimulatorExp.time_windows_num = 0
    SimulatorExp.precedence_pair_num = 3
    execute_llm(
        N=30,
        seed=0,
        iter_num=0,
        llm_model="gpt-4o",
        experiment_name="gpt4o_precedence_pair",
    )


def test_o1():
    SimulatorExp.time_windows_num = 3
    SimulatorExp.precedence_pair_num = 3
    execute_llm(
        N=30,
        seed=3,
        iter_num=0,
        llm_model="o1",
        experiment_name="o1_all_constraints",
    )


def test_o1_time_windows():
    SimulatorExp.time_windows_num = 3
    SimulatorExp.precedence_pair_num = 0
    execute_llm(
        N=30,
        seed=0,
        iter_num=0,
        llm_model="o1",
        experiment_name="o1_time_windows",
    )


def test_o1_precedence_pair():
    SimulatorExp.time_windows_num = 0
    SimulatorExp.precedence_pair_num = 3
    execute_llm(
        N=30,
        seed=0,
        iter_num=0,
        llm_model="o1",
        experiment_name="o1_precedence_pair",
    )
