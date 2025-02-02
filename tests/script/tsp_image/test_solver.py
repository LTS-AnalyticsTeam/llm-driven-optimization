import pytest
import sys
from pathlib import Path
from tsp_image.simulator import SimulatorImage
from tsp_image.solver import milp, llm
from unittest.mock import patch
from io import StringIO

OUTPUT_DIR = Path(__file__).parent / "__output__" / "solver"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
N = 30
SEED = 1


def show_result(sim: SimulatorImage, tour, save_path):
    print("tour", tour)
    print("obj_func:", sim.obj_func(tour))
    sim.vizualize(tour, path=save_path)


def execute_milp(N: int, seed: int, experiment_name: str):
    milp_logs_dir = OUTPUT_DIR / experiment_name
    milp_logs_dir.mkdir(exist_ok=True, parents=True)
    sim = SimulatorImage(N, seed=seed)
    model = milp.TSPImage(sim.g)
    tour = model.solve(
        log_dir=milp_logs_dir,
        timelimit=600,
        mipgap=0.0,
        tee=True,
    )
    show_result(sim, tour, f"{milp_logs_dir}/tsp_milp_solution.png")
    is_valid, message = sim.is_valid_tour(tour, log=True)
    assert is_valid, message


def execute_llm(N: int, seed: int, iter_num: int, llm_model: str, experiment_name: str):
    sim = SimulatorImage(N, seed=seed)
    tour = llm.LLMSolverImage.solve(sim, iter_num=iter_num, llm_model=llm_model)
    show_result(sim, tour, f"{OUTPUT_DIR}/tsp_llm_solution_{experiment_name}.png")
    is_valid, message = sim.is_valid_tour(tour, log=True)
    print(is_valid, message)  # エラーの可能性があるため、asset文はなし


@patch("builtins.input", side_effect=["0.4", "0.525", "0.45", "0.1", "0.05", "0.3"])
@patch("sys.stdout", new_callable=StringIO)
def test_milp(mock_stdout, mock_input):
    SimulatorImage.area_partition = True
    SimulatorImage.restricted_area = True
    execute_milp(N=N, seed=SEED, experiment_name="milp_all_constraints")
    sys.__stdout__.write(mock_stdout.getvalue())


@patch("builtins.input", side_effect=["0.4", "0.525"])
@patch("sys.stdout", new_callable=StringIO)
def test_milp_area_partition(mock_stdout, mock_input):
    SimulatorImage.area_partition = True
    SimulatorImage.restricted_area = False
    execute_milp(N=N, seed=SEED, experiment_name="milp_area_partition")
    sys.__stdout__.write(mock_stdout.getvalue())


@patch("builtins.input", side_effect=["0.45", "0.1", "0.05", "0.3"])
@patch("sys.stdout", new_callable=StringIO)
def test_milp_restricted_area(mock_stdout, mock_input):
    SimulatorImage.area_partition = False
    SimulatorImage.restricted_area = True
    execute_milp(N=N, seed=SEED, experiment_name="milp_restricted_area")
    sys.__stdout__.write(mock_stdout.getvalue())


@patch("builtins.input", side_effect=["0.4", "0.525", "0.45", "0.1", "0.05", "0.3"])
@patch("sys.stdout", new_callable=StringIO)
def test_gpt4o(mock_stdout, mock_input):
    SimulatorImage.area_partition = True
    SimulatorImage.restricted_area = True
    execute_llm(
        N=N,
        seed=SEED,
        iter_num=0,
        llm_model="gpt-4o",
        experiment_name="gpt4o_all_constraints",
    )
    sys.__stdout__.write(mock_stdout.getvalue())


@patch("builtins.input", side_effect=["0.4", "0.525"])
@patch("sys.stdout", new_callable=StringIO)
def test_gpt4o_area_partition(mock_stdout, mock_input):
    SimulatorImage.area_partition = True
    SimulatorImage.restricted_area = False
    execute_llm(
        N=N,
        seed=SEED,
        iter_num=0,
        llm_model="gpt-4o",
        experiment_name="gpt4o_area_partition",
    )
    sys.__stdout__.write(mock_stdout.getvalue())


@patch("builtins.input", side_effect=["0.45", "0.1", "0.05", "0.3"])
@patch("sys.stdout", new_callable=StringIO)
def test_gpt4o_restricted_area(mock_stdout, mock_input):
    SimulatorImage.area_partition = False
    SimulatorImage.restricted_area = True
    execute_llm(
        N=N,
        seed=SEED,
        iter_num=0,
        llm_model="gpt-4o",
        experiment_name="gpt4o_restricted_area",
    )
    sys.__stdout__.write(mock_stdout.getvalue())


@patch("builtins.input", side_effect=["0.4", "0.525", "0.45", "0.1", "0.05", "0.3"])
@patch("sys.stdout", new_callable=StringIO)
def test_o1(mock_stdout, mock_input):
    SimulatorImage.area_partition = True
    SimulatorImage.restricted_area = True
    execute_llm(
        N=N,
        seed=SEED,
        iter_num=0,
        llm_model="o1",
        experiment_name="o1_all_constraints",
    )
    sys.__stdout__.write(mock_stdout.getvalue())


@patch("builtins.input", side_effect=["0.4", "0.525"])
@patch("sys.stdout", new_callable=StringIO)
def test_o1_area_partition(mock_stdout, mock_input):
    SimulatorImage.area_partition = True
    SimulatorImage.restricted_area = False
    execute_llm(
        N=N,
        seed=SEED,
        iter_num=0,
        llm_model="o1",
        experiment_name="o1_area_partition",
    )
    sys.__stdout__.write(mock_stdout.getvalue())


@patch("builtins.input", side_effect=["0.45", "0.1", "0.05", "0.3"])
@patch("sys.stdout", new_callable=StringIO)
def test_o1_precedence_pair(mock_stdout, mock_input):
    SimulatorImage.area_partition = False
    SimulatorImage.restricted_area = True
    execute_llm(
        N=N,
        seed=SEED,
        iter_num=0,
        llm_model="o1",
        experiment_name="o1_restricted_area",
    )
    sys.__stdout__.write(mock_stdout.getvalue())
