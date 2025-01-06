from tsp_with_constraints.experiment import SimulatorExp
from tsp_with_constraints.experiment import (
    generate_problem,
    run_all_solver,
    main,
)
from pathlib import Path
import joblib
import json

OUTPUT_DIR = Path(__file__).parent / "__output__" / "experiment"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SAMPLE_NUM = 5


def test_generate_problem():
    SimulatorExp.time_windows_constraints = True
    SimulatorExp.precedence_constraints = True
    generate_problem([9, 10, 11], SAMPLE_NUM, OUTPUT_DIR / "sim_dict.joblib")
    sim_dict = joblib.load(OUTPUT_DIR / "sim_dict.joblib")
    for sim_list in sim_dict.values():
        sim0, sim1 = sim_list[0], sim_list[1]
        assert len(sim_list) == SAMPLE_NUM
        assert sim0.g.nodes[0]["x"] != sim1.g.nodes[0]["x"]


def test_run_all_solver():
    sim_dict = joblib.load(OUTPUT_DIR / "sim_dict.joblib")
    sim_list = [sim_list for sim_list in sim_dict.values()][0]
    run_all_solver(sim_list, OUTPUT_DIR)


def test_main():
    sim_dict = joblib.load(OUTPUT_DIR / "sim_dict.joblib")
    main(sim_dict, OUTPUT_DIR / "main")
