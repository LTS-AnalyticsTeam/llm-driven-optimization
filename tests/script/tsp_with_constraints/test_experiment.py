from tsp_with_constraints.experiment import SimulatorExp
from tsp_with_constraints.experiment import (
    run_all_solver,
    main,
)
from pathlib import Path
import json

OUTPUT_DIR = Path(__file__).parent / "__output__" / "experiment"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def test_run_all_solver():
    run_all_solver(10, 5, OUTPUT_DIR)


def test_main():
    SimulatorExp.time_windows_constraints = True
    SimulatorExp.precedence_constraints = False
    main([9, 10, 11], 3, OUTPUT_DIR / "main")
