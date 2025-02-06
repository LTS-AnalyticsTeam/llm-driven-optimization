import pytest
from io import StringIO
from unittest.mock import patch
from tqdm import tqdm
import joblib
from pathlib import Path

from tsp_image import SimulatorImage, milp

OUTPUT_DIR = Path(__file__).parent / "__output__" / "experiment_result_image"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PROBLEM_SETTINGS = {
    20: [
        {
            "seed": 1,
            "ap_x_center": 0.4,
            "ap_y_center": 0.3,
            "ra_x": 0.2,
            "ra_y": 0.3,
            "ra_w": 0.2,
            "ra_h": 0.05,
        },
        {
            "seed": 5,
            "ap_x_center": 0.4,
            "ap_y_center": 0.6,
            "ra_x": 0.5,
            "ra_y": 0.8,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
        {
            "seed": 9,
            "ap_x_center": 0.4,
            "ap_y_center": 0.55,
            "ra_x": 0.6,
            "ra_y": 0.0,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
        {
            "seed": 16,
            "ap_x_center": 0.5,
            "ap_y_center": 0.4,
            "ra_x": 0.4,
            "ra_y": 0.8,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
        {
            "seed": 19,
            "ap_x_center": 0.6,
            "ap_y_center": 0.6,
            "ra_x": 0.5,
            "ra_y": 0.1,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
    ],
    30: [
        {
            "seed": 14,
            "ap_x_center": 0.45,
            "ap_y_center": 0.55,
            "ra_x": 0.7,
            "ra_y": 0.35,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
        {
            "seed": 16,
            "ap_x_center": 0.5,
            "ap_y_center": 0.6,
            "ra_x": 0.4,
            "ra_y": 0.8,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
        {
            "seed": 19,
            "ap_x_center": 0.6,
            "ap_y_center": 0.6,
            "ra_x": 0.5,
            "ra_y": 0.8,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
        {
            "seed": 31,
            "ap_x_center": 0.5,
            "ap_y_center": 0.6,
            "ra_x": 0.0,
            "ra_y": 0.5,
            "ra_w": 0.2,
            "ra_h": 0.05,
        },
        {
            "seed": 34,
            "ap_x_center": 0.475,
            "ap_y_center": 0.5,
            "ra_x": 0.7,
            "ra_y": 0.0,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
    ],
    40: [
        {
            "seed": 1,
            "ap_x_center": 0.6,
            "ap_y_center": 0.62,
            "ra_x": 0.35,
            "ra_y": 0.8,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
        {
            "seed": 10,
            "ap_x_center": 0.48,
            "ap_y_center": 0.5,
            "ra_x": 0.3,
            "ra_y": 0.0,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
        {
            "seed": 11,
            "ap_x_center": 0.4,
            "ap_y_center": 0.54,
            "ra_x": 0.75,
            "ra_y": 0.8,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
        {
            "seed": 14,
            "ap_x_center": 0.42,
            "ap_y_center": 0.48,
            "ra_x": 0.35,
            "ra_y": 0.3,
            "ra_w": 0.05,
            "ra_h": 0.2,
        },
        {
            "seed": 18,
            "ap_x_center": 0.44,
            "ap_y_center": 0.4,
            "ra_x": 0.0,
            "ra_y": 0.35,
            "ra_w": 0.2,
            "ra_h": 0.05,
        },
    ],
}


def generate_simulator(save_name: str):
    save_path = OUTPUT_DIR.joinpath(save_name)
    sim_dict: dict[int, list[SimulatorImage]] = {}
    for problem_size, settings in tqdm(
        PROBLEM_SETTINGS.items(), desc="Loop Problem Size", leave=False
    ):
        vis_dir = save_path.joinpath("vizuaisation", f"problem={problem_size}")
        vis_dir.mkdir(exist_ok=True, parents=True)
        sim_dict[problem_size] = []
        for setting in tqdm(settings, desc="Loop Setting", leave=False):
            seed = setting["seed"]
            input_values = [
                str(setting["ap_x_center"]),
                str(setting["ap_y_center"]),
                str(setting["ra_x"]),
                str(setting["ra_y"]),
                str(setting["ra_w"]),
                str(setting["ra_h"]),
            ]
            with patch("builtins.input", side_effect=input_values):
                with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                    sim = SimulatorImage(problem_size, seed)
                    milp_model = milp.TSPImage(sim.g)
                    tour = milp_model.solve(timelimit=3600, tee=True)
                    is_valid, _ = sim.is_valid_tour(tour)
                    if is_valid:
                        sim.opt_tour = tour
                        sim_dict[problem_size].append(sim)
                        base_name = f"problem={problem_size}_seed={seed}"
                        sim.vizualize(tour, vis_dir.joinpath(f"{base_name}_tour.png"))

                    print(mock_stdout.getvalue())

    joblib.dump(sim_dict, save_path / "problem.joblib")


def test_simulator_without_constraints():
    SimulatorImage.area_partition = False
    SimulatorImage.restricted_area = False
    generate_simulator("01_simulator_without_constraints")


def test_simulator_with_area_partition_constraints():
    SimulatorImage.area_partition = True
    SimulatorImage.restricted_area = False
    generate_simulator("02_simulator_with_area_partition_constraints")


def test_simulator_with_all_constraints():
    SimulatorImage.area_partition = True
    SimulatorImage.restricted_area = True
    generate_simulator("03_simulator_with_all_constraints")
