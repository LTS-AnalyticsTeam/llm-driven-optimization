from experiment import ExperimentTSPImage
from pathlib import Path
import pytest

OUTPUT_DIR = Path(__file__).parent / "__output__" / "experiment_result"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SIM_NUM = 5
PROBLEM_SIZE = 10
SAMPLE_NUM = 30


def experiment_tps_exp_gen_problem(area_partition: bool, restricted_area: bool):
    save_dir = (
        OUTPUT_DIR
        / f"experiment_tps_exp-area_partition={area_partition}-restricted_area={restricted_area}"
    )
    exp = ExperimentTSPImage(
        save_dir=save_dir,
        area_partition=area_partition,
        restricted_area=restricted_area,
    )
    exp.gen_problem([PROBLEM_SIZE], SIM_NUM)


def experiment_tps_exp_run(area_partition: bool, restricted_area: bool):
    save_dir = (
        OUTPUT_DIR
        / f"experiment_tps_exp-area_partition={area_partition}-restricted_area={restricted_area}"
    )
    exp = ExperimentTSPImage(
        save_dir=save_dir,
        area_partition=area_partition,
        restricted_area=restricted_area,
    )
    exp.run(sample_num=SAMPLE_NUM)


if __name__ == "__main__":
    import tsp

    experiment_tps_exp_gen_problem(area_partition=True, restricted_area=True)
    # experiment_tps_exp_run(area_partition=True, restricted_area=True)
