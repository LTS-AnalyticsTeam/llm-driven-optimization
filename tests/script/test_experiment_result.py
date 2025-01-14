from experiment import ExperimentTSP, ExperimentTSPExp
from pathlib import Path
import pytest

OUTPUT_DIR = Path(__file__).parent / "__output__" / "experiment_result"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SIM_NUM = 5
SAMPLE_NUM = 30


def test_experiment_tsp_gen_problem():
    exp = ExperimentTSP(OUTPUT_DIR / "experiment_tsp")
    exp.gen_problem([10, 30, 50], SIM_NUM)


def test_experiment_tsp_run():
    exp = ExperimentTSP(OUTPUT_DIR / "experiment_tsp")
    exp.run(sample_num=SAMPLE_NUM)


def experiment_tps_exp_gen_problem(time_windows_num: int, precedence_pair_num: int):
    save_dir = (
        OUTPUT_DIR
        / f"experiment_tps_exp-time_windows_num={time_windows_num}-precedence_pair_num={precedence_pair_num}"
    )
    exp = ExperimentTSPExp(
        save_dir=save_dir,
        time_windows_num=time_windows_num,
        precedence_pair_num=precedence_pair_num,
    )
    exp.gen_problem([30], SIM_NUM)


def experiment_tps_exp_run(time_windows_num: int, precedence_pair_num: int):
    save_dir = (
        OUTPUT_DIR
        / f"experiment_tps_exp-time_windows_num={time_windows_num}-precedence_pair_num={precedence_pair_num}"
    )
    exp = ExperimentTSPExp(
        save_dir=save_dir,
        time_windows_num=time_windows_num,
        precedence_pair_num=precedence_pair_num,
    )
    exp.run(sample_num=SAMPLE_NUM)


def test_experiment_tps_exp_gen_problem_0_3():
    experiment_tps_exp_gen_problem(time_windows_num=0, precedence_pair_num=3)


def test_experiment_tps_exp_run_0_3():
    experiment_tps_exp_run(time_windows_num=0, precedence_pair_num=3)


def test_experiment_tps_exp_gen_problem_3_0():
    experiment_tps_exp_gen_problem(time_windows_num=3, precedence_pair_num=0)


def test_experiment_tps_exp_run_3_0():
    experiment_tps_exp_run(time_windows_num=3, precedence_pair_num=0)


def test_experiment_tps_exp_gen_problem_3_3():
    experiment_tps_exp_gen_problem(time_windows_num=3, precedence_pair_num=3)


def test_experiment_tps_exp_run_3_3():
    experiment_tps_exp_run(time_windows_num=3, precedence_pair_num=3)


def test_experiment_tps_exp_run_all():
    experiment_tps_exp_run(time_windows_num=0, precedence_pair_num=3)
    experiment_tps_exp_run(time_windows_num=3, precedence_pair_num=0)
    experiment_tps_exp_run(time_windows_num=3, precedence_pair_num=3)
