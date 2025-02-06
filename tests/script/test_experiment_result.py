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


def experiment_tps_exp_gen_problem(
    time_windows_num: int, precedence_pair_num: int, problem_size=30
):
    save_dir = (
        OUTPUT_DIR
        / f"experiment_tps_exp-time_windows_num={time_windows_num}-precedence_pair_num={precedence_pair_num}"
    )
    exp = ExperimentTSPExp(
        save_dir=save_dir,
        time_windows_num=time_windows_num,
        precedence_pair_num=precedence_pair_num,
    )
    exp.gen_problem([problem_size], SIM_NUM)


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


def test_experiment_tps_exp_gen_problem_F_T():
    experiment_tps_exp_gen_problem(time_windows_num=0, precedence_pair_num=3)


def test_experiment_tps_exp_run_F_T():
    experiment_tps_exp_run(time_windows_num=0, precedence_pair_num=3)


def test_experiment_tps_exp_gen_problem_T_F():
    experiment_tps_exp_gen_problem(time_windows_num=3, precedence_pair_num=0)


def test_experiment_tps_exp_run_T_F():
    experiment_tps_exp_run(time_windows_num=3, precedence_pair_num=0)


def test_experiment_tps_exp_gen_problem_T_T():
    experiment_tps_exp_gen_problem(time_windows_num=1, precedence_pair_num=1, problem_size=10)  # fmt: skip
    experiment_tps_exp_gen_problem(time_windows_num=2, precedence_pair_num=2, problem_size=20)  # fmt: skip
    experiment_tps_exp_gen_problem(time_windows_num=3, precedence_pair_num=3, problem_size=30)  # fmt: skip
    experiment_tps_exp_gen_problem(time_windows_num=4, precedence_pair_num=4, problem_size=40)  # fmt: skip


def test_experiment_tps_exp_run_T_T():
    experiment_tps_exp_run(time_windows_num=1, precedence_pair_num=1)
    experiment_tps_exp_run(time_windows_num=2, precedence_pair_num=2)
    experiment_tps_exp_run(time_windows_num=3, precedence_pair_num=3)
    experiment_tps_exp_run(time_windows_num=4, precedence_pair_num=4)
