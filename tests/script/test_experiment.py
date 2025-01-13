from experiment import ExperimentTSP, ExperimentTSPExp
from pathlib import Path
import joblib


OUTPUT_DIR = Path(__file__).parent / "__output__" / "experiment"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SIM_NUM = 5


def test_experiment_tsp_gen_problem():
    exp = ExperimentTSP(OUTPUT_DIR / "experiment_tsp")
    exp.gen_problem([9, 10, 11], SIM_NUM)
    sim_dict = joblib.load(exp.problem_file_path)
    for sim_list in sim_dict.values():
        sim0, sim1 = sim_list[0], sim_list[1]
        assert len(sim_list) == SIM_NUM
        assert sim0.g.nodes[0]["x"] != sim1.g.nodes[0]["x"]


def test_experiment_tsp_run():
    exp = ExperimentTSP(OUTPUT_DIR / "experiment_tsp")
    exp.run(sample_num=30)


def test_experiment_tsp_exp_gen_problem():
    exp = ExperimentTSPExp(
        save_dir=OUTPUT_DIR / "experiment_tsp_exp",
        precedence_pair_num=3,
        time_windows_num=3,
    )
    exp.gen_problem([9, 10, 11], SIM_NUM)
    sim_dict = joblib.load(exp.problem_file_path)
    for sim_list in sim_dict.values():
        sim0, sim1 = sim_list[0], sim_list[1]
        assert len(sim_list) == SIM_NUM
        assert sim0.g.nodes[0]["x"] != sim1.g.nodes[0]["x"]


def test_experiment_tsp_exp_run():
    exp = ExperimentTSPExp(
        save_dir=OUTPUT_DIR / "experiment_tsp_exp",
        precedence_pair_num=3,
        time_windows_num=3,
    )
    exp.run()
