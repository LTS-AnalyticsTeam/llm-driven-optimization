from experiment import Experiment1, Experiment2
from pathlib import Path
import joblib


OUTPUT_DIR = Path(__file__).parent / "__output__" / "experiment"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SAMPLE_NUM = 5


def test_exp1():
    exp = Experiment1(OUTPUT_DIR / "exp1")
    exp.gen_problem([9, 10, 11], SAMPLE_NUM)
    sim_dict = joblib.load(exp.problem_file_path)
    for sim_list in sim_dict.values():
        sim0, sim1 = sim_list[0], sim_list[1]
        assert len(sim_list) == SAMPLE_NUM
        assert sim0.g.nodes[0]["x"] != sim1.g.nodes[0]["x"]

    exp.run()


def test_exp2():

    exp = Experiment2(
        save_dir=OUTPUT_DIR / "exp2",
        precedence_constraints=True,
        time_windows_constraints=True,
    )
    exp.gen_problem([9, 10, 11], SAMPLE_NUM)
    sim_dict = joblib.load(exp.problem_file_path)
    for sim_list in sim_dict.values():
        sim0, sim1 = sim_list[0], sim_list[1]
        assert len(sim_list) == SAMPLE_NUM
        assert sim0.g.nodes[0]["x"] != sim1.g.nodes[0]["x"]

    exp.run()
