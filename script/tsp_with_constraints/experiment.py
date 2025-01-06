from .solver import milp, llm
from .simulator import SimulatorExp
from tsp.experiment import Result, Summary, summarize, convert_to_pandas

import json
import joblib
from tqdm import tqdm
from pathlib import Path

SimDict = dict[int, list[SimulatorExp]]


def generate_problem(
    problem_sizes: list[int], sample_num: int, save_path: Path
) -> None:
    """
    問題を生成する関数。
    """

    sim_dict: SimDict = {}
    for problem_size in tqdm(problem_sizes, desc="Loop Problem Size", leave=False):
        progress_bar = tqdm(total=sample_num, desc="Loop Sampling", leave=False)
        i = 0
        sim_dict[problem_size] = []
        while progress_bar.n < sample_num:
            sim = SimulatorExp(problem_size, seed=i)
            milp_model = milp.TSPExp(sim.g)
            try:
                tour = milp_model.solve(timelimit=10)
                is_valid, _ = sim.is_valid_tour(tour)
                if is_valid:
                    sim_dict[problem_size].append(sim)
                    progress_bar.update(1)
            except:
                pass
            i += 1

    joblib.dump(sim_dict, save_path)

    return None


def run_all_solver(
    sim_list: list[SimulatorExp],
    save_dir: Path,
) -> Result:
    """
    TSP の各種ソルバを実行して結果を保存する関数。
    """
    result = {}
    i = 0
    for sim in tqdm(sim_list, desc="Loop Sampling", leave=False):
        result[i] = {}
        milp_model = milp.TSPExp(sim.g)
        tours = {
            "milp": milp_model.solve(),
            "gpt-4o": llm.LLMSolverExp.solve(sim, iter_num=1, llm_model="gpt-4o"),
            "o1": llm.LLMSolverExp.solve(sim, iter_num=1, llm_model="o1"),
        }
        for k, tour in tours.items():
            obj_value = sim.obj_func(tour)
            is_valid, messeage = sim.is_valid_tour(tour)
            result[i][k] = {
                "tour": tour,
                "obj_value": obj_value,
                "is_valid": is_valid,
                "messeage": messeage,
            }
        i += 1
    with open(save_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main(sim_dict: SimDict, save_dir: Path):
    """
    複数の問題サイズについて run → summarize → pandas.DataFrame化 を行い、
    それらの結果をまとめて CSV に出力するメイン関数。
    """
    summaries = {}
    for problem_size, sim_list in tqdm(sim_dict.items(), desc="Loop Problem Size"):
        save_dir_this_loop = save_dir / f"problem_size_{problem_size}"
        save_dir_this_loop.mkdir(exist_ok=True, parents=True)
        result = run_all_solver(sim_list, save_dir_this_loop)
        summary = summarize(result, ["milp", "gpt-4o", "o1"])
        summaries[problem_size] = summary

    summaries_df = convert_to_pandas(summaries)
    summaries_df.to_csv(save_dir / "summary.csv", index=True)

    return None
