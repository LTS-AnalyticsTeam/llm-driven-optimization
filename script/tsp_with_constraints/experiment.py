from .solver import milp, llm
from .simulator import SimulatorExp
from tsp.experiment import Result, Summary, summarize, convert_to_pandas

import json
from tqdm import tqdm
from pathlib import Path


def run_all_solver(
    problem_size: int,
    sample_num: int,
    save_dir: Path,
) -> Result:
    """
    TSP の各種ソルバを実行して結果を保存する関数。
    """
    result = {}
    for i in tqdm(range(sample_num), desc="Loop Sampling", leave=False):
        result[i] = {}
        sim = SimulatorExp(
            problem_size,
            seed=i,
        )
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
    with open(save_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main(problem_sizes: list[int], sample_num: int, save_dir: Path):
    """
    複数の問題サイズについて run → summarize → pandas.DataFrame化 を行い、
    それらの結果をまとめて CSV に出力するメイン関数。
    """
    summaries = {}
    for problem_size in tqdm(problem_sizes, desc="Loop Problem Size"):
        save_dir_this_loop = save_dir / f"problem_size_{problem_size}"
        save_dir_this_loop.mkdir(exist_ok=True, parents=True)
        result = run_all_solver(problem_size, sample_num, save_dir_this_loop)
        summary = summarize(result, ["milp", "gpt-4o", "o1"])
        summaries[problem_size] = summary

    summaries_df = convert_to_pandas(summaries)
    summaries_df.to_csv(save_dir / "summary.csv", index=True)

    return None
