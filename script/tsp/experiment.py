from .solver import nn, fi, milp, llm
from tsp.simulator import define_problem, obj_func, is_valid_tour
import json
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as sw


Result = dict[int, dict[str, dict[str, any]]]
Summary = dict[str, dict[str, any]]


def run_all_solver(problem_size: int, sample_num: int, save_dir: Path) -> Result:
    """
    TSP の各種ソルバを実行して結果を保存する関数。
    """
    result = {}
    for i in tqdm(range(sample_num), desc="Loop Sampling", leave=False):
        result[i] = {}
        g = define_problem(problem_size, seed=i)
        milp_model = milp.TSP(g)
        tours = {
            "nn": nn.solve(g),
            "fi": fi.solve(g),
            "milp": milp_model.solve(),
            "gpt-4o": llm.solve(g, iter_num=1, llm_model="gpt-4o"),
            "o1": llm.solve(g, iter_num=1, llm_model="o1"),
        }
        for k, tour in tours.items():
            result[i][k] = {
                "tour": tour,
                "obj_value": obj_func(g, tour),
                "is_valid": is_valid_tour(g, tour)[0],
            }
    with open(save_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def summarize(result: Result) -> Summary:
    """
    各ソルバの結果を集計し、以下の情報を返す関数。
      - gap_mean
      - gap_95%_CI
      - validation_success_rate
      - gap_zero_rate
    """
    # 対象とするソルバ一覧
    solvers = ["nn", "fi", "milp", "gpt-4o", "o1"]

    # 各ソルバごとに目的関数値とGAPを溜めるリストを用意
    obj_values = {solver: [] for solver in solvers}
    gap_percentages = {solver: [] for solver in solvers if solver != "milp"}
    # 各ソルバごとのvalidation成功数と試行数
    valid_count_by_solver = {solver: 0 for solver in solvers}
    total_by_solver = {solver: 0 for solver in solvers}
    # 各ソルバごとにgap==0の数を記録
    gap_zero_count = {solver: 0 for solver in solvers if solver != "milp"}

    # 各インスタンス(i)についてデータを集計
    for data in result.values():
        milp_obj = data["milp"]["obj_value"]
        for solver, solver_data in data.items():
            obj = solver_data["obj_value"]
            total_by_solver[solver] += 1
            if solver_data["is_valid"]:
                valid_count_by_solver[solver] += 1

            # Skip gap calculations if obj is NaN
            if np.isnan(obj):
                pass
            else:
                obj_values[solver].append(obj)
                if solver != "milp":
                    gap = ((obj - milp_obj) / milp_obj) * 100
                    gap_percentages[solver].append(gap)
                    if gap == 0:
                        gap_zero_count[solver] += 1

    # 結果をまとめるための辞書
    summary = {}

    for solver in solvers:
        # validation成功率
        if total_by_solver[solver] == 0:
            val_success_rate = 0.0
        else:
            val_success_rate = (
                float(valid_count_by_solver[solver])
                / float(total_by_solver[solver])
                * 100.0
            )

        # milpと比較したGAP(%)の統計量を計算
        if solver == "milp":
            # milp自身のGAPは 0.0 とする
            gap_mean = 0.0
            gap_ci_lower = 0.0
            gap_ci_upper = 0.0
            gap_zero_rate = 100.0
        else:
            arr_gap = np.array(gap_percentages[solver])
            descr_gap = sw.DescrStatsW(arr_gap)
            gap_mean = float(descr_gap.mean)
            gap_ci_lower, gap_ci_upper = descr_gap.tconfint_mean(alpha=0.05)
            gap_ci_lower = float(gap_ci_lower)
            gap_ci_upper = float(gap_ci_upper)
            # gap==0の割合計算
            zero_count = gap_zero_count[solver]
            gap_zero_rate = (
                (zero_count / len(arr_gap)) * 100.0 if len(arr_gap) > 0 else 0.0
            )

        summary[solver] = {
            "gap_mean": gap_mean,
            "gap_95%_CI": [gap_ci_lower, gap_ci_upper],
            "validation_success_rate": val_success_rate,
            "gap_zero_rate": gap_zero_rate,
        }

    return summary


def convert_to_pandas(summary_dict: dict[int, Summary]) -> pd.DataFrame:
    """
    summarize() で返された結果を集約した辞書（複数の problem_size 分）を
    pandas.DataFrame に変換する。

    data は以下のような構造を想定：
    {
      10: {
        "nn": {
          "gap_mean": -7.54,
          "gap_95%_CI": [-19.26, 4.18],
          "validation_success_rate": 100.0,
          "gap_zero_rate": 40.0
        },
        "fi": {...},
        "milp": {...},
        "gpt-4o": {...},
        "o1": {...}
      },
      20: {
        "nn": {...},
        "fi": {...},
        ...
      }
    }

    返却する DataFrame では:
      - 行 (index) が problem_size
      - 列 (columns) が (solver, metric) の MultiIndex

    metric は下記のとおり:
      1. gap_mean
      2. gap_95%_CI_lower
      3. gap_95%_CI_upper
      4. validation_success_rate
    """
    rows = []
    indexes = []

    for problem_size, solver_dict in summary_dict.items():
        row = {}
        for solver, stats in solver_dict.items():
            row[(solver, "gap_mean")] = stats["gap_mean"]
            row[(solver, "gap_95%_CI_lower")] = stats["gap_95%_CI"][0]
            row[(solver, "gap_95%_CI_upper")] = stats["gap_95%_CI"][1]
            row[(solver, "validation_success_rate")] = stats["validation_success_rate"]
            row[(solver, "gap_zero_rate")] = stats["gap_zero_rate"]

        rows.append(row)
        indexes.append(problem_size)

    df = pd.DataFrame(rows, index=indexes)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["solver", "metric"])
    df = df.sort_index()

    return df


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
        summary = summarize(result)
        summaries[problem_size] = summary

    summaries_df = convert_to_pandas(summaries)
    summaries_df.to_csv(save_dir / "summary.csv", index=True)

    return None
