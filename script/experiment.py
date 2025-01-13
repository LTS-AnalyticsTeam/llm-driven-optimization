import tsp
import tsp_exp
import json
import random
import uuid
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import statsmodels.stats.weightstats as sw
from abc import ABC, abstractmethod

SimDict = dict[int, list[tsp.Simulator]]
Result = dict[int, dict[str, dict[str, any]]]
Summary = dict[str, dict[str, any]]
"""
Result は以下のような構造を想定している。
```json
{
    "0": {
        "nn": {
            "tour": [0,11,19,5,18,3,7,2,1,12,17,4,13,10,6,16,15,9,14,8],
            "obj_value": 3.3179157402627433,
            "is_valid": true
        },
        "fi": {
            "tour": [7,3,18,5,19,11,0,14,8,9,15,16,6,10,13,4,2,1,12,17],
            "obj_value": 2.9766692741699385,
            "is_valid": true
        },
        "milp": {
            "tour": [0,19,11,5,18,3,7,17,12,1,2,4,13,10,6,16,15,9,8,14],
            "obj_value": 2.9728338167860118,
            "is_valid": true
        },
        "gpt-4o": {
            "tour": [8,9,14,0,11,5,19,3,18,7,12,17,1,2,4,13,10,6,16,15],
            "obj_value": 3.5283091857677995,
            "is_valid": true
        },
        "o1": {
            "tour": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
            "obj_value": 9.43642130406245,
            "is_valid": true
        }
    },
    "1": {
        "nn": {...},
        "fi": {...},
        "milp": {...},
        "gpt-4o": {...},
        "o1":{...}
    },
    "2": {...},
    "3": {...},
    ...
}
```

Summary は以下のような構造を想定している。
```json
{
  "nn": {
    "gap_mean": 16.776245677421088,
    "gap_std": 8.9260136820131,
    "gap_se": 8.9260136820131,
    "gap_95%_CI": [0.8497677366138063, 32.70272361822837],
    "validation_success_rate": 100.0,
    "gap_zero_rate": 0.0
  },
  "fi": {
    "gap_mean": 3.2077061265450113,
    "gap_std": 8.9260136820131,
    "gap_se": 8.9260136820131,
    "gap_95%_CI": [-2.1113909721030697, 8.526803225193092],
    "validation_success_rate": 100.0,
    "gap_zero_rate": 20.0
  },
  "milp": {
    "gap_mean": 0.0,
    "gap_std": 0,
    "gap_se": 0,    
    "gap_95%_CI": [0.0, 0.0],
    "validation_success_rate": 100.0,
    "gap_zero_rate": 100.0
  },
  "gpt-4o": {
    "gap_mean": 23.901450929801783,
    "gap_std": 8.9260136820131,
    "gap_se": 8.9260136820131,    
    "gap_95%_CI": [4.804552412408558, 42.99834944719501],
    "validation_success_rate": 100.0,
    "gap_zero_rate": 0.0
  },
  "o1": {
    "gap_mean": 198.4170221655523,
    "gap_std": 8.9260136820131,
    "gap_se": 8.9260136820131,    
    "gap_95%_CI": [169.74708465947012, 227.0869596716345],
    "validation_success_rate": 100.0,
    "gap_zero_rate": 0.0
  }
}
```
"""


class Experiment(ABC):

    solvers: list[str]

    def __init__(self, save_dir: Path):
        save_dir.mkdir(exist_ok=True, parents=True)
        self.save_dir = save_dir
        self.problem_file_path = self.save_dir / "problem.joblib"
        self.summary_file_path = self.save_dir / "summary.csv"

    def _result_path(self, problem_size: int) -> Path:
        return self.save_dir / f"result_p-size-{problem_size}.json"

    @abstractmethod
    def _gen_sim(self, problem_size: int, seed: int) -> tsp.Simulator:
        pass

    @abstractmethod
    def _gen_milp_model(self, sim: tsp.Simulator) -> tsp.milp.TSP:
        pass

    @abstractmethod
    def _get_tours(self, sim: tsp.Simulator) -> dict[str, list[int]]:
        pass

    def gen_problem(self, problem_sizes: list[int], sim_num: int) -> None:
        """
        問題を生成する関数。
        """

        sim_dict: SimDict = {}
        for problem_size in tqdm(problem_sizes, desc="Loop Problem Size", leave=False):
            progress_bar = tqdm(total=sim_num, desc="Loop Sampling", leave=False)
            i = 0
            sim_dict[problem_size] = []
            while progress_bar.n < sim_num:
                sim = self._gen_sim(problem_size, seed=i)
                milp_model = self._gen_milp_model(sim)
                try:
                    tour = milp_model.solve(timelimit=10)
                    is_valid, _ = sim.is_valid_tour(tour)
                    if is_valid:
                        sim.opt_tour = tour
                        sim_dict[problem_size].append(sim)
                        progress_bar.update(1)
                except:
                    pass
                i += 1

        joblib.dump(sim_dict, self.problem_file_path)

        return None

    def run(self, sample_num: int = 30) -> None:
        """
        複数の問題サイズについて run → summarize → pandas.DataFrame化 を行い、
        それらの結果をまとめて CSV に出力するメイン関数。
        """
        sim_dict: SimDict = joblib.load(self.problem_file_path)

        summaries = {}
        for p_size, sim_list in tqdm(sim_dict.items(), desc="Loop Problem Size"):
            try:
                # すでに結果がある場合は読み込む
                with open(self._result_path(p_size), "r", encoding="utf-8") as f:
                    result = json.load(f)
            except:
                # 結果がない場合は初期化
                result = {}

            p_bar = tqdm(total=sample_num, desc="Loop Sampling", leave=False)
            p_bar.update(len(result.keys()))

            while p_bar.n < sample_num:
                for sim in random.sample(sim_list, len(sim_list)):
                    try:
                        key = str(uuid.uuid4())
                        result[key] = {}
                        tours = self._get_tours(sim)
                        for solver, tour in tours.items():
                            obj_value = sim.obj_func(tour)
                            is_valid, messeage = sim.is_valid_tour(tour)
                            result[key][solver] = {
                                "tour": tour,
                                "obj_value": obj_value,
                                "is_valid": is_valid,
                                "messeage": messeage,
                            }
                        with open(
                            self._result_path(p_size), "w", encoding="utf-8"
                        ) as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        p_bar.update(1)
                    except:
                        pass

            summary = self._summarize(result, self.solvers)
            summaries[p_size] = summary

        summaries_df = self._convert_to_pandas(summaries)
        summaries_df.to_csv(self.summary_file_path, index=True, encoding="utf-8-sig")

        return None

    def _summarize(self, result: Result, solvers: list[str]) -> Summary:
        """
        各ソルバの結果を集計し、以下の情報を返す関数。
        - gap_mean
        - gap_std
        - gap_se
        - gap_95%_CI
        - validation_success_rate
        - gap_zero_rate
        """
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
                gap_std = 0.0
                gap_se = 0.0
                gap_ci_lower = 0.0
                gap_ci_upper = 0.0
                gap_zero_rate = 100.0
            else:
                arr_gap = np.array(gap_percentages[solver])
                descr_gap = sw.DescrStatsW(arr_gap)
                gap_mean = float(descr_gap.mean)
                gap_ci_lower, gap_ci_upper = descr_gap.tconfint_mean(alpha=0.05)
                gap_std = float(np.std(arr_gap, ddof=1))  # sample std
                gap_se = gap_std / np.sqrt(len(arr_gap)) if len(arr_gap) > 0 else 0.0

                # gap==0の割合計算
                zero_count = gap_zero_count[solver]
                gap_zero_rate = (
                    (zero_count / len(arr_gap)) * 100.0 if len(arr_gap) > 0 else 0.0
                )

            summary[solver] = {
                "gap_mean": gap_mean,
                "gap_std": gap_std,
                "gap_se": gap_se,
                "gap_95%_CI": [gap_ci_lower, gap_ci_upper],
                "validation_success_rate": val_success_rate,
                "gap_zero_rate": gap_zero_rate,
            }

        return summary

    def _convert_to_pandas(self, summary_dict: dict[int, Summary]) -> pd.DataFrame:
        """
        summarize() で返された結果を集約した辞書（複数の problem_size 分）を
        pandas.DataFrame に変換する。

        summary_dict は以下のような構造を想定：
        {
        10: {
            "nn": {
                "gap_mean": -7.54,
                "gap_std" = 2.2,
                "gap_se" = 0.9,
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
        2. gap_std
        3. gap_se
        4. gap_95%_CI_lower
        5. gap_95%_CI_upper
        6. validation_success_rate
        """
        rows = []
        indexes = []

        for problem_size, solver_dict in summary_dict.items():
            row = {}
            for solver, stats in solver_dict.items():
                row[(solver, "gap_mean")] = stats["gap_mean"]
                row[(solver, "gap_std")] = stats["gap_std"]
                row[(solver, "gap_se")] = stats["gap_se"]
                row[(solver, "gap_95%_CI_lower")] = stats["gap_95%_CI"][0]
                row[(solver, "gap_95%_CI_upper")] = stats["gap_95%_CI"][1]
                row[(solver, "validation_success_rate")] = stats[
                    "validation_success_rate"
                ]
                row[(solver, "gap_zero_rate")] = stats["gap_zero_rate"]

            rows.append(row)
            indexes.append(problem_size)

        df = pd.DataFrame(rows, index=indexes)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["solver", "metric"])
        df = df.sort_index()

        return df


class ExperimentTSP(Experiment):

    solvers = ["nn", "fi", "milp", "gpt-4o", "o1"]

    def _gen_sim(self, problem_size: int, seed: int) -> tsp.Simulator:
        return tsp.Simulator(problem_size, seed)

    def _gen_milp_model(self, sim: tsp.Simulator) -> tsp.milp.TSP:
        return tsp.milp.TSP(sim.g)

    def _get_tours(self, sim: tsp.Simulator) -> dict[str, list[int]]:
        return {
            "nn": tsp.nn.solve(sim.g),
            "fi": tsp.fi.solve(sim.g),
            "milp": sim.opt_tour,
            "gpt-4o": tsp.llm.LLMSolver.solve(sim, iter_num=0, llm_model="gpt-4o"),
            "o1": tsp.llm.LLMSolver.solve(sim, iter_num=0, llm_model="o1"),
        }


class ExperimentTSPExp(Experiment):

    solvers = ["milp", "gpt-4o", "o1"]

    def __init__(
        self,
        save_dir: Path,
        time_windows_num=0,
        precedence_pair_num=0,
    ):
        super().__init__(save_dir)
        tsp_exp.SimulatorExp.time_windows_num = time_windows_num
        tsp_exp.SimulatorExp.precedence_pair_num = precedence_pair_num

    def _gen_sim(self, problem_size: int, seed: int) -> tsp_exp.SimulatorExp:
        return tsp_exp.SimulatorExp(problem_size, seed)

    def _gen_milp_model(self, sim: tsp_exp.SimulatorExp) -> tsp_exp.milp.TSPExp:
        return tsp_exp.milp.TSPExp(sim.g)

    def _get_tours(self, sim: tsp_exp.SimulatorExp) -> dict[str, list[int]]:
        return {
            "milp": sim.opt_tour,
            "gpt-4o": tsp_exp.llm.LLMSolverExp.solve(sim, iter_num=0, llm_model="gpt-4o"),
            "o1": tsp_exp.llm.LLMSolverExp.solve(sim, iter_num=0, llm_model="o1"),
        }  # fmt: skip
