import networkx as nx
import pyomo.environ as pyo
from pathlib import Path
import cplex


class TSP(pyo.ConcreteModel):
    """
    Pyomo の ConcreteModel を継承した TSP モデルクラス。

    Attributes:
        g (nx.Graph): エッジに距離(weight)が設定された完全グラフ
        start (int): ツアーを開始する頂点
        N (int): グラフの頂点数
        nodes (pyo.Set): 範囲 [0, 1, 2, ... N-1] を表す Pyomo の集合
        edges (pyo.Set): (i, j) (i ≠ j) を全て含む Pyomo の集合
        weight (pyo.Param): edges 上で定義されるエッジの距離
        M (pyo.Param): サブツアー制約で使う十分に大きな定数 (本例ではNを用いる)
        x (pyo.Var): i→j 移動を表すバイナリ変数 (x[i,j] = 1 のときi→jが採択)
        u (pyo.Var): サブツアー除去用の巡回順位 (MTZの定式化で使用)
        obj (pyo.Objective): 総移動距離を最小化する目的関数
        out_degree_con (pyo.Constraint): 各頂点の out-degree=1 制約
        in_degree_con (pyo.Constraint): 各頂点の in-degree=1 制約
        subtour_con (pyo.Constraint): サブツアーを除去するための MTZ 制約
    """

    def __init__(self, g: nx.Graph, *args, **kwargs):
        """
        コンストラクタ。必要な属性をセットし、
        決定変数・目的関数・制約条件を定義する。

        Args:
            g (nx.Graph): エッジに距離(weight)が設定された完全グラフ
            start (int): ツアーを開始する頂点 (デフォルト 0)
        """
        super().__init__(*args, **kwargs)

        # グラフやノード情報などを保持
        self.g = g
        self.start = 0
        self.N = len(g.nodes)

        # 定義メソッドを順に呼び出す
        self._define_sets()
        self._define_params()
        self._define_decision_variables()
        self._define_objective_function()
        self._define_constraints()

    def _define_sets(self):
        """ノード集合および有向辺集合を Pyomo の Set として定義する。"""
        self.nodes = pyo.Set(initialize=range(self.N))

        self.edges = pyo.Set(
            initialize=[(i, j) for i in range(self.N) for j in range(self.N) if i != j]
        )

    def _define_params(self):
        """エッジの重み (distance) とサブツアー制約用 M を Pyomo の Param として定義する。"""

        # エッジの重み (距離)
        def _weight_init(m: TSP, i, j):
            return self.g[i][j]["weight"]

        self.weight = pyo.Param(self.edges, initialize=_weight_init)

        # サブツアー除去制約に用いる十分に大きな定数 M
        self.M = pyo.Param(initialize=self.N)

    def _define_decision_variables(self):
        """巡回順序変数 (u) とエッジ選択変数 (x) を Pyomo の Var として定義する。"""
        # x[i, j]: i→j の辺を使う (1) or 使わない (0)
        self.x = pyo.Var(self.edges, domain=pyo.Binary)

        # u[i]: MTZ法で用いる巡回順序 (0 <= u[i] <= N-1)
        self.u = pyo.Var(
            self.nodes, domain=pyo.NonNegativeIntegers, bounds=(0, self.N - 1)
        )

        # スタートノードの巡回順位は 0 に固定
        self.u[self.start].fix(0)

    def _define_objective_function(self):
        """総移動距離を最小化する目的関数を定義する。"""

        def _obj_rule(m: TSP):
            return sum(m.weight[i, j] * m.x[i, j] for (i, j) in m.edges)

        self.obj = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    def _define_constraints(self):
        """out-degree=1, in-degree=1, サブツアー除去 (MTZ) の制約を定義する。"""

        # 各頂点からは必ず1本の辺が出る (out-degree=1)
        def _out_degree_rule(m: TSP, i):
            return sum(m.x[i, j] for j in m.nodes if i != j) == 1

        # 各頂点には必ず1本の辺が入る (in-degree=1)
        def _in_degree_rule(m: TSP, j):
            return sum(m.x[i, j] for i in m.nodes if i != j) == 1

        # サブツアー除去制約 (Miller–Tucker–Zemlin, MTZ)
        # u[j] >= u[i] + 1 - M * (1 - x[i, j])
        # (i != j) かつ (i != m.start) かつ (j != m.start) の場合のみ適用
        def _subtour_rule(m: TSP, i, j):
            if i != j and i != m.start and j != m.start:
                return m.u[j] >= m.u[i] + 1 - m.M * (1 - m.x[i, j])
            else:
                return pyo.Constraint.Skip

        self.out_degree_con = pyo.Constraint(self.nodes, rule=_out_degree_rule)
        self.in_degree_con = pyo.Constraint(self.nodes, rule=_in_degree_rule)
        self.subtour_con = pyo.Constraint(self.nodes, self.nodes, rule=_subtour_rule)

    def solve(
        self,
        log_dir: Path = Path("/tmp/cplex_log"),
        timelimit=100,
        mipgap=0,
        tee=False,
        guarantee_optimal=True,
    ) -> list[int]:
        """
        ソルバーを呼び出して TSP を解き、訪問順を返す。

        Args:
            logfile (Path): ログファイルの出力先
            timelimit (int): ソルバーの制限時間 (秒)
            mipgap (float): MIP Gap
            tee (bool): True のとき解過程を標準出力に表示

        Returns:
            list[int]: TSP の訪問順序 (例: [0, 2, 5, 1, 4, 3])
        """
        log_dir.mkdir(exist_ok=True, parents=True)
        # ソルバーの指定
        solver = pyo.SolverFactory("cplex")
        solver.options["logfile"] = str(log_dir / "tsp_milp.log")
        solver.options["timelimit"] = timelimit
        solver.options["mip_tolerances_mipgap"] = mipgap
        solver.options["mip_tolerances_absmipgap"] = mipgap

        result = solver.solve(self, tee=tee)
        self.display(filename=log_dir / "tsp_display.log")  # ログの保存

        if guarantee_optimal:
            # 最適解が得られたかどうかを確認
            if result.solver.termination_condition != pyo.TerminationCondition.optimal:
                raise Exception(
                    f"The solution is not optimal. Solver termination_condition: {result.solver.termination_condition}"
                )
        # 解の復元
        next_node = {}
        for i, j in self.edges:
            if pyo.value(self.x[i, j]) > 0.5:
                next_node[i] = j

        # ツアーを復元する
        tour = [self.start]
        current = self.start
        while True:
            current = next_node[current]
            if current == self.start:
                break
            tour.append(current)

        return tour

    def analyze_conflict(self, log_dir: Path):
        # 求解
        solver = pyo.SolverFactory("cplex_persistent")
        solver.set_instance(self)
        solver.solve(self, tee=True)
        cplex_instance: cplex.Cplex = solver._solver_model
        cplex_instance.write(str(log_dir / "tsp_model.lp"))
        # IISの解析
        cplex_instance: cplex.Cplex = solver._solver_model
        cplex_instance.conflict.refine()
        cplex_instance.conflict.write(str(log_dir / "tsp_conflict_iis.ilp"))
        # エラー報告
        status = cplex_instance.solution.get_status()
        print(
            f"Model is infeasible. Solution status {status}: {cplex_instance.solution.status[status]}"
        )
        return None
