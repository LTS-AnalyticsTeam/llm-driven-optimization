from tsp.solver.milp import TSP
import pyomo.environ as pyo


class TSPExp(TSP):
    """TSP with time windows and precedence constraints."""

    def _define_params(self):
        """
        エッジの重み (distance) とサブツアー制約用に使用するパラメータを定義する。
        ここでは Big-M を、(N-1)*最大距離 に設定する。
        """

        # エッジの重み (距離)
        def _weight_init(m: TSP, i, j):
            return self.g[i][j]["weight"]

        self.weight = pyo.Param(self.edges, initialize=_weight_init)
        # すべてのエッジの weight を走査し、最大距離を取得
        max_dist = max(pyo.value(self.weight[e]) for e in self.edges)
        # 時間関連や到着時刻制約用のビッグM
        self.big_M = pyo.Param(initialize=max_dist * (self.N - 1))

    def _define_decision_variables(self):
        """
        x[i,j], f[i,j], t[i] を定義する。
        """
        # x[i,j]: i→j の辺を使う(1)かどうか
        self.x = pyo.Var(self.edges, domain=pyo.Binary)

        # f[i,j]: ノード間フロー量 (サブツアー防止用 1コモディティフロー)
        self.f = pyo.Var(self.edges, domain=pyo.NonNegativeReals)

        # t[i]: ノード i に到着する「時刻」
        # ここでは 0 以上で連続値とし、startノードは t=0 に固定
        self.t = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)
        self.t[self.start].fix(0)

    def _define_constraints(self):
        """
        高速化のために、フロー形式で巡回セールスマン問題を定義。
        さらに、時間窓と前後関係制約を追加する。

        フロー形式の制約:
        - out-degree=1 制約
        - in-degree=1 制約
        - フロー保存則 (startノードが N-1 単位送り出し、他ノードは 1 単位受け取る)
        - フロー容量制約 (f[i,j] <= (N-1) * x[i,j])
        """

        # 1) out-degree = 1
        def _out_degree_rule(m, i):
            return sum(m.x[i, j] for j in m.nodes if j != i) == 1

        self.out_degree_con_new = pyo.Constraint(self.nodes, rule=_out_degree_rule)

        # 2) in-degree = 1
        def _in_degree_rule(m, j):
            return sum(m.x[i, j] for i in m.nodes if i != j) == 1

        self.in_degree_con_new = pyo.Constraint(self.nodes, rule=_in_degree_rule)

        # 3) フロー保存則
        #   - startノードからは (N-1)単位のフローを送り出し、他のノードでは 1 単位を受け取る
        #   - sum_{j!=i} f[i,j] - sum_{k!=i} f[k,i] = ? の形
        def _flow_balance_rule(m, i):
            # 出発ノードの場合: 流出 - 流入 = (N - 1)
            if i == m.start:
                return (
                    sum(m.f[m.start, j] for j in m.nodes if j != m.start)
                    - sum(m.f[k, m.start] for k in m.nodes if k != m.start)
                    == m.N - 1
                )
            # それ以外の場合: 流入 - 流出 = 1
            else:
                return (
                    sum(m.f[k, i] for k in m.nodes if k != i)
                    - sum(m.f[i, j] for j in m.nodes if j != i)
                    == 1
                )

        self.flow_balance_con = pyo.Constraint(self.nodes, rule=_flow_balance_rule)

        # 4) フロー容量制約
        #    f[i,j] <= (N-1) * x[i,j]
        def _flow_capacity_rule(m, i, j):
            if i == j:
                return pyo.Constraint.Skip
            else:
                return m.f[i, j] <= (m.N - 1) * m.x[i, j]

        self.flow_capacity_con = pyo.Constraint(self.edges, rule=_flow_capacity_rule)

        # 5) 到着時刻の定義 (travel time + big_M)
        #    t[j] >= t[i] + weight[i,j] - big_M * (1 - x[i,j])
        def _arrival_time_rule(m, i, j):
            if i == j or j == m.start:
                return pyo.Constraint.Skip
            return m.t[j] >= m.t[i] + m.weight[i, j] - m.big_M * (1 - m.x[i, j])

        self.arrival_time_con = pyo.Constraint(self.edges, rule=_arrival_time_rule)

        # 6) 時間窓制約
        self.time_window_con = pyo.ConstraintList()
        for i in self.nodes:
            tw = self.g.nodes[i].get("time_window", None)
            if tw is not None:
                start_tw, end_tw = tw
                self.time_window_con.add(self.t[i] >= start_tw)
                self.time_window_con.add(self.t[i] <= end_tw)

        # 7) 作業順序（precedence）制約
        self.precedence_con = pyo.ConstraintList()
        # precedence_pairs はグラフの "precedence_pairs" 属性に格納されている前提
        for pair in self.g.graph.get("precedence_pairs", []):
            before, after = pair["before"], pair["after"]
            self.precedence_con.add(self.t[before] <= self.t[after])
