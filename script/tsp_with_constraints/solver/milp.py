import tsp
import pyomo.environ as pyo


class TSP(tsp.solver.milp.TSP):
    """TSP with time windows and precedence constraints."""

    def _define_params(self):
        """エッジの重み (distance) とサブツアー制約用 M を Pyomo の Param として定義する。"""
        super()._define_params()
        self.big_M = pyo.Param(initialize=self.N * 100)

    def _define_decision_variables(self):
        super()._define_decision_variables()
        self.t = pyo.Var(self.nodes, domain=pyo.NonNegativeReals)
        self.t[self.start].fix(0)

    def _define_constraints(self):
        super()._define_constraints()

        def _arrival_time_rule(m: TSP, i, j):
            if j == m.start:
                return pyo.Constraint.Skip
            else:
                return m.t[j] >= m.t[i] + m.weight[i, j] - self.big_M * (1 - m.x[i, j])

        self.arrival_time_con = pyo.Constraint(self.edges, rule=_arrival_time_rule)

        self.time_window_con = pyo.ConstraintList()
        for i in self.nodes:
            tw = self.g.nodes[i]["time_window"]
            if tw is not None:
                start, end = tw
                self.time_window_con.add(self.t[i] >= start)
                self.time_window_con.add(self.t[i] <= end)

        self.precedence_con = pyo.ConstraintList()
        for pair in self.g.graph["precedence_pairs"]:
            before, after = pair["before"], pair["after"]
            self.precedence_con.add(self.t[before] <= self.t[after])
