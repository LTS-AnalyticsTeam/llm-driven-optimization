from tsp.solver.milp import TSP
import pyomo.environ as pyo


class TSPImage(TSP):
    """TSP with time windows and precedence constraints."""

    def _define_decision_variables(self):
        super()._define_decision_variables()
        # 移動が禁止されたエッジの固定
        for i, j in self.edges:
            if not self.g.edges[i, j]["is_allowed"]:
                self.x[i, j].fix(0)
