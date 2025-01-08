from tsp.solver.llm import LLMSolver
from tsp.simulator import Simulator
import networkx as nx
from pathlib import Path
from tqdm import tqdm


class LLMSolverExp(LLMSolver):

    def __init__(self, sim: Simulator):
        super().__init__(sim)
        self.TMP_DIR = Path(__file__).parent / "__tmp__"
        self.TMP_DIR.mkdir(exist_ok=True, parents=True)

    def _get_system_prompt(self) -> list[dict]:
        messages = super()._get_system_prompt()
        start_node = self.sim.g.graph["start"]
        time_windows = {
            f"Node-{n}": (
                round(self.sim.g.nodes[n]["time_window"][0], self.ROUND_DIGITS),
                round(self.sim.g.nodes[n]["time_window"][1], self.ROUND_DIGITS),
            )
            for n in self.sim.g.nodes
            if self.sim.g.nodes[n]["time_window"] is not None
        }
        precedence_pairs = self.sim.g.graph["precedence_pairs"]
        messages[0]["content"].append(
            {
                "type": "text",
                "text": f"次の条件は問題における制約条件です。この条件を必ず考慮して出力してください。",
            }
        )
        messages[0]["content"].append(
            {"type": "text", "text": f"Start node: {start_node}"}
        )
        messages[0]["content"].append(
            {"type": "text", "text": f"Time windows: {time_windows}"}
        )
        messages[0]["content"].append(
            {"type": "text", "text": f"Precedence pairs: {precedence_pairs}"}
        )
        return messages
