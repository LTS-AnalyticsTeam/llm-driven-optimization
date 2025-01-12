from tsp.solver.llm import LLMSolver


class LLMSolverExp(LLMSolver):

    def _get_system_prompt(self):
        node_coords = {
            n: (
                round(self.sim.g.nodes[n]["x"], self.ROUND_DIGITS),
                round(self.sim.g.nodes[n]["y"], self.ROUND_DIGITS),
            )
            for n in self.sim.g.nodes
        }
        distances = {
            (u, v): round(self.sim.g[u][v]["weight"], self.ROUND_DIGITS)
            for (u, v) in self.sim.g.edges
        }
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

        # TSP問題設定のシステムプロンプト
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.PROMPT.format(
                            problem="\n".join(
                                [
                                    f"## Node coordinates  \n{node_coords}",
                                    f"## Edges and distances  \n{distances}",
                                ]
                            ),
                            constraints="\n".join(
                                [
                                    f"## Constraint0  \nここで「start node」とは、必ずツアーの最初に訪問し、最終的に戻る拠点とするノードを指します。  \nStart node: {start_node}",
                                    f"## Constraint1  \n特定のノードには訪問が許可される時間帯（最早到着時刻と最遅到着時刻）が定義されています。以下は「Node-n: (最早到着時刻, 最遅到着時刻)」を表します。  \nTime windows: {time_windows}",
                                    f"## Constraint2  \n以下のノードペアについては、「前者を先に訪問しなければならない（後者を後に訪問しなければならない）」という順序制約があります。  \nPrecedence pairs: {precedence_pairs}",
                                ]
                            ),
                        ),
                    },
                ],
            },
        ]
        return messages
