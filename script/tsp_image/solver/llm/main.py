from tsp.solver.llm import LLMSolver


class LLMSolverImage(LLMSolver):

    USE_IMAGE = True

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
                                    "## Constraint1  \n- この制約条件を考慮する上で画像を深く読み込んで各ノードがどのエリアA,B,C,Dに所属しているか深く考えてください。  \n- 画像で定義されているエリアA,B,C,D,Aの順序で巡回するツアーを作成してください。  \n- 順序を無視してエリアを逆走したり、各エリアのノードを回る前に先のエリアに進んだりすることは禁止されています。  \n- 次のエリアに近いノードから次のエリア移動できると良い距離を最小化しやすいです。\n"
                                    "## Constraint2  \n- この制約条件は画像を深く読み込んで考えてください。  \n- 最終的な巡回路は、画像に表示された立ち入り禁止エリア（黄色と黒のエリア）を横切ってはならない。  \n- ノード間の移動は直線距離の移動を前提として考えてください。",
                                ]
                            ),
                        ),
                    },
                ],
            },
        ]
        return messages
