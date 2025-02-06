from tsp.simulator import Simulator
import networkx as nx
from openai import OpenAI
from pathlib import Path
import shutil
import json
import base64
import uuid
from tqdm import tqdm


class LLMSolver:

    USE_IMAGE = False
    ROUND_DIGITS = 2

    def __init__(self, sim: Simulator):
        self.sim = sim
        self.TMP_DIR = Path(__file__).parent / "__tmp__" / str(uuid.uuid4())
        self.TMP_DIR.mkdir(exist_ok=True, parents=True)
        self.CLIENT = OpenAI()

        with open(Path(__file__).parent / "schema.json", "r", encoding="utf-8") as f:
            self.JSON_SCHEMA = json.load(f)

        with open(
            Path(__file__).parent / "system_prompt.txt", "r", encoding="utf-8"
        ) as f:
            self.PROMPT = f.read()

    def _img2base64(self, image_path: Path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode("utf-8")

    def _solve_by_llm(
        self, messages, model, prompt_log: Path = None
    ) -> tuple[list[int], str]:
        if prompt_log:
            masked_messages = []
            for m in messages:
                masked_m = {"role": m["role"], "content": []}
                for c in m["content"]:
                    if "image_url" in c:
                        masked_m["content"].append(
                            {"type": "text", "text": ">>>画像が入力されています。"}
                        )
                    else:
                        masked_m["content"].append(c)
                masked_messages.append(masked_m)

            with open(prompt_log, "w", encoding="utf-8") as f:
                json.dump(masked_messages, f, ensure_ascii=False, indent=2)

        if model == "gpt-4o":
            max_completion_tokens = 16384
        if model == "o1":
            max_completion_tokens = 100000

        response = self.CLIENT.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": self.JSON_SCHEMA},
            temperature=1.0,  # o1が1.0しか対応していないため、1.0に固定
            max_completion_tokens=max_completion_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        output = json.loads(response.choices[0].message.content)
        tour = output["solution"]
        # 最初と最後の要素が同じ場合、最後の要素を削除
        if len(tour) > 1:
            if tour[0] == tour[-1]:
                tour.pop()
        return tour

    def _write_result(self, i, tour):
        result = (
            f"------\n"
            f"ツアー（決定変数）: {tour}\n"
            f"ツアーの妥当性: {self.sim.is_valid_tour(tour)}\n"
            f"目的関数の値: {round(self.sim.obj_func(tour), self.ROUND_DIGITS)}\n"
            f"------\n"
        )
        with open(
            self.TMP_DIR / f"result_tsp_solution_{i}.txt", "w", encoding="utf-8"
        ) as f:
            f.write(result)
        self.sim.vizualize(tour, path=self.TMP_DIR / f"vizual_tsp_solution_{i}.png")
        return result

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
                            constraints="なし",
                        ),
                    },
                ],
            },
        ]
        return messages

    def _initalize_message(self) -> list[dict]:
        messages = self._get_system_prompt()

        # 問題の可視化
        tsp_img_path = self.TMP_DIR / f"vizual_tsp_problem.png"
        self.sim.vizualize_nodes(path=tsp_img_path)

        # sysmtemプロンプトに画像を入力することができないため、ユーザプロンプトで画像を入力
        if self.USE_IMAGE:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._img2base64(tsp_img_path)}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "これは巡回セールスマン問題を示す画像です。巡回セールスマン問題を解いてください。",
                        },
                    ],
                },
            ]
        else:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "巡回セールスマン問題を解いてください。",
                        }
                    ],
                },
            ]

        return messages

    def _solve(
        self, iter_num: int = 0, llm_model: str = "gpt-4o", save_log=False
    ) -> list[int]:
        self.TMP_DIR.mkdir(exist_ok=True, parents=True)
        progress_bar = tqdm(total=iter_num + 1, desc="Loop LLM Solver", leave=False)
        i = 0
        messages = self._initalize_message()
        tour = self._solve_by_llm(
            messages, llm_model, self.TMP_DIR / f"prompt_log_{i}.txt"
        )
        result = self._write_result(i, tour)
        progress_bar.update(1)

        # 解の改善
        if iter_num > 0:
            messages += [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "これまでの出力結果です。この結果を踏まえて、より良い解を出力する戦略を考えながら巡回セールスマン問題を解いてください。",
                        }
                    ],
                },
            ]
            for i in range(1, iter_num + 1):
                if self.USE_IMAGE:
                    messages += [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{self._img2base64(self.TMP_DIR / f"vizual_tsp_solution_{i-1}.png")}"
                                    },
                                },
                                {"type": "text", "text": result},
                            ],
                        },
                    ]
                else:
                    messages += [
                        {"role": "user", "content": [{"type": "text", "text": result}]}
                    ]
                tour = self._solve_by_llm(
                    messages, llm_model, self.TMP_DIR / f"prompt_log_{i}.txt"
                )
                result = self._write_result(i, tour)
                progress_bar.update(1)

        if not save_log:
            shutil.rmtree(self.TMP_DIR, ignore_errors=True)
        return tour

    @classmethod
    def solve(
        cls,
        sim: Simulator,
        iter_num: int = 0,
        llm_model: str = "gpt-4o",
        save_log=False,
    ) -> list[int]:
        self = cls(sim)
        tour = self._solve(iter_num, llm_model)
        return tour
