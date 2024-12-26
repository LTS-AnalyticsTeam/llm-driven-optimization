from tsp.simulator import obj_func, vizualize
import networkx as nx
from openai import OpenAI
from pathlib import Path
import json
import base64
from tqdm import tqdm
from dotenv import load_dotenv

TMP_DIR = Path(__file__).parent / "__tmp__"
TMP_DIR.mkdir(exist_ok=True, parents=True)
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)
CLIENT = OpenAI()

with open(Path(__file__).parent / "schema.json", "r", encoding="utf-8") as f:
    JSON_SCHEMA = json.load(f)


with open(Path(__file__).parent / "prompt.txt", "r", encoding="utf-8") as f:
    PROMPT = f.read()


def img2base64(image_path: Path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")


def llm(messages, model) -> tuple[list[int], str]:
    response = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    output = json.loads(response.choices[0].message.content)
    return output["solution"], output["reasoning"]


def solve(g: nx.Graph, iter_num: int = 5) -> list[int]:

    progress_bar = tqdm(total=iter_num + 1)

    # 初期解の出力
    tsp_img_path = TMP_DIR / f"vizual_tsp_problem.png"
    vizualize(g, path=tsp_img_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img2base64(tsp_img_path)}"
                    },
                },
                {"type": "text", "text": PROMPT},
            ],
        },
    ]

    tour, reasoning = llm(messages, "gpt-4o")
    progress_bar.update(1)

    messages += [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "これまでの出力結果です。この結果を踏まえて、より良い解を出力する戦略を考えてください。",
                }
            ],
        },
    ]

    # 解の改善
    for i in range(1, iter_num + 1):
        tsp_img_path = TMP_DIR / f"vizual_tsp_solution_{i}.png"
        vizualize(g, tour, path=tsp_img_path)
        result = (
            f"------"
            f"ツアー（決定変数）: {tour}\n"
            f"目的関数の値: {obj_func(g, tour)}\n"
            f"理由: {reasoning}\n"
            f"------"
        )
        with open(TMP_DIR / f"result_tsp_solution_{i}.txt", "w", encoding="utf-8") as f:
            f.write(result)

        messages += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img2base64(tsp_img_path)}"
                        },
                    },
                    {"type": "text", "text": result},
                ],
            },
        ]
        tour, reasoning = llm(messages, "gpt-4o")
        progress_bar.update(1)

    return tour
