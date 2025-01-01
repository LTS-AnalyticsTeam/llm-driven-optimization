from tsp.simulator import obj_func, vizualize, is_valid_tour
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
ROUND_DIGITS = 2

with open(Path(__file__).parent / "schema.json", "r", encoding="utf-8") as f:
    JSON_SCHEMA = json.load(f)


with open(Path(__file__).parent / "system_prompt.txt", "r", encoding="utf-8") as f:
    PROMPT = f.read()


def img2base64(image_path: Path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")


def solve_by_llm(messages, model, prompt_log: Path = None) -> tuple[list[int], str]:
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
    tour, reasoning = output["solution"], output["reasoning"]
    # 最初と最後の要素が同じ場合、最後の要素を削除
    if tour[0] == tour[-1]:
        tour.pop()
    return tour, reasoning


def write_result(i, g, tour, reasoning):
    result = (
        f"------\n"
        f"ツアー（決定変数）: {tour}\n"
        f"ツアーの妥当性: {is_valid_tour(g, tour)}\n"
        f"目的関数の値: {round(obj_func(g, tour), ROUND_DIGITS)}\n"
        f"求解戦略: {reasoning}\n"
        f"------\n"
    )
    with open(TMP_DIR / f"result_tsp_solution_{i}.txt", "w", encoding="utf-8") as f:
        f.write(result)
    vizualize(g, tour, path=TMP_DIR / f"vizual_tsp_solution_{i}.png")
    return result


def solve(g: nx.Graph, iter_num: int = 5, llm_model: str = "gpt-4o") -> list[int]:
    if llm_model == "o1":
        return list(g.nodes)

    progress_bar = tqdm(total=iter_num + 1, desc="Loop LLM Solver", leave=False)
    i = 0

    # 初期解の出力
    tsp_img_path = TMP_DIR / f"vizual_tsp_problem.png"
    vizualize(g, path=tsp_img_path)

    node_coords = {
        n: (round(g.nodes[n]["x"], ROUND_DIGITS), round(g.nodes[n]["y"], ROUND_DIGITS))
        for n in g.nodes
    }
    distances = {(u, v): round(g[u][v]["weight"], ROUND_DIGITS) for (u, v) in g.edges}

    # TSP問題設定のシステムプロンプト
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "text", "text": f"Node coordinates: {node_coords}"},
                {"type": "text", "text": f"Edges and distances: {distances}"},
            ],
        },
    ]

    # sysmtemプロンプトに画像を入力することができないため、ユーザプロンプトで画像を入力
    messages += [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "これは巡回セールスマン問題を示す画像です。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img2base64(tsp_img_path)}"
                    },
                },
            ],
        },
    ]

    tour, reasoning = solve_by_llm(messages, llm_model, TMP_DIR / f"prompt_log_{i}.txt")
    result = write_result(i, g, tour, reasoning)
    progress_bar.update(1)

    messages += [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "これまでの出力結果です。この結果を踏まえて、より良い解を出力する戦略を考えながら巡回セールスマン問題を解いてください。",
                }
            ],
        },
    ]
    # 解の改善
    for i in range(1, iter_num + 1):
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
        tour, reasoning = solve_by_llm(
            messages, "gpt-4o", TMP_DIR / f"prompt_log_{i}.txt"
        )
        result = write_result(i, g, tour, reasoning)
        progress_bar.update(1)

    return tour
