from tsp.experiment import run_all_solver, summarize, convert_to_pandas, main
from pathlib import Path
import json

OUTPUT_DIR = Path(__file__).parent / "__output__" / "experiment"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def test_run_all_solver():
    run_all_solver(20, 5, OUTPUT_DIR)


def test_summarize():
    with open(OUTPUT_DIR / "result.json", "r", encoding="utf-8") as f:
        result = json.load(f)
    summary = summarize(result)
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def test_main():
    main([10, 20, 30], 3, OUTPUT_DIR / "main")


def test_convert_to_pandas():
    with open(OUTPUT_DIR / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    summary_dict = {10: summary, 20: summary, 30: summary, 40: summary, 50: summary}
    df = convert_to_pandas(summary_dict)
    print(df)
    df.to_csv(OUTPUT_DIR / "summary_dict_df.csv")
