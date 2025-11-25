
import os
from pathlib import Path

from dotenv import load_dotenv

from utils.midas_api import MidasClient


## This is only necessary to run if you want to download additional paper ids to the paper_ids.tsv file.

def _main() -> None:
    load_dotenv()
    api_key = os.getenv("MIDAS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MIDAS_API_KEY in environment/.env")

    client = MidasClient(api_key=api_key)

    ids = client.get_new_paper_ids("2025-06-01")

    papersAsJsonArray = []
    out_path = Path(__file__).resolve().parents[1] / "data" / "paper_ids.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in ids:
            data = client.get_paper(i)
            paperId = data.get("paperID")
            paperTitle = data.get("title")
            f.write(f"{paperId}\t{paperTitle}\n")
            papersAsJsonArray.append(data)

    with (Path(__file__).resolve().parents[1] / "data" / "papers.json").open("w", encoding="utf-8") as f:
        import json
        json.dump(papersAsJsonArray, f, indent=2)

    print(f"Wrote {len(ids)} paper IDs to {out_path}")


if __name__ == "__main__":
    _main()
