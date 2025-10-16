
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

    ids = client.get_new_paper_ids("2020-01-01")

    out_path = Path(__file__).resolve().parents[1] / "data" / "paper_ids.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in ids:
            data = client.get_paper(i)
            paperId = data.get("paperID")
            paperTitle = data.get("title")
            f.write(f"{paperId}\t{paperTitle}\n")

if __name__ == "__main__":
    _main()
