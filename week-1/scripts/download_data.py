
import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

API_URL = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_CATEGORIES = ["cs.CL", "cs.LG", "cs.CV", "cs.AI", "stat.ML"]
PAGE_SIZE = 200
OUT_PATH = Path("data/raw/arxiv_abstracts.jsonl")


def fetch_page(category: str, start: int, retries: int = 3) -> list[dict]:
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": PAGE_SIZE,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(API_URL, params=params, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            print(f"  request failed ({exc}); retry {attempt + 1}/{retries}")
            time.sleep(5 * (attempt + 1))
    else:
        return []

    root = ET.fromstring(resp.text)
    docs = []
    for entry in root.findall("atom:entry", NS):
        docs.append(
            {
                "id": entry.findtext("atom:id", "", NS).strip(),
                "title": entry.findtext("atom:title", "", NS).strip(),
                "text": entry.findtext("atom:summary", "", NS).strip(),
                "published": entry.findtext("atom:published", "", NS).strip(),
                "category": category,
            }
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=1500)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = set()
    total = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for category in args.categories:
            got = 0
            for start in range(0, args.per_category, PAGE_SIZE):
                docs = fetch_page(category, start)
                if not docs:
                    break
                for doc in docs:
                    if doc["id"] in seen_ids:
                        continue
                    seen_ids.add(doc["id"])
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    got += 1
                time.sleep(3)
            total += got
            print(f"{category}: {got} documents (total {total})")
    print(f"Saved {total} documents to {OUT_PATH}")


if __name__ == "__main__":
    main()
