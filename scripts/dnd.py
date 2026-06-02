# scripts/download_papers.py

import arxiv
import requests
import json
from pathlib import Path

papers_dir = Path("data/papers")
papers_dir.mkdir(parents=True, exist_ok=True)

metadata = []

client = arxiv.Client()

search = arxiv.Search(
    query='au:"Yann LeCun"',
    max_results=30,
    sort_by=arxiv.SortCriterion.Relevance
)

for i, paper in enumerate(client.results(search), start=1):

    try:
        paper_id = paper.get_short_id()

        pdf_path = papers_dir / f"{paper_id}.pdf"

        if not pdf_path.exists():

            response = requests.get(
                paper.pdf_url,
                timeout=30
            )

            response.raise_for_status()

            with open(pdf_path, "wb") as f:
                f.write(response.content)

        metadata.append(
            {
                "id": paper_id,
                "title": paper.title,
                "published": str(paper.published.date()),
                "updated": str(paper.updated.date()),
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "entry_id": paper.entry_id,
                "categories": paper.categories,
                "primary_category": paper.primary_category,
                "source_type": "paper"
            }
        )

        print(f"[{i}] Downloaded: {paper.title}")

    except Exception as e:
        print(f"Failed: {paper.title}")
        print(e)

with open(
    "data/papers_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

print(f"\nDownloaded {len(metadata)} papers.")