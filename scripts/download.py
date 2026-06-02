import arxiv
import json
from pathlib import Path

papers_dir = Path("data/papers")
papers_dir.mkdir(parents=True, exist_ok=True)

metadata = []

search = arxiv.Search(
    query='au:"Yann LeCun"',
    max_results=50,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for paper in search.results():

    try:
        paper_id = paper.get_short_id()

        paper.download_pdf(
            dirpath=str(papers_dir),
            filename=f"{paper_id}.pdf"
        )

        metadata.append(
            {
                "id": paper_id,
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "published": str(paper.published.date()),
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "source_type": "paper"
            }
        )

        print(f"Downloaded: {paper.title}")

    except Exception as e:
        print(e)

with open("data/papers_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print(f"Downloaded {len(metadata)} papers")