import json
import fitz
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter



PDF_DIR = Path("data/papers")
METADATA_FILE = Path("data/papers_metadata.json")
OUTPUT_FILE = Path("data/paper_chunks.json")



with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

metadata_lookup = {
    item["id"]: item
    for item in metadata
}


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_chunks = []



for pdf_path in PDF_DIR.glob("*.pdf"):

    paper_id = pdf_path.stem

    if paper_id not in metadata_lookup:
        print(f"Skipping {paper_id}")
        continue

    print(f"Processing {pdf_path.name}")

    doc = fitz.open(pdf_path)

    text = ""

    for page in doc:
        text += page.get_text()

    chunks = splitter.split_text(text)

    meta = metadata_lookup[paper_id]

    year = None

    if meta.get("published"):
        year = int(meta["published"][:4])

    for idx, chunk in enumerate(chunks):

        all_chunks.append(
            {
                "chunk_id": f"{paper_id}_{idx}",
                "paper_id": paper_id,
                "text": chunk,
                "title": meta["title"],
                "year": year,
                "source_type": "paper"
            }
        )

    print(f"  -> {len(chunks)} chunks")



with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_chunks,
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"\nTotal chunks: {len(all_chunks)}")