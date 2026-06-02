import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter



TRANSCRIPT_DIR = Path("data/transcripts")
METADATA_FILE = TRANSCRIPT_DIR / "transcripts_metadata.json"
OUTPUT_FILE = Path("data/transcript_chunks.json")


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



for txt_path in TRANSCRIPT_DIR.glob("*.txt"):

    source_id = txt_path.stem

    if source_id not in metadata_lookup:
        print(f"Skipping {source_id}")
        continue

    print(f"Processing {txt_path.name}")

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = splitter.split_text(text)

    meta = metadata_lookup[source_id]

    for idx, chunk in enumerate(chunks):

        all_chunks.append(
            {
                "chunk_id": f"{source_id}_{idx}",
                "source_id": source_id,
                "text": chunk,
                "title": meta.get("title"),
                "year": meta.get("year"),
                "source_type": "transcript",
                "chunk_index": idx
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