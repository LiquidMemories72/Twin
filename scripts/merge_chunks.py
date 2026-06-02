import json

with open("data/paper_chunks.json", "r", encoding="utf-8") as f:
    paper_chunks = json.load(f)

with open("data/transcript_chunks.json", "r", encoding="utf-8") as f:
    transcript_chunks = json.load(f)

all_chunks = paper_chunks + transcript_chunks

with open("data/all_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print("Total chunks:", len(all_chunks))