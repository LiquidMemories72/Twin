import fitz
from pathlib import Path

pdf = next(Path("data/papers").glob("*.pdf"))

doc = fitz.open(pdf)

text = ""

for page in doc:
    text += page.get_text()

print(text[:1000])