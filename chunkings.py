import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load JSON
with open("sabudh_clean_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Better splitter (reduces sentence breaking)      Splits text intelligently:

#First tries splitting by paragraphs (\n\n)
#Then lines (\n)
##Then sentences (., ?, !)
#Then words
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", "? ", "! ", " "]
)

# Fix joined words
#What it fixes:
#Extra spaces removed
def fix_text(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Check meaningful text     Keeps only chunks with > 30 alphabet characters
def is_meaningful(text):
    return len(re.findall(r"[a-zA-Z]", text)) > 30

# Remove unwanted chunks
def is_useful(text):
    if "Co-Working" in text:
        return False
    return True

# Merge broken chunks    ( fixes bad split like "Data science is great. Python is used." → ["Data science is great", ". Python is used"])
def merge_bad_chunks(chunks):
    merged = []
    buffer = ""

    for chunk in chunks:
        chunk = chunk.strip()

        if chunk.startswith(".") or len(chunk.split()) < 5:
            buffer += " " + chunk
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = chunk

    if buffer:
        merged.append(buffer.strip())

    return merged


final_data = []

# Loop all sections
for section_name, items in data.items():

    if not isinstance(items, list):
        continue
 
    for item in items: # extracting the field from json
        url = item.get("url")
        headings = item.get("headings", [])
        paragraphs = item.get("paragraphs", [])

        all_chunks = []
        seen = set()

        for para in paragraphs: # processing each paragraph

            para = fix_text(para) #Fix text

            if not is_meaningful(para): #Filter small text
                continue

            chunks = splitter.split_text(para)
            chunks = merge_bad_chunks(chunks)

            for chunk in chunks:
                chunk = chunk.strip()

                if not is_meaningful(chunk) or not is_useful(chunk):
                    continue

                # avoid duplicates but keep order
                if chunk not in seen:
                    all_chunks.append(chunk)
                    seen.add(chunk)

        #  Merge "Prerequisites" with previous chunk
        final_chunks = []
        for chunk in all_chunks:
            if chunk.lower().startswith("prerequisites"):
                if final_chunks:
                    final_chunks[-1] += " " + chunk
            else:
                final_chunks.append(chunk)

        # Save only if data exists
        if final_chunks:
            final_data.append({
                "section": section_name,
                "url": url,
                "headings": headings,
                "chunks": final_chunks
            })

# Save output
with open("chunked4_output.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(" FINAL CLEAN CHUNKING DONE!")