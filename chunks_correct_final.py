import json
import re

# -----------------------------
# Load JSON
# -----------------------------
with open("sabudh_clean_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# -----------------------------
# Fix text
# -----------------------------
def fix_text(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# Meaningful text check
# -----------------------------
def is_meaningful(text):
    return len(re.findall(r"[a-zA-Z]", text)) > 30

# -----------------------------
# Remove unwanted chunks
# -----------------------------
def is_useful(text):
    return "Co-Working" not in text

# -----------------------------
# REAL sliding overlap chunking
# -----------------------------
def chunk_text(text, chunk_size=400, overlap=100):

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        # If end exceeds text
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Move end to nearest word boundary
        while end < text_length and text[end] != " ":
            end += 1

        chunk = text[start:end].strip()

        chunks.append(chunk)

        # Overlap
        start = end - overlap

        # Move start to next word boundary
        while start < text_length and text[start] != " ":
            start += 1

    return chunks

# -----------------------------
# Final output
# -----------------------------
final_data = []

for section_name, items in data.items():

    if not isinstance(items, list):
        continue

    for item in items:

        url = item.get("url")
        headings = item.get("headings", [])
        paragraphs = item.get("paragraphs", [])

        # Combine all paragraphs
        full_text = " ".join(paragraphs)

        full_text = fix_text(full_text)

        if not is_meaningful(full_text):
            continue

        # -----------------------------
        # REAL overlap chunking
        # -----------------------------
        chunks = chunk_text(
            full_text,
            chunk_size=400,
            overlap=100
        )

        all_chunks = []
        seen = set()

        for chunk in chunks:

            chunk = chunk.strip()

            if not is_meaningful(chunk):
                continue

            if not is_useful(chunk):
                continue

            # Avoid duplicates
            if chunk not in seen:
                all_chunks.append(chunk)
                seen.add(chunk)

        # Save output
        if all_chunks:

            final_data.append({
                "section": section_name,
                "url": url,
                "headings": headings,
                "chunks": all_chunks
            })

# -----------------------------
# Save JSON
# -----------------------------
with open("chunked2_output.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print("✅ FINAL OVERLAP CHUNKING DONE")