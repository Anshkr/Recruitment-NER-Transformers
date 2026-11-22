
from pathlib import Path
import json
import re
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILES = {
    "train": PROCESSED_DIR / "train_clean.jsonl",
    "test": PROCESSED_DIR / "test_clean.jsonl",
}

OUTPUT_FILES = {
    "train": PROCESSED_DIR / "train_preprocessed.jsonl",
    "test": PROCESSED_DIR / "test_preprocessed.jsonl",
}


def normalize_text(text):
    """Normalize whitespace without changing meaningful characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_entities(text, entities):
    """Keep only entities whose offsets correctly match their text."""
    valid_entities = []

    for entity in entities:
        start = entity.get("start")
        end = entity.get("end")
        label = entity.get("label")
        entity_text = entity.get("text")

        if not isinstance(start, int) or not isinstance(end, int):
            continue

        if not isinstance(label, str) or not isinstance(entity_text, str):
            continue

        if start < 0 or end > len(text) or start >= end:
            continue

        actual_text = text[start:end]

        if actual_text == entity_text:
            valid_entities.append({
                "start": start,
                "end": end,
                "label": label,
                "text": entity_text
            })

    return valid_entities


def process_file(split):
    input_path = INPUT_FILES[split]
    output_path = OUTPUT_FILES[split]

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    total = 0
    retained = 0
    rejected = 0
    label_counts = Counter()

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for line in infile:
            if not line.strip():
                continue

            total += 1
            record = json.loads(line)

            original_text = record.get("text", "")
            entities = record.get("entities", [])

            # Preserve offsets by avoiding character-changing normalization.
            text = original_text

            valid_entities = validate_entities(text, entities)

            if not text.strip() or not valid_entities:
                rejected += 1
                continue

            record["text"] = text
            record["entities"] = valid_entities

            outfile.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            retained += 1
            for entity in valid_entities:
                label_counts[entity["label"]] += 1

    print(f"\n{split.upper()} SET")
    print("-" * 40)
    print("Total records:", total)
    print("Retained records:", retained)
    print("Rejected records:", rejected)
    print("Valid entity counts:")

    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print("Saved to:", output_path)


if __name__ == "__main__":
    for split in ("train", "test"):
        process_file(split)

    print("\nPreprocessing completed.")
