
from pathlib import Path
from lxml import etree
from collections import Counter
import pandas as pd
import hashlib
import json

# ==========================================
# 1. CONFIGURATION
# ==========================================

TRAIN_DIR = Path("data/raw/train")
TEST_DIR = Path("data/raw/test")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Original dataset labels mapped to standard NER labels
LABEL_MAP = {
    "Name": "PERSON",
    "Designation": "DESIGNATION",
    "Location": "LOCATION",
    "Email Address": "EMAIL",
    "Companies worked at": "COMPANY",
    "Tech Tools": "TECH_SKILL",
    "Job Specific Skills": "JOB_SKILL",
    "Years of Experience": "EXPERIENCE",
    "Degree": "DEGREE",
    "College Name": "COLLEGE",
    "Graduation Year": "GRAD_YEAR"
}


# ==========================================
# 2. EXTRACT TEXT AND ENTITY ANNOTATIONS
# ==========================================

def extract_resume(file_path):

    parser = etree.XMLParser(
        recover=False,
        resolve_entities=False,
        no_network=True
    )

    tree = etree.parse(str(file_path), parser)
    root = tree.getroot()

    text_parts = []
    position = 0
    raw_entities = []

    def add_text(value):
        nonlocal position

        if value:
            text_parts.append(value)
            position += len(value)

    def walk(element):

        add_text(element.text)

        for child in element:

            start = position

            walk(child)

            end = position

            if child.tag.lower() == "label":

                original_label = child.get("type", "")

                raw_entities.append({
                    "start": start,
                    "end": end,
                    "original_label": original_label
                })

            add_text(child.tail)

    walk(root)

    full_text = "".join(text_parts)

    entities = []

    for item in raw_entities:

        start = item["start"]
        end = item["end"]

        entity_text = full_text[start:end]

        # Trim whitespace inside annotation boundaries
        left_trim = len(entity_text) - len(entity_text.lstrip())
        right_trim = len(entity_text.rstrip())

        start += left_trim
        end = item["end"] - (len(entity_text) - right_trim)

        entity_text = full_text[start:end]

        label = LABEL_MAP.get(item["original_label"])

        if not entity_text.strip():
            continue

        if label is None:
            continue

        entities.append({
            "text": entity_text,
            "start": start,
            "end": end,
            "label": label
        })

    return full_text, entities


# ==========================================
# 3. CLEAN AND VALIDATE RESUME RECORDS
# ==========================================

def process_folder(folder, split):

    records = []
    seen = set()
    rejected = Counter()

    files = sorted(folder.rglob("*.xml"))

    for file_path in files:

        try:

            text, entities = extract_resume(file_path)

            # Calculate outer whitespace offsets
            leading = len(text) - len(text.lstrip())
            trailing = len(text.rstrip())

            # Trim only outer whitespace
            text = text[leading:trailing]

            if not text:
                rejected["empty_resume"] += 1
                continue

            valid_entities = []

            for entity in entities:

                start = entity["start"] - leading
                end = entity["end"] - leading

                # Validate entity boundaries
                if start < 0 or end > len(text) or start >= end:
                    rejected["invalid_span"] += 1
                    continue

                entity_text = text[start:end]

                # Verify original annotation text
                if entity_text != entity["text"]:
                    rejected["invalid_span"] += 1
                    continue

                valid_entities.append({
                    "text": entity_text,
                    "start": start,
                    "end": end,
                    "label": entity["label"]
                })

            # Detect duplicate resumes
            normalized = " ".join(text.split()).casefold()

            digest = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()

            if digest in seen:
                rejected["duplicate"] += 1
                continue

            seen.add(digest)

            if not valid_entities:
                rejected["no_valid_entities"] += 1
                continue

            records.append({
                "resume_id": file_path.stem,
                "split": split,
                "text": text,
                "entities": valid_entities
            })

        except Exception as error:

            rejected["processing_error"] += 1

            print(
                f"Skipped {file_path.name}: {error}"
            )

    return records, rejected


# ==========================================
# 4. PROCESS TRAINING DATA
# ==========================================

print("\n========== PROCESSING TRAINING DATA ==========")

train_records, train_rejected = process_folder(
    TRAIN_DIR, "train"
)


# ==========================================
# 5. PROCESS TESTING DATA
# ==========================================

print("\n========== PROCESSING TESTING DATA ==========")

test_records, test_rejected = process_folder(
    TEST_DIR, "test"
)


# ==========================================
# 6. CHECK CROSS-SPLIT DUPLICATES
# ==========================================

def get_hashes(records):

    hashes = set()

    for record in records:

        normalized = " ".join(
            record["text"].split()
        ).casefold()

        digest = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        hashes.add(digest)

    return hashes


train_hashes = get_hashes(train_records)
test_hashes = get_hashes(test_records)

overlap = train_hashes.intersection(test_hashes)


# ==========================================
# 7. SAVE CLEANED DATASETS
# ==========================================

def save_records(records, filename):

    output_file = OUTPUT_DIR / filename

    pd.DataFrame(records).to_json(
        output_file,
        orient="records",
        lines=True,
        force_ascii=False
    )

    print("Saved:", output_file)


save_records(train_records, "train_clean.jsonl")
save_records(test_records, "test_clean.jsonl")


# ==========================================
# 8. GENERATE CLEANING SUMMARY
# ==========================================

def entity_distribution(records):

    counts = Counter()

    for record in records:

        for entity in record["entities"]:
            counts[entity["label"]] += 1

    return dict(counts)


summary = {

    "training": {
        "original_files": len(
            list(TRAIN_DIR.rglob("*.xml"))
        ),
        "retained_records": len(train_records),
        "rejected_records": dict(train_rejected),
        "entity_distribution": entity_distribution(
            train_records
        )
    },

    "testing": {
        "original_files": len(
            list(TEST_DIR.rglob("*.xml"))
        ),
        "retained_records": len(test_records),
        "rejected_records": dict(test_rejected),
        "entity_distribution": entity_distribution(
            test_records
        )
    },

    "cross_split_duplicate_resumes": len(overlap)
}


# Save summary
summary_file = OUTPUT_DIR / "cleaning_summary.json"

with open(summary_file, "w", encoding="utf-8") as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n========== DATA CLEANING SUMMARY ==========")

print(
    "Training records retained:",
    len(train_records)
)

print(
    "Testing records retained:",
    len(test_records)
)

print(
    "Training rejected:",
    dict(train_rejected)
)

print(
    "Testing rejected:",
    dict(test_rejected)
)

print(
    "Training entity distribution:",
    entity_distribution(train_records)
)

print(
    "Testing entity distribution:",
    entity_distribution(test_records)
)

print(
    "Duplicate resumes across splits:",
    len(overlap)
)

print("\nSummary saved:", summary_file)

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")
