
from pathlib import Path
from collections import Counter
from lxml import etree
import hashlib
import json

# Dataset locations
TRAIN_DIR = Path("data/raw/train")
TEST_DIR = Path("data/raw/test")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_annotations(file_path):
    """Extract resume text and annotated entity spans from XML."""

    parser = etree.XMLParser(
        recover=False,
        resolve_entities=False,
        no_network=True
    )

    root = etree.fromstring(file_path.read_bytes(), parser)

    text_parts = []
    position = 0
    raw_spans = []

    def append_text(value):
        nonlocal position

        if value:
            text_parts.append(value)
            position += len(value)

    def walk(element):
        append_text(element.text)

        for child in element:
            if not isinstance(child.tag, str):
                if child.tail:
                    append_text(child.tail)
                continue

            is_label = child.tag.lower() == "label"

            start = position
            walk(child)
            end = position

            if is_label:
                label_type = child.get("type", "UNKNOWN")
                raw_spans.append((start, end, label_type))

            append_text(child.tail)

    walk(root)

    full_text = "".join(text_parts)
    entities = []

    for start, end, label in raw_spans:
        entity_text = full_text[start:end]

        # Remove whitespace around the annotated entity
        left_trim = len(entity_text) - len(entity_text.lstrip())
        right_trim = len(entity_text.rstrip())

        start += left_trim
        end = (end - len(entity_text)) + right_trim

        entity_text = full_text[start:end]

        if not entity_text:
            continue

        entities.append({
            "text": entity_text,
            "label": label,
            "start": start,
            "end": end
        })

    return full_text, entities


def audit_folder(folder):
    files = sorted(folder.rglob("*.xml"))

    label_counts = Counter()
    malformed_files = []
    empty_annotations = []
    duplicate_hashes = Counter()

    total_entities = 0
    verified_spans = 0
    failed_spans = 0
    empty_resumes = 0

    for file_path in files:
        try:
            text, entities = extract_annotations(file_path)

            if not text.strip():
                empty_resumes += 1

            # Detect duplicate resume content
            normalized = " ".join(text.split()).casefold()

            if normalized:
                digest = hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest()

                duplicate_hashes[digest] += 1

            if not entities:
                empty_annotations.append(file_path.name)

            total_entities += len(entities)

            for entity in entities:
                label_counts[entity["label"]] += 1

                start = entity["start"]
                end = entity["end"]

                # Verify annotation offsets
                if text[start:end] == entity["text"]:
                    verified_spans += 1
                else:
                    failed_spans += 1

        except Exception as error:
            malformed_files.append({
                "file": file_path.name,
                "error": str(error)
            })

    duplicate_groups = sum(
        1 for count in duplicate_hashes.values()
        if count > 1
    )

    return {
        "total_files": len(files),
        "successfully_parsed": len(files) - len(malformed_files),
        "total_entities": total_entities,
        "entity_counts": dict(label_counts),
        "malformed_files": malformed_files,
        "resumes_without_annotations": empty_annotations,
        "empty_resumes": empty_resumes,
        "duplicate_groups": duplicate_groups,
        "verified_spans": verified_spans,
        "failed_spans": failed_spans,
        "resume_hashes": list(duplicate_hashes.keys())
    }


# Audit training and testing folders
print("========== TRAINING DATA AUDIT ==========")
train_report = audit_folder(TRAIN_DIR)

print(json.dumps(
    {k: v for k, v in train_report.items()
     if k != "resume_hashes"},
    indent=4
))

print("\n========== TESTING DATA AUDIT ==========")
test_report = audit_folder(TEST_DIR)

print(json.dumps(
    {k: v for k, v in test_report.items()
     if k != "resume_hashes"},
    indent=4
))

# Check for overlapping resumes between splits
train_hashes = set(train_report["resume_hashes"])
test_hashes = set(test_report["resume_hashes"])

overlap = train_hashes.intersection(test_hashes)

print("\n========== DATA LEAKAGE CHECK ==========")
print("Duplicate resumes across train and test:", len(overlap))

# Save audit report
final_report = {
    "train": {
        k: v for k, v in train_report.items()
        if k != "resume_hashes"
    },
    "test": {
        k: v for k, v in test_report.items()
        if k != "resume_hashes"
    },
    "cross_split_duplicates": len(overlap)
}

report_path = OUTPUT_DIR / "dataset_audit.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(final_report, file, indent=4)

print("\nAudit report saved to:", report_path)
print("\nDATASET AUDIT COMPLETED!")
