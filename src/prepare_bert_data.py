
from pathlib import Path
import json

from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_NAME = "bert-base-cased"
MAX_LENGTH = 512

LABELS = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-DESIGNATION", "I-DESIGNATION",
    "B-LOCATION", "I-LOCATION",
    "B-EMAIL", "I-EMAIL",
    "B-COMPANY", "I-COMPANY",
    "B-TECH_SKILL", "I-TECH_SKILL",
    "B-JOB_SKILL", "I-JOB_SKILL",
    "B-EXPERIENCE", "I-EXPERIENCE",
    "B-DEGREE", "I-DEGREE",
    "B-COLLEGE", "I-COLLEGE",
    "B-GRAD_YEAR", "I-GRAD_YEAR"
]

LABEL2ID = {label: index for index, label in enumerate(LABELS)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def convert_record(record):
    text = record["text"]
    entities = record["entities"]

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True
    )

    offsets = encoding["offset_mapping"]
    token_labels = [LABEL2ID["O"]] * len(offsets)

    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["label"]

        begin_label = "B-" + entity_type
        inside_label = "I-" + entity_type

        if begin_label not in LABEL2ID:
            continue

        first_token = True

        for token_index, (token_start, token_end) in enumerate(offsets):

            # Ignore special tokens and tokens outside the entity.
            if token_start == token_end:
                continue

            overlaps = token_start < end and token_end > start

            if overlaps:
                if first_token:
                    token_labels[token_index] = LABEL2ID[begin_label]
                    first_token = False
                else:
                    token_labels[token_index] = LABEL2ID[inside_label]

    encoding.pop("offset_mapping")

    # Ignore special tokens when calculating training loss.
    encoding["labels"] = [
        -100 if start == end else label
        for (start, end), label in zip(offsets, token_labels)
    ]

    return encoding


def process_split(split):
    input_path = DATA_DIR / f"{split}_preprocessed.jsonl"
    output_path = DATA_DIR / f"{split}_bert.jsonl"

    records = load_jsonl(input_path)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            encoding = convert_record(record)

            output = {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": encoding["labels"]
            }

            file.write(
                json.dumps(output, ensure_ascii=False) + "\n"
            )

    print(f"{split.upper()} records processed: {len(records)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    print("Loading tokenizer:", MODEL_NAME)

    process_split("train")
    process_split("test")

    print("\nBERT data preparation completed.")
    print("Number of labels:", len(LABELS))
