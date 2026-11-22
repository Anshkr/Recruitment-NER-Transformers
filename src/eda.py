
from pathlib import Path
import json
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Train": DATA_DIR / "train_preprocessed.jsonl",
    "Test": DATA_DIR / "test_preprocessed.jsonl",
}


def load_data(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    train_data = load_data(FILES["Train"])
    test_data = load_data(FILES["Test"])

    all_data = train_data + test_data

    print("\n========== DATASET OVERVIEW ==========")
    print("Training resumes:", len(train_data))
    print("Testing resumes:", len(test_data))
    print("Total resumes:", len(all_data))

    # Entity distribution
    entity_counts = Counter(
        entity["label"]
        for record in all_data
        for entity in record["entities"]
    )

    entity_df = pd.DataFrame(
        entity_counts.items(),
        columns=["Entity", "Count"]
    ).sort_values("Count", ascending=False)

    print("\n========== ENTITY DISTRIBUTION ==========")
    print(entity_df.to_string(index=False))

    # Resume-level statistics
    resume_stats = []

    for record in all_data:
        resume_stats.append({
            "Text Length": len(record["text"]),
            "Entity Count": len(record["entities"]),
            "Split": record.get("split", "Unknown")
        })

    stats_df = pd.DataFrame(resume_stats)

    print("\n========== RESUME STATISTICS ==========")
    print(stats_df[["Text Length", "Entity Count"]].describe())

    # Chart 1: Entity distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=entity_df,
        x="Entity",
        y="Count"
    )
    plt.title("Distribution of Named Entities")
    plt.xlabel("Entity Category")
    plt.ylabel("Number of Entities")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "entity_distribution.png")
    plt.close()

    # Chart 2: Resume text lengths
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=stats_df,
        x="Text Length",
        bins=30,
        kde=True
    )
    plt.title("Resume Text Length Distribution")
    plt.xlabel("Number of Characters")
    plt.ylabel("Number of Resumes")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "resume_length_distribution.png")
    plt.close()

    # Chart 3: Entities per resume
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=stats_df,
        x="Entity Count",
        bins=20,
        kde=True
    )
    plt.title("Number of Entities per Resume")
    plt.xlabel("Number of Entities")
    plt.ylabel("Number of Resumes")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "entities_per_resume.png")
    plt.close()

    print("\nCharts saved in:", SCREENSHOT_DIR)
    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
