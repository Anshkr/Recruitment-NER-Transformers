from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)
from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from seqeval.scheme import IOB2
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "recruitment_bert"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main():
    test_path = DATA_DIR / "test_bert.jsonl"

    if not MODEL_DIR.exists():
        raise FileNotFoundError("Trained model not found.")

    records = load_jsonl(test_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(DEVICE)
    model.eval()

    id2label = model.config.id2label

    all_predictions = []
    all_references = []

    for record in records:
        input_ids = torch.tensor(
            [record["input_ids"]],
            device=DEVICE
        )

        attention_mask = torch.tensor(
            [record["attention_mask"]],
            device=DEVICE
        )

        true_labels = record["labels"]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        predicted_ids = torch.argmax(
            outputs.logits,
            dim=2
        )[0].cpu().tolist()

        predicted_tags = []
        reference_tags = []

        for pred_id, true_id in zip(
            predicted_ids,
            true_labels
        ):
            if true_id == -100:
                continue

            predicted_tags.append(id2label[int(pred_id)])
            reference_tags.append(id2label[int(true_id)])

        all_predictions.append(predicted_tags)
        all_references.append(reference_tags)

    # ---------- Entity-level metrics (seqeval, strict IOB2) ----------
    precision = precision_score(
        all_references,
        all_predictions,
        mode="strict",
        scheme=IOB2
    )

    recall = recall_score(
        all_references,
        all_predictions,
        mode="strict",
        scheme=IOB2
    )

    f1 = f1_score(
        all_references,
        all_predictions,
        mode="strict",
        scheme=IOB2
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    report = classification_report(
        all_references,
        all_predictions,
        mode="strict",
        scheme=IOB2,
        digits=4
    )

    print("\n========== ENTITY CLASSIFICATION REPORT ==========")
    print(report)

    # Save metrics and report.
    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report
    }

    with (DATA_DIR / "evaluation_results.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)

    # ---------- Token-level confusion matrix ----------
    flat_true = [
        tag
        for sequence in all_references
        for tag in sequence
    ]

    flat_pred = [
        tag
        for sequence in all_predictions
        for tag in sequence
    ]

    # Put "O" last so it appears as the final column.
    entity_labels = sorted(
        (set(flat_true) | set(flat_pred)) - {"O"}
    )
    all_labels = entity_labels + ["O"]

    matrix = confusion_matrix(
        flat_true,
        flat_pred,
        labels=all_labels
    )

    # Rows: true entity labels only (O row removed, it dominates the plot).
    # Columns: every label, including O, so "entity missed" errors stay visible.
    matrix = matrix[:len(entity_labels), :]

    # Convert counts to row percentages (share of each true label).
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_pct = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0
    )

    # Hide tiny values so the heatmap stays readable.
    annotations = np.where(
        matrix_pct >= 0.005,
        np.vectorize(lambda v: f"{v:.2f}")(matrix_pct),
        ""
    )

    plt.figure(figsize=(14, 11))

    sns.heatmap(
        matrix_pct,
        xticklabels=all_labels,
        yticklabels=entity_labels,
        cmap="Blues",
        annot=annotations,
        fmt="",
        annot_kws={"size": 7},
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Share of true label"}
    )

    plt.title(
        "BERT NER Token-Level Confusion Matrix "
        "(row-normalized, O row excluded)"
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        SCREENSHOT_DIR / "confusion_matrix.png",
        dpi=200
    )

    plt.close()

    print("\nEvaluation results saved.")
    print("Confusion matrix saved in screenshots folder.")


if __name__ == "__main__":
    main()