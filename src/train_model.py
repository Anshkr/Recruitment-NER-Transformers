
from pathlib import Path
import json
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "recruitment_bert"
OUTPUT_DIR = BASE_DIR / "models" / "training_checkpoints"

MODEL_NAME = "bert-base-cased"

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

ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}


def load_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def compute_metrics(eval_prediction):
    predictions, true_labels = eval_prediction

    predicted_ids = np.argmax(predictions, axis=2)

    true_predictions = []
    true_references = []

    for prediction, reference in zip(predicted_ids, true_labels):
        predicted_sequence = []
        reference_sequence = []

        for predicted_id, reference_id in zip(prediction, reference):
            if reference_id == -100:
                continue

            predicted_sequence.append(ID2LABEL[int(predicted_id)])
            reference_sequence.append(ID2LABEL[int(reference_id)])

        true_predictions.append(predicted_sequence)
        true_references.append(reference_sequence)

    return {
        "precision": precision_score(true_references, true_predictions),
        "recall": recall_score(true_references, true_predictions),
        "f1": f1_score(true_references, true_predictions),
        "accuracy": accuracy_score(true_references, true_predictions)
    }


def main():
    train_path = DATA_DIR / "train_bert.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}"
        )

    records = load_jsonl(train_path)

    if len(records) < 2:
        raise ValueError("Not enough training records to create a validation split.")

    # Split only the training dataset.
    train_records, val_records = train_test_split(
        records,
        test_size=0.20,
        random_state=42
    )

    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    print("Training resumes:", len(train_dataset))
    print("Validation resumes:", len(val_dataset))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("\nStarting BERT fine-tuning...")

    trainer.train()

    print("\nFinal validation evaluation:")
    metrics = trainer.evaluate()

    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print("\nModel saved to:", MODEL_DIR)
    print("Training completed.")


if __name__ == "__main__":
    main()
