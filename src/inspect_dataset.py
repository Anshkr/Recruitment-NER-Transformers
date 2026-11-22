
from datasets import load_dataset

# Dataset name from Hugging Face
DATASET_NAME = "yashpwr/resume-ner-training-data"

print("Loading recruitment NER dataset...")

# Download and load the dataset
dataset = load_dataset(DATASET_NAME)

# Display dataset information
print("\n========== DATASET INFORMATION ==========")
print(dataset)

# Inspect every available split
for split_name, split_data in dataset.items():

    print(f"\n========== {split_name.upper()} SPLIT ==========")

    print("Number of records:", len(split_data))
    print("Column names:", split_data.column_names)
    print("Dataset features:", split_data.features)

    # Display one sample
    print("\nFirst record:")
    sample = split_data[0]

    for key, value in sample.items():
        print(f"\n{key}:")
        print(str(value)[:1000])

print("\nDataset inspection completed!")
