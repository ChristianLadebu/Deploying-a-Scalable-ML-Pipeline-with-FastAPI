import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

# TODO: load the cencus.csv data
# Use the folder this script is in as the project root.
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)

data = pd.read_csv(data_path)

# TODO: split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["salary"],
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# TODO: use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Reset slice output file each run (so it doesn't keep appending forever)
slice_out_path = os.path.join(project_path, "slice_output.txt")
with open(slice_out_path, "w", encoding="utf-8") as f:
    f.write("")

# Compute the performance on model slices
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        slice_df = test[test[col] == slicevalue]
        count = slice_df.shape[0]

        # Process this slice (reuse encoder/lb)
        X_slice, y_slice, _, _ = process_data(
            slice_df,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        # Inference + metrics
        slice_preds = inference(model, X_slice)
        p, r, fb = compute_model_metrics(y_slice, slice_preds)

        with open(slice_out_path, "a", encoding="utf-8") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
            print("", file=f)

# Save model + preprocessing artifacts
save_model(model=model, encoder=encoder, lb=lb)
