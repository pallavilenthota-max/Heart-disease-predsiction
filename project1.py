
# HEART DISEASE PREDICTION
# Comparison of 7 Machine Learning Models

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# STEP 1: DATASET PATH
# --------------------------------------------------

HEART_PATH = r"C:\Users\palla\Downloads\Heart_Disease_Detection.csv"

OUTPUT_DIR = "heart_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# STEP 2: LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(HEART_PATH)

# Remove extra spaces from column names
df.columns = df.columns.astype(str).str.strip()

print("\nHEART DISEASE DATASET")
print("-" * 50)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# --------------------------------------------------
# STEP 3: AUTOMATICALLY FIND TARGET COLUMN
# --------------------------------------------------

# Match column names without depending on capitalization
# or spaces between words.

def normalize_column(name):
    return "".join(
        ch.lower() for ch in str(name)
        if ch.isalnum()
    )

target_candidates = [
    "heart disease",
    "HeartDisease",
    "Heart Disease",
    "target",
    "num",
    "output",
    "condition",
    "diagnosis",
    "disease",
    "result"
]

normalized_columns = {
    normalize_column(col): col
    for col in df.columns
}

target = None

for candidate in target_candidates:
    key = normalize_column(candidate)

    if key in normalized_columns:
        target = normalized_columns[key]
        break

if target is None:
    raise ValueError(
        "Could not identify the target column. "
        "Check the column names printed above and set "
        "target manually to the correct disease column."
    )

print("\nDetected target column:", target)

# --------------------------------------------------
# STEP 4: CLEAN THE DATA
# --------------------------------------------------

df = df.dropna(subset=[target]).copy()

X = df.drop(columns=[target])
y_original = df[target]

# Remove columns that contain only missing values
X = X.dropna(axis=1, how="all")

# Remove identifier-like columns if present
id_columns = [
    col for col in X.columns
    if normalize_column(col) in [
        "id", "patientid", "patientnumber", "recordid"
    ]
]

if id_columns:
    X = X.drop(columns=id_columns)
    print("\nRemoved identifier columns:", id_columns)

# Convert features to numeric values
X = X.apply(pd.to_numeric, errors="coerce")

# Remove rows where all input features are missing
valid_rows = ~X.isna().all(axis=1)

X = X.loc[valid_rows]
y_original = y_original.loc[valid_rows]

print("\nInput features:")
print(X.columns.tolist())

print("\nOriginal target values:")
print(y_original.value_counts())

# --------------------------------------------------
# STEP 5: CONVERT TARGET TO BINARY CLASSIFICATION
# --------------------------------------------------

# For standard HeartDisease datasets:
# 0 = No disease
# 1 = Disease
#
# For Cleveland-style datasets:
# 0 = No disease
# 1, 2, 3, 4 = Disease
#
# For datasets using 1 = No disease and 2 = Disease:
# 1 = No disease
# 2 = Disease

target_name = normalize_column(target)

target_numeric = pd.to_numeric(
    y_original,
    errors="coerce"
)

if target_numeric.notna().all():

    unique_values = set(target_numeric.unique())

    if unique_values.issubset({0, 1}):
        y = (target_numeric == 1).astype(int)

    elif unique_values.issubset({1, 2}):
        # Common alternative coding:
        # 1 = no disease, 2 = disease
        y = (target_numeric == 2).astype(int)

    elif target_name == "num":
        # Cleveland dataset: values above 0 indicate disease
        y = (target_numeric > 0).astype(int)

    else:
        raise ValueError(
            "Unexpected numeric target values: "
            + str(sorted(unique_values))
            + ". Please confirm the dataset's target encoding."
        )

else:
    # Handle labels such as Yes/No or Presence/Absence
    label_values = (
        y_original.astype(str)
        .str.strip()
        .str.lower()
    )

    positive_labels = {
        "yes", "true", "present", "presence",
        "disease", "positive", "1", "1.0",
        "heart disease", "diseased"
    }

    negative_labels = {
        "no", "false", "absent", "absence",
        "healthy", "negative", "0", "0.0",
        "no disease", "normal"
    }

    unique_labels = set(label_values.unique())

    if unique_labels.issubset(
        positive_labels | negative_labels
    ):
        y = label_values.isin(positive_labels).astype(int)

    else:
        raise ValueError(
            "Cannot safely interpret target labels: "
            + str(sorted(unique_labels))
        )

y = pd.Series(y, index=X.index, name="Disease")

print("\nTarget mapping:")
print("0 = No disease")
print("1 = Disease")

print("\nProcessed target distribution:")
print(y.value_counts())

if y.nunique() != 2:
    raise ValueError(
        "The target must contain both disease classes."
    )

if y.value_counts().min() < 2:
    raise ValueError(
        "Each class must have at least two records "
        "for a stratified train-test split."
    )

# --------------------------------------------------
# STEP 6: TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# STEP 7: DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced"
        ))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            random_state=42
        ))
    ]),

    "Gaussian Naive Bayes": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ])
}

# --------------------------------------------------
# STEP 8: TRAIN AND EVALUATE ALL MODELS
# --------------------------------------------------

results = []
predictions = {}

for name, model in models.items():

    print("\nTraining model:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

# --------------------------------------------------
# STEP 9: DISPLAY MODEL COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("HEART DISEASE MODEL COMPARISON")
print("=" * 70)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "heart_model_comparison.csv"
    ),
    index=False
)

# --------------------------------------------------
# STEP 10: MODEL COMPARISON GRAPH
# --------------------------------------------------

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-score"]
].plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1)
)

plt.title("Heart Disease Prediction - Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Evaluation Score")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "heart_model_comparison.png"
    ),
    dpi=300
)

plt.show()

# --------------------------------------------------
# STEP 11: CONFUSION MATRIX FOR BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_predictions = predictions[best_model_name]

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title(
    "Confusion Matrix - " + best_model_name
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "heart_confusion_matrix.png"
    ),
    dpi=300
)

plt.show()

# --------------------------------------------------
# STEP 12: TARGET DISTRIBUTION GRAPH
# --------------------------------------------------

target_counts = y.value_counts().reindex(
    [0, 1],
    fill_value=0
)

target_counts.index = [
    "No Disease",
    "Disease"
]

target_counts.plot(
    kind="bar",
    figsize=(6, 4)
)

plt.title("Heart Disease Target Distribution")
plt.xlabel("Target Class")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "heart_target_distribution.png"
    ),
    dpi=300
)

plt.show()

# --------------------------------------------------
# STEP 13: CLASSIFICATION REPORT
# --------------------------------------------------

print("\nBEST MODEL BY F1-SCORE:")
print(best_model_name)

print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["No Disease", "Disease"],
        zero_division=0
    )
)

# --------------------------------------------------
# STEP 14: FINAL OUTPUT
# --------------------------------------------------

print("\nPROJECT COMPLETED SUCCESSFULLY")

print(
    "\nResults and graphs saved in:",
    os.path.abspath(OUTPUT_DIR)
)

print("\nGenerated files:")
print("1. heart_model_comparison.csv")
print("2. heart_model_comparison.png")
print("3. heart_confusion_matrix.png")
print("4. heart_target_distribution.png")

