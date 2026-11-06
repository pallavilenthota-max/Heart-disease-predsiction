import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Step 1: Load Dataset
df = pd.read_csv(r"C:\Users\palla\Downloads\Heart_Disease_Detection.csv")

print("Dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Step 2: Separate Features and Target
target = "Heart_Disease"

df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = df[target]

# Step 3: Identify Numeric and Categorical Features
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers)

# Step 4: Define SVM Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf"))
])

# Step 5: Apply 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score, zero_division=0
    ),
    "recall": make_scorer(
        recall_score, zero_division=0
    ),
    "f1": make_scorer(
        f1_score, zero_division=0
    )
}

cv_results = cross_validate(
    model, X, y, cv=skf, scoring=scoring, n_jobs=1
)

# Step 6: Display Fold-Wise Results
print("\n5-Fold Cross-Validation Results:")

for i in range(5):
    print("\nFold", i + 1)

    print(
        "Accuracy:",
        round(cv_results["test_accuracy"][i], 4)
    )

    print(
        "Precision:",
        round(cv_results["test_precision"][i], 4)
    )

    print(
        "Recall:",
        round(cv_results["test_recall"][i], 4)
    )

    print(
        "F1-Score:",
        round(cv_results["test_f1"][i], 4)
    )

# Step 7: Average Performance
print("\nAverage Cross-Validation Performance:")

for metric in scoring:
    values = cv_results["test_" + metric]

    print(
        metric.capitalize() + ":",
        round(values.mean(), 4),
        "+/-",
        round(values.std(), 4)
    )

# Step 8: Out-of-Fold Predictions
y_pred = cross_val_predict(
    model, X, y, cv=skf, n_jobs=1
)

cm = confusion_matrix(y, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

print("\nRows = Actual; Columns = Predicted")
print("Class 0 = No Disease; Class 1 = Disease")

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
).plot(cmap="Blues")

plt.title("Heart Disease Detection - 5-Fold Cross-Validation")
plt.tight_layout()
plt.show()

print("\nExercise 3 Completed Successfully!")
print("Note: This educational model is not intended for clinical diagnosis.")

