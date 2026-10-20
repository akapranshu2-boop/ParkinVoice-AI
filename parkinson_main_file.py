import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score)
from sklearn.metrics import (precision_score, recall_score, f1_score)

df = pd.read_csv("parkinsons.csv")
df["subject"] = df["name"].str.extract(r"(S\d+)")
X = df.drop(columns=["name", "status", "subject"])
y = df["status"]

groups = df["subject"]

sgkf = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)
for train_idx, test_idx in sgkf.split(X, y, groups):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

cm = confusion_matrix(y_test, predictions, labels=[0, 1])

print("Confusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("True Negative:", tn)
print("False Positive:", fp)
print("False Negative:", fn)
print("True Positive:", tp)

# Decision Tree Classifier with Stratified Group K-Fold Cross-Validation
sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_accuracies = []

for fold, (train_idx, test_idx) in enumerate(
    sgkf.split(X, y, groups), start=1
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    fold_accuracies.append(accuracy)

    print(f"Fold {fold} Accuracy: {accuracy * 100:.2f}%")

print(
    "Mean CV Accuracy:",
    round(np.mean(fold_accuracies) * 100, 2),
    "%"
)

# Random Forest Classifier with Stratified Group K-Fold Cross-Validation
sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_scores = []

for fold, (train_idx, test_idx) in enumerate(
    sgkf.split(X, y, groups), start=1
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = accuracy_score(y_test, predictions)
    rf_scores.append(score)

    print(f"Fold {fold} Accuracy: {score * 100:.2f}%")

print("Mean Random Forest Accuracy:",
      round(np.mean(rf_scores) * 100, 2), "%")

# Model Comparison with Stratified Group K-Fold Cross-Validation(Random Forest vs Decision Tree)

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

for name, estimator in models.items():
    fold_metrics = []

    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)

        fold_metrics.append({
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, predictions, zero_division=0
            )
        })

    avg = pd.DataFrame(fold_metrics).mean()

    results.append({
        "Model": name,
        **(avg * 100).round(2).to_dict()
    })

comparison = pd.DataFrame(results)
print(comparison)

# Final Model Training and Evaluation (Random Forest Classifier 5-Fold Out-of-Fold Accuracy)
sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    random_state=42
)

oof_pred = cross_val_predict(
    rf_model,
    X,
    y,
    groups=groups,
    cv=sgkf,
    method="predict"
)

print("5-Fold Out-of-Fold Accuracy:",
      round(accuracy_score(y, oof_pred) * 100, 2), "%")

print("Balanced Accuracy:",
      round(balanced_accuracy_score(y, oof_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y,
    oof_pred,
    labels=[0, 1],
    target_names=["Control", "Parkinson's group"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y, oof_pred, labels=[0, 1]))