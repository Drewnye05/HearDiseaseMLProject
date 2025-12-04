import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.inspection import permutation_importance

heart_data = pd.read_csv("heart_disease_uci.csv")
heart_data["sex"] = heart_data["sex"].str.lower()
heart_data["fbs"] = heart_data["fbs"].astype(str).str.lower()
heart_data["exang"] = heart_data["exang"].astype(str).str.lower()
heart_data = heart_data.drop(["id", "dataset", "ca", "thal"], axis=1, errors="ignore")

# one-hot encoding
heart_data = pd.get_dummies(
    heart_data,
    columns=["sex", "fbs", "exang", "cp", "restecg", "slope"],
    drop_first=True
)

# target ints to binary (they were measuring severity)
heart_data["target_value"] = (heart_data["num"] > 0).astype(int)
heart_data = heart_data.drop("num", axis=1)

# drop missing data, create X and y subsets
heart_data = heart_data.dropna()
X = heart_data.drop("target_value", axis=1)
y = heart_data["target_value"]
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

params = {
    'n_neighbors': [3, 5, 7, 8, 9, 10, 11, 12, 13],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'manhattan'],
    'p': [1, 2]
}

grid = GridSearchCV(KNeighborsClassifier(), params, cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

results = grid.cv_results_
mean_scores = results['mean_test_score']
best_running = []
current_best = -1

for score in mean_scores:
    current_best = max(current_best, score)
    best_running.append(current_best)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(best_running) + 1), best_running, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Best CV Score So Far")
plt.title("Grid Search Convergence")
plt.grid(True)
plt.show()

print("Best parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)

optimal_knn = grid.best_estimator_

final_cv_scores = cross_val_score(optimal_knn, X_train, y_train, cv=cv)
print("Final Repeated Stratified CV Accuracy on training data (using best model):",
      final_cv_scores.mean())
print("All fold accuracies:", final_cv_scores)

y_pred = optimal_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Heart Disease KNN Model Accuracy: ", accuracy)
print("\nClassification Report:\n", report)

f1_pos = f1_score(y_test, y_pred, pos_label=1)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"F1 (class 1, positive): {f1_pos:.6f}")
print(f"Macro F1: {f1_macro:.6f}")
print(f"Weighted F1 (final F1 score): {f1_weighted:.6f}")

# print confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()

perm_result = permutation_importance(
    optimal_knn,
    X_test,
    y_test,
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

importances = perm_result.importances_mean
std = perm_result.importances_std
indices = np.argsort(importances)  # sort from least to most important

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices], xerr=std[indices])
plt.yticks(range(len(indices)), np.array(feature_names)[indices])
plt.xlabel("Mean decrease in accuracy")
plt.title("Permutation Feature Importance (KNN)")
plt.tight_layout()
plt.show()

k_values = range(1, 26)
accuracies = []
f1_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, preds))
    f1_scores.append(f1_score(y_test, preds))

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', label='Accuracy')
plt.plot(k_values, f1_scores, marker='s', label='F1-score', linestyle='--')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('KNN Performance vs. k')
plt.legend()
plt.grid(True)
plt.show()

