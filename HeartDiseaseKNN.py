import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

heart_data = pd.read_csv("heart_disease_uci.csv")
heart_data["sex"] = heart_data["sex"].str.lower()
heart_data["fbs"] = heart_data["fbs"].astype(str).str.lower()
heart_data["exang"] = heart_data["exang"].astype(str).str.lower()
heart_data = heart_data.drop(["id", "dataset", "ca", "thal"], axis=1, errors="ignore")
#onhot encoding
heart_data = pd.get_dummies(heart_data, columns=["sex","fbs","exang","cp","restecg","slope"],drop_first=True)
#target ints to binary (they were measuring severity)

heart_data["target_value"] = (heart_data["num"] > 0).astype(int)
heart_data = heart_data.drop("num", axis = 1)

#drop missing data, create X and y subsets
heart_data = heart_data.dropna()
X = heart_data.drop("target_value", axis=1)
y = heart_data["target_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

params = {'n_neighbors': [3, 5, 7, 9, 11,],'weights': ['uniform', 'distance'],'metric': ['minkowski', 'manhattan'],'p': [1, 2]}
grid = GridSearchCV(KNeighborsClassifier(), params, cv=5, n_jobs=-1)
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

y_pred = optimal_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Heart Disease KNN Model Accuracy: ", accuracy)
print("\nClassification Report:\n", report)


#print confusion matrix and f1 scores
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
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
