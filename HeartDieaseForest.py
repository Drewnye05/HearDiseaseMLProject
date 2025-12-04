#imports
import DataProcessing as dp
#data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve

# get data
pipeLine = dp.DataPipeline('heart_disease_uci.csv','num',test_size=0.2)
pipeLine.run(binary = True)


#get features from pipeline
#feature_names = list(pipeLine.df.drop(columns=["num"]).columns)



X_train, X_test, y_train, y_test = pipeLine.getData()
feature_names = pipeLine.df.drop(columns=["num", "id"]).columns.tolist()
#print(len(feature_names), len(X_train[0]))




def rf_grid_search():
    print("STARTING RANDOM FOREST.......................\nShould take A few minuets........")
    #init
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Hyperparameters
    grid = {
        #test out different hyperparams
        "n_estimators": [100, 200, 400, 600],      # trees
        "max_depth": [None, 10, 20, 30],           # depth
        "min_samples_split": [2, 5, 10],           # higher values prevent overfitting
        "min_samples_leaf": [1, 2, 4],             
        "max_features": ["sqrt", "log2", None],    # feature sampling strategy
        "bootstrap": [True],
        "class_weight": ["balanced"]
    }


    # init grid search
    GS = GridSearchCV(
        estimator=rf,
        param_grid=grid,
        cv=5,             # 5-K CV
        scoring="f1",
        verbose=1,        
        n_jobs=-1
    )

    # Train models on all hyperparameter combinations
    GS.fit(X_train, y_train)

    # Scores
    print("Best Hyperparameters:", GS.best_params_)
    print(f"Best Cross-Val F1 Score:{GS.best_score_:.2f}")

    # Evaluate on test dataset
    best_model = GS.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report\n", classification_report(y_test,y_pred))
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"Averaged F1:{f1_macro}")  

    return best_model

def plot_feature_importance(model,features):
    
    feature_names = list(features)

    importances = model.feature_importances_

    plt.figure(figsize=(8, 5))
    plt.bar(feature_names, importances)
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

#first train
best_rf_model = rf_grid_search()
#print(len(feature_names), len(best_rf_model.feature_importances_))
print("Generating Feautre Importance Plot...")
plot_feature_importance(best_rf_model,feature_names)




print("\n===== FEATURE IMPORTANCE RANKING =====")
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": best_rf_model.feature_importances_
}).sort_values(by="importance", ascending=True)

print(importance_df)

print("Dropping all features below .02")

threshold = 0.02
low_features = importance_df[importance_df["importance"] < threshold]["feature"].tolist()

print("\nDropping low-importance features:", low_features)

pipeLine.df = pipeLine.df.drop(columns=low_features, errors="ignore")

# Rebuild feature list and train/test data
#feature_names = list(pipeLine.df.drop(columns=["num"]).columns)
X_train, X_test, y_train, y_test = pipeLine.getData()
feature_names = pipeLine.df.drop(columns=["num", "id"]).columns.tolist()


print("\n========== RETRAINING MODEL WITH REDUCED FEATURES ==========")
best_rf_model = rf_grid_search()


cm = confusion_matrix(y_test, best_rf_model.predict(X_test))

 

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Reduced Feature Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


y_prob = best_rf_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
      

plt.figure(figsize=(6,4))
plt.plot(recall, precision, label="PR Curve")
plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
