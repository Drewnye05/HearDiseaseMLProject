#imports
import DataProcessing as dp
#data 
import numpy as np
import matplotlib.pyplot as plt
#sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# get data
pipeLine = dp.DataPipeline('heart_disease_uci.csv','num',test_size=0.2)
pipeLine.run(binary = True)

#get features from pipeline
feature_names = list(pipeLine.df.drop(columns=["num"]).columns)

X_train, X_test, y_train, y_test = pipeLine.getData()



def rf_sklean(): 
    rf = RandomForestClassifier(
        n_estimators=100,     # trees
        random_state = 42,      # seed
        n_jobs=-1             # speeds up processing 
    )
    # train
    rf.fit(X_train, y_train)
    #pred
    y_pred = rf.predict(X_test)
    #accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
 

def rf_grid_search():
    print("Starting Random Forest....\nShould take A few minuets....")
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
        "bootstrap": [True, False]                 # bootstrap methods
    }


    # init grid search
    GS = GridSearchCV(
        estimator=rf,
        param_grid=grid,
        cv=5,             # 5-K CV
        scoring="accuracy",
        verbose=1,        # show progress (optional)
        n_jobs=-1
    )

    # Train models on all hyperparameter combinations
    GS.fit(X_train, y_train)

    # Scores
    print("Best Hyperparameters:", GS.best_params_)
    print(f"Best Cross-Val Accuracy:{GS.best_score_:.2f}")

    # Evaluate on test dataset
    best_model = GS.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report\n", classification_report(y_test,y_pred))

    return best_model

best_rf_model = rf_grid_search()
print("Generating Feautre Importance Plot...")


def plot_feature_importance(model,features):
    
    feature_names = list(features)

    importances = model.feature_importances_

    plt.figure(figsize=(8, 5))
    plt.bar(feature_names, importances)
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

plot_feature_importance(best_rf_model,feature_names)