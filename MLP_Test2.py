# Keras imports 
import tensorflow as tf 
import random
from keras.utils import set_random_seed
import keras.initializers
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Input
from keras.callbacks import EarlyStopping,Callback
from keras.optimizers import Adam
from keras.regularizers import l1,l2
import keras_tuner as kt
from keras.metrics import Accuracy

# Scikit-learn imports 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,ConfusionMatrixDisplay, make_scorer
from sklearn.utils import resample
from sklearn.ensemble import BaggingClassifier # New: For Ensemble Model
from scikeras.wrappers import KerasClassifier # New: Wrapper for using Keras in Scikit-learn

# data handling file 
import MLPdataPipeline as dp
#plotting imports 
import matplotlib.pyplot as plt
import numpy as np 

# --- SEEDING FOR REPRODUCIBILITY ---
SEED = 42
set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print('working on it...')

# --- DATA LOADING ---
# Data is loaded once. X_full/y_full used for CV/Bagging training. X_test_final/y_test_final for final test.
pipeLine = dp.DataPipeline('heart_disease_uci.csv','num',test_size=.2)
pipeLine.run(binary = True)
X_full, X_test_final, y_full, y_test_final = pipeLine.getData()

# For existing functions that expect X_train/X_test, we alias them
X_train, X_test, y_train, y_test = X_full, X_test_final, y_full, y_test_final


# --- CALLBACKS & HELPERS ---
class F1History(Callback):
    """Custom callback to track F1 Score during Keras training."""
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.f1_scores = []
        self.val_f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        # ----- TRAIN F1 -----
        y_pred_train = (self.model.predict(self.X_train, verbose=0) > 0.5).astype(int)
        f1_train = f1_score(self.y_train, y_pred_train)
        self.f1_scores.append(f1_train)

        # ----- VALIDATION F1 -----
        if self.X_val.shape[0] > 0:
            y_pred_val = (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int)
            f1_val = f1_score(self.y_val, y_pred_val)
            self.val_f1_scores.append(f1_val)


def get_bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, confidence_level=0.95, metric_name="Metric"):
    """ Calculates the Confidence Interval using Bootstrapping. """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    stats = []
    
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), replace=True, random_state=i)
        boot_true = y_true[indices]
        boot_pred = y_pred[indices]
        score = metric_func(boot_true, boot_pred)
        stats.append(score)
    
    alpha = (1.0 - confidence_level) / 2.0
    lower = np.percentile(stats, alpha * 100)
    upper = np.percentile(stats, (1.0 - alpha) * 100)
    
    return lower, upper


# --- OPTIMIZED MODEL BUILDER FUNCTION ---

def build_model_cv(input_dim):
    """ 
    Builds the Keras MLP model with the optimal hyperparameters found by Keras Tuner.
    This function is used by KerasMLP and cross_val_mlp.
    """
    mlpModel = Sequential()
    mlpModel.add(Input(shape=(input_dim,)))
    
    # Optimal Hyperparameters from Tuner:
    mlpModel.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
    mlpModel.add(Dropout(0.2))

    mlpModel.add(Dense(24, activation="relu", kernel_regularizer=l2(0.0005)))
    mlpModel.add(Dropout(0.30)) 

    mlpModel.add(Dense(1, activation="sigmoid"))

    mlpModel.compile(
        optimizer=Adam(0.00031761637960126866),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return mlpModel

# --- CROSS-VALIDATION LOGIC ---
def cross_val_mlp(X, y, n_splits=5, epochs=55, batch_size=32, patience=8): # Patience increased to 8
    """
    Performs Stratified K-Fold Cross-Validation on the Keras MLP model.
    """
    print("\n" + "=" * 50)
    print(f"🔬 Starting {n_splits}-Fold Cross-Validation...")
    print("=" * 50)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    fold_acc_scores = []
    fold_f1_scores = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        
        print(f"\n--- FOLD {fold+1}/{n_splits} ---")
        
        # Use standard NumPy indexing
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # Rebuild fresh model for each fold
        model = build_model_cv(X_train_fold.shape[1])
        
        # Increased patience
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        y_pred_val = (model.predict(X_val_fold, verbose=0) > 0.5).astype(int)
        
        fold_acc = accuracy_score(y_val_fold, y_pred_val)
        fold_f1 = f1_score(y_val_fold, y_pred_val)
        
        fold_acc_scores.append(fold_acc)
        fold_f1_scores.append(fold_f1)
        
        print(f"Fold {fold+1} Validation Accuracy: {fold_acc * 100:.2f}%")
        print(f"Fold {fold+1} Validation F1 Score: {fold_f1 * 100:.2f}%")
        
    # Summarize results
    mean_acc = np.mean(fold_acc_scores)
    std_acc = np.std(fold_acc_scores)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    
    print("\n" + "=" * 50)
    print("✨ Cross-Validation Final Results (Mean ± Std Dev)")
    print("-" * 50)
    print(f"Avg Accuracy: {mean_acc * 100:.2f}% (± {std_acc * 100:.2f}%)")
    print(f"Avg F1 Score: {mean_f1 * 100:.2f}% (± {std_f1 * 100:.2f}%)")
    print("=" * 50)
    
    return fold_acc_scores, fold_f1_scores


# --- ENSEMBLE BAGGING CLASSIFIER ---
def ensemble_bagging_mlp(X_train, y_train, X_test, y_test, n_estimators=10):
    """
    Creates an ensemble using Bagging with the Keras MLP as the base estimator.
    This is intended to reduce the model's performance variance.
    """
    print("\n" + "=" * 75)
    print(f"🛡️ Starting Ensemble Bagging ({n_estimators} Models)...")
    print("=" * 75)

    # 1. Wrap the Keras model for use with Scikit-learn's BaggingClassifier
    # We pass build_fn to the KerasClassifier, and then provide the fit arguments
    # via **fit_params.
    keras_model_wrapper = KerasClassifier(
        model=lambda: build_model_cv(X_train.shape[1]), # Wrapper must be stateless
        loss="binary_crossentropy",
        epochs=55,
        batch_size=32,
        optimizer=Adam, # KerasClassifier wraps optimizer
        optimizer__learning_rate=0.00031761637960126866,
        callbacks=[EarlyStopping(monitor='loss', patience=8)], # Use loss since Bagging doesn't use val_data easily
        verbose=0 # Turn off verbose output from 10 models
    )

    # 2. Define the Bagging Classifier
    bagging = BaggingClassifier(
        estimator=keras_model_wrapper,
        n_estimators=n_estimators,
        max_samples=0.8, # Use 80% of data for each model (common practice)
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1 # Use all processors
    )

    # 3. Train the ensemble
    print("Training 10 ensemble models...")
    bagging.fit(X_train, y_train)
    
    # 4. Predict and Evaluate
    y_predict_train = bagging.predict(X_train)
    y_predict_test = bagging.predict(X_test)

    train_f1 = round(f1_score(y_train, y_predict_train), 2)
    test_f1 = round(f1_score(y_test, y_predict_test), 2)
    train_acc = round(accuracy_score(y_train, y_predict_train), 2)
    test_acc = round(accuracy_score(y_test, y_predict_test), 2)

    # Calculate Confidence Intervals
    acc_lower, acc_upper = get_bootstrap_ci(y_test, y_predict_test, accuracy_score)
    f1_lower, f1_upper = get_bootstrap_ci(y_test, y_predict_test, f1_score)
    
    # --- PRINTING ---
    print("\n" + "=" * 35)
    print("Bagging Ensemble Final Results")
    print("-" * 35)
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy:  {test_acc * 100:.2f}%")
    print(f"  > 95% CI:       [{acc_lower * 100:.2f}%, {acc_upper * 100:.2f}%]")
    print(f"\nTraining F1 Score: {train_f1 * 100:.2f}%")
    print(f"Testing F1 Score:  {test_f1 * 100:.2f}%")
    print(f"  > 95% CI:       [{f1_lower * 100:.2f}%, {f1_upper * 100:.2f}%]")
    print("=" * 35)
    
    return bagging


# --- PLOTTING FUNCTION (Unchanged) ---
def plot_all_metrics(history, f1_history, y_true, y_pred, model_name="MLP Model"):
    """ Plots Accuracy, F1 Score, and the Confusion Matrix in a single figure. """
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Accuracy Curves
    ax[0].plot(epochs, history.history['accuracy'], label="Train Acc", linewidth=2)
    ax[0].plot(epochs, history.history['val_accuracy'], label="Val Acc", linewidth=2)
    ax[0].set_title(f"{model_name} Accuracy Over Epochs", fontsize=12)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Plot 2: F1 Curves
    ax[1].plot(epochs, f1_history.f1_scores, label="Train F1", linewidth=2)
    ax[1].plot(epochs, f1_history.val_f1_scores, label="Val F1", linewidth=2)
    ax[1].set_title(f"{model_name} F1 Score Over Epochs", fontsize=12)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("F1 Score")
    ax[1].set_ylim([0, 1])
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Plot 3: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=["No Disease (0)", "Disease (1)"]
    )
    
    disp.plot(cmap=plt.cm.Blues, ax=ax[2], values_format='d', colorbar=False)
    ax[2].set_title("Test Set Confusion Matrix", fontsize=12)
    
    plt.tight_layout()
    plt.show()


# --- FINAL MODEL TRAINING & EVALUATION (Unchanged structure) ---
def KerasMLP():
    """
    Trains the final single MLP model on X_full and evaluates on X_test_final.
    """
    mlpModel = build_model_cv(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8, # Increased patience
        restore_best_weights=True
    )
    f1_callback = F1History(X_train, y_train, X_test, y_test)

    history = mlpModel.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=55,
        batch_size=32,
        callbacks=[early_stop, f1_callback],
        verbose=1
    )

    y_predict_train = (mlpModel.predict(X_train, verbose=0) > 0.5).astype(int)
    y_predict_test = (mlpModel.predict(X_test, verbose=0) > 0.5).astype(int)

    train_f1 = round(f1_score(y_train, y_predict_train), 2)
    test_f1 = round(f1_score(y_test, y_predict_test), 2)
    train_acc = round(accuracy_score(y_train, y_predict_train), 2)
    test_acc = round(accuracy_score(y_test, y_predict_test), 2)
    
    # Calculate Confidence Intervals using Bootstrapping
    acc_lower, acc_upper = get_bootstrap_ci(y_test, y_predict_test, accuracy_score)
    f1_lower, f1_upper = get_bootstrap_ci(y_test, y_predict_test, f1_score)
    err_lower = (1 - acc_upper) * 100
    err_upper = (1 - acc_lower) * 100
    test_error = (1 - test_acc) * 100

    # --- PRINTING ---
    print("=" * 75)
    print("\nTraining results (Point Estimates)")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Training F1 Score: {train_f1 * 100:.2f}%")
    
    print("\n" + "-"*30)
    print("FINAL SINGLE-MODEL TESTING RESULTS (with 95% CI)")
    print("-" * 30)
    
    # Accuracy Print
    print(f"Testing Accuracy:  {test_acc * 100:.2f}%")
    print(f"  > 95% CI:       [{acc_lower * 100:.2f}%, {acc_upper * 100:.2f}%]")
    
    # Error Print
    print(f"\nTesting Error:     {test_error:.2f}%")
    print(f"  > 95% CI:       [{err_lower:.2f}%, {err_upper:.2f}%]")

    # F1 Print
    print(f"\nTesting F1 Score:  {test_f1 * 100:.2f}%")
    print(f"  > 95% CI:       [{f1_lower * 100:.2f}%, {f1_upper * 100:.2f}%]")
    
    print("=" * 75)

    plot_all_metrics(
        history, 
        f1_callback, 
        y_test, 
        y_predict_test, 
        model_name="Heart Disease MLP"
    ) 
    
    return mlpModel


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # 1. Run Cross-Validation on the optimal single MLP model (for robust mean performance)
    cross_val_mlp(X_full, y_full, n_splits=5) 
    
    # 2. Run the final SINGLE MLP model training and test set evaluation
    print("\nRunning final single MLP model...")
    model = KerasMLP()
    
    # 3. Run the Bagging ENSEMBLE model to check for improved stability (reduced variance)
    print("\nRunning Bagging Ensemble Model for Stability Check...")
    ensemble_model = ensemble_bagging_mlp(X_full, y_full, X_test_final, y_test_final)