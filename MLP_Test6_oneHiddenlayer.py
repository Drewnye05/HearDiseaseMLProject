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
from keras.metrics import Accuracy, Recall 
# sklearn imports 
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,ConfusionMatrixDisplay,recall_score
from sklearn.utils import resample
from imblearn.combine import SMOTETomek
# data handling file 
# import DataProcessing as dp
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
pipeLine = dp.DataPipeline('heart_disease_uci.csv','num',test_size=.2)
pipeLine.run(binary = True)
X_full, X_test_final, y_full, y_test_final = pipeLine.getData()

# Aliases for compatibility
X_train, X_test, y_train, y_test = X_full, X_test_final, y_full, y_test_final


# --- CALLBACKS & HELPERS ---
class F1History(Callback):
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

# --- MODEL BUILDING FUNCTIONS ---

# Function for Keras Tuner (Simplified to ONE Hidden Layer)
def build_model(hp: kt.HyperParameters):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))

    # Layer 1 (Single Hidden Layer)
    hp_units = hp.Int("units", min_value=16, max_value=64, step=16)
    model.add(Dense(
        hp_units, 
        activation="relu",
        kernel_regularizer=l2(hp.Choice("l2_reg", [1e-4, 5e-4, 1e-3]))
    ))
    model.add(Dropout(hp.Float("drop_rate", min_value=.2, max_value=0.5, step=0.1)))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    # Learning rate
    lr = hp.Float("learning_rate", 1e-4, 1e-3, sampling="log")

    model.compile(
        optimizer=Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", "recall"] # Includes Recall for monitoring
    )
    
    # Store batch size
    batch_size = hp.Choice("batch_size", [16, 32, 64])
    model.batch_size = batch_size 
   
    return model

# Simplified model builder for CV and Final Training (Single Hidden Layer, Placeholder HPs)
def build_model_cv(input_dim):
    """ Builds the Keras MLP model with a single hidden layer. """
    mlpModel = Sequential()
    mlpModel.add(Input(shape=(input_dim,)))
    
    # Single Hidden Layer (PLACEHOLDER HPs: Update after running tunerSearch)
    # Recommended to use the best HPs found by the tuner.
    mlpModel.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
    mlpModel.add(Dropout(0.2)) 

    mlpModel.add(Dense(1, activation="sigmoid"))

    mlpModel.compile(
        optimizer=Adam(0.0004529223684351832),
        loss="binary_crossentropy",
        metrics=["accuracy", "recall"] # Includes Recall
    )
    return mlpModel

# Function for Keras Tuner Search
def tunerSearch():
    class CustomTuner(kt.BayesianOptimization):
        def run_trial(self, trial, *args, **kwargs):
            trial.metrics.register('val_f1',direction = 'max')
            model = self.hypermodel.build(trial.hyperparameters)
            batch_size = model.batch_size
            kwargs['batch_size'] = batch_size
            history = model.fit(*args,**kwargs)
            val_data = kwargs.get('validation_data')
            if val_data:
                X_val, y_val = val_data
                y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int)
                f1 = f1_score(y_val, y_pred)
                trial.metrics.update("val_f1", f1)
                history.history['val_f1'] = [f1] * len(history.history['val_loss'])
                trial.metrics.update("val_f1", f1)
            return history
        
    tuner = CustomTuner(
        hypermodel=build_model,
        objective=[
            kt.Objective("val_accuracy", direction="max"),
            kt.Objective('val_f1', direction='max')],
        max_trials=50,
        executions_per_trial=3,
        directory="tuner_results_single_layer",
        project_name="heart_disease_mlp_single",
        overwrite =True
    )

    print("searching best hyperparameters...")

    tuner.search(
        X_train, y_train,
        validation_data = (X_test,y_test),
        epochs=60, 
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        ],
        verbose=1
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    print("\nBest Hyperparameters (Single Layer Model):")
    print(f"Units Layer 1: {best_hps.get('units')}")
    print(f"Dropout 1: {best_hps.get('drop_rate')}")
    print(f"L2 regularization 1: {best_hps.get('l2_reg')}")
    print(f"Learning rate: {best_hps.get('learning_rate')}")
    print(f"Batch Size: {best_hps.get('batch_size')}")
    print(f"Best Accuracy: {best_trial.metrics.get_last_value('val_accuracy')}")
    print(f"Best F1: {best_trial.metrics.get_last_value('val_f1')}")
    
    return tuner


# --- CROSS-VALIDATION LOGIC (NO SMOTE) ---
# --- CROSS-VALIDATION LOGIC (with SMOTETomek) ---
def cross_val_mlp(X, y, n_splits=5, epochs=55, batch_size=32, patience=5):
    """
    Performs Stratified K-Fold Cross-Validation on the Keras MLP model 
    using SMOTETomek for training fold balancing.
    """
    print("\n" + "=" * 50)
    print(f"🔬 Starting {n_splits}-Fold Cross-Validation (with SMOTETomek)...")
    print("=" * 50)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    # Initialize SMOTETomek for robust balancing
    smote_tomek = SMOTETomek(random_state=SEED) 
    
    fold_acc_scores = []
    fold_f1_scores = []
    fold_recall_scores = []
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        
        print(f"\n--- FOLD {fold+1}/{n_splits} ---")
        
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # Apply SMOTETomek ONLY to the training data 
        X_train_resampled, y_train_resampled = smote_tomek.fit_resample(
            X_train_fold, y_train_fold
        )
        
        model = build_model_cv(X_train_resampled.shape[1])
        
        early_stop = EarlyStopping(
            monitor='val_recall', # Monitoring val_recall
            mode='max',
            patience=patience,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_resampled, y_train_resampled, # Fit on RESAMPLED data
            validation_data=(X_val_fold, y_val_fold), # Validate on ORIGINAL data
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        y_pred_val = (model.predict(X_val_fold, verbose=0) > 0.5).astype(int)
        
        fold_acc = accuracy_score(y_val_fold, y_pred_val)
        fold_f1 = f1_score(y_val_fold, y_pred_val)
        fold_rec = recall_score(y_val_fold,y_pred_val)
        
        fold_acc_scores.append(fold_acc)
        fold_f1_scores.append(fold_f1)
        fold_recall_scores.append(fold_rec)
        
        print(f"Fold {fold+1} Validation Accuracy: {fold_acc * 100:.2f}%")
        print(f"Fold {fold+1} Validation F1 Score: {fold_f1 * 100:.2f}%")
        print(f"Fold {fold+1} Validation Recall Score: {fold_rec * 100:.2f}%")
        
    # Summarize results
    mean_acc = np.mean(fold_acc_scores)
    std_acc = np.std(fold_acc_scores)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    mean_rec = np.mean(fold_recall_scores)
    std_rec = np.std(fold_recall_scores)
    
    print("\n" + "=" * 50)
    print("✨ Cross-Validation Final Results (Mean ± Std Dev)")
    print("-" * 50)
    print(f"Avg Accuracy: {mean_acc * 100:.2f}% (± {std_acc * 100:.2f}%)")
    print(f"Avg F1 Score: {mean_f1 * 100:.2f}% (± {std_f1 * 100:.2f}%)")
    print(f"Avg Recall Score: {mean_rec * 100:.2f}% (± {std_rec * 100:.2f}%)")
    print("=" * 50)
    
    return fold_acc_scores, fold_f1_scores


# --- PLOTTING FUNCTION ---
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
    plt.close() # Use close to prevent display errors

# --- FINAL MODEL TRAINING & EVALUATION ---
def KerasMLP():
    """
    Trains the final model using optimal hyperparameters on the full training data (X_full, y_full) 
    and evaluates on the held-out test data (X_test_final, y_test_final).
    """
    mlpModel = build_model_cv(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor='val_recall',
        mode= 'max',
        patience=10,
        restore_best_weights=True
    )
    f1_callback = F1History(X_train, y_train, X_test, y_test)

    # Epochs set to 55 as determined by previous analysis
    history = mlpModel.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=55,
        batch_size=32,
        callbacks=[early_stop, f1_callback],
        verbose=0
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
    print("FINAL TESTING RESULTS (with 95% CI)")
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
    
    # 3. ACTION REQUIRED: RUN TUNER FIRST
    # Run the tuner to find the optimal hyperparameters for the new single-layer model.
    # The tunerSearch() function is UNCOMMENTED below and will run first.
    # After it finishes, it will print the best HPs. Use those to update build_model_cv.
    # print("\nStarting hyperparameter tuning (Single Layer Model)...")
    # tuner = tunerSearch() 

    # 1. Run Cross-Validation on the full training data (X_full, y_full)
    # This will use the PLACEHOLDER HPs until you update build_model_cv.
    print("\n--- Running Cross-Validation ---")
    cross_val_mlp(X_full, y_full, n_splits=5) 
    
    # 2. Train the final model on all data used for CV and test on the held-out set
    print("\n--- Running final MLP model training on all available data ---")
    model = KerasMLP()