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
# sklearn imports 
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score
# data handling file 
import DataProcessing as dp
#plotting imports 
import matplotlib.pyplot as plt
import numpy as np 

SEED = 42
set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print('working on it...')
# all of the data is being assigned here
pipeLine = dp.DataPipeline('heart_disease_uci.csv','num',test_size=.2)
pipeLine.run(binary = True)
X_train, X_test, y_train, y_test = pipeLine.getData()


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
        y_pred_val = (self.model.predict(self.X_val, verbose=0) > 0.5).astype(int)
        f1_val = f1_score(self.y_val, y_pred_val)
        self.val_f1_scores.append(f1_val)


def build_model(hp: kt.HyperParameters):

    model = Sequential()

    model.add(Input(shape=(X_train.shape[1],)))

    # Layer 1
    hp_units1 = hp.Int("units1", min_value=16, max_value=64, step=16)
    model.add(Dense(
        hp_units1, 
        activation="relu",
        kernel_regularizer=l2(hp.Choice("l2_1", [1e-4, 5e-4, 1e-3]))
    ))
    # model.add(BatchNormalization())
    model.add(Dropout(hp.Float("drop1", min_value=.2, max_value=0.5, step=0.1)))

    # Layer 2
    hp_units2 = hp.Int("units2", min_value=8, max_value=32, step=8)
    model.add(Dense(
        hp_units2,
        activation="relu",
        kernel_regularizer=l2(hp.Choice("l2_2", [1e-4, 5e-4, 1e-3]))
    ))
    # model.add(BatchNormalization())
    model.add(Dropout(hp.Float("drop2", min_value=0.2, max_value=0.5, step=0.1)))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    # Learning rate
    lr = hp.Float("learning_rate", 1e-4, 1e-3, sampling="log")

    model.compile(
        optimizer=Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    # Store batch size in model for access during fit
    batch_size = hp.Choice("batch_size", [16, 32, 64])
    model.batch_size = batch_size 
   
    return model

def tunerSearch():
    # Custom tuner to handle batch_size
    class CustomTuner(kt.BayesianOptimization):
        def run_trial(self, trial, *args, **kwargs):

            trial.metrics.register('val_f1',direction = 'max')
            # Get the model with batch_size stored
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
            
            
            # return super(CustomTuner, self).run_trial(trial, *args, **kwargs)
            return history
        
    tuner = CustomTuner(
        hypermodel=build_model,
        objective=[
            kt.Objective("val_accuracy", direction="max"),
            kt.Objective('val_f1', direction='max')],
        max_trials=50,
        executions_per_trial=1,
        directory="tuner_results",
        project_name="heart_disease_mlp",
        overwrite =True
    )

    print("searching best hyperparameters...")


    tuner.search(
        X_train, y_train,
        validation_data = (X_test,y_test),
        epochs=50,  # Can make this tunable by adding to build_model
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        ],
        verbose=1
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    print("\nBest Hyperparameters:")
    print(f"Units Layer 1: {best_hps.get('units1')}")
    print(f"Units Layer 2: {best_hps.get('units2')}")
    print(f"Dropout 1: {best_hps.get('drop1')}")
    print(f"Dropout 2: {best_hps.get('drop2')}")
    print(f"L2 regularization 1: {best_hps.get('l2_1')}")
    print(f"L2 regularization 2: {best_hps.get('l2_2')}")
    print(f"Learning rate: {best_hps.get('learning_rate')}")
    print(f"Batch Size: {best_hps.get('batch_size')}")
    print(f"Best Accuracy: {best_trial.metrics.get_last_value('val_accuracy')}")
    print(f"Best F1: {best_trial.metrics.get_last_value('val_f1')}")
    
    return tuner

def plot_metrics(history, f1_history):

    epochs = range(1, len(history.history['accuracy']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Curves
    ax[0].plot(epochs, history.history['accuracy'], label="Train Acc", linewidth=2)
    ax[0].plot(epochs, history.history['val_accuracy'], label="Val Acc", linewidth=2)
    ax[0].set_title("Accuracy Over Epochs", fontsize=14)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # F1 Curves
    ax[1].plot(epochs, f1_history.f1_scores, label="Train F1", linewidth=2)
    ax[1].plot(epochs, f1_history.val_f1_scores, label="Val F1", linewidth=2)
    ax[1].set_title("F1 Score Over Epochs", fontsize=14)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("F1 Score")
    ax[1].set_ylim([0, 1])
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def KerasMLP():
    mlpModel = Sequential()

    mlpModel.add(Input(shape=(X_train.shape[1],)))
    
    # mlpModel.add(Dense(128,activation="swish",kernel_regularizer=l2(0.001)))
    # mlpModel.add(BatchNormalization())
    # mlpModel.add(Dropout(0.30000000000000004))

    # mlpModel.add(Dense(64,activation='swish',kernel_regularizer=l2(0.001)))
    # mlpModel.add(BatchNormalization())
    # mlpModel.add(Dropout(0.1))

    # mlpModel.add(Dense(1,activation='sigmoid'))

    # mlpModel.compile(
    #     optimizer= Adam(0.005),
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    mlpModel.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
    mlpModel.add(Dropout(0.2))
    mlpModel.add(Dense(24, activation="relu", kernel_regularizer=l2(0.0005)))
    mlpModel.add(Dropout(0.30000000000000004))
    mlpModel.add(Dense(1, activation="sigmoid"))
    mlpModel.compile(
        optimizer=Adam(0.00031761637960126866),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    f1_callback = F1History(X_train, y_train, X_test, y_test)

    history = mlpModel.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, f1_callback],
        verbose=1
    )

    y_predict_train = (mlpModel.predict(X_train) > 0.5).astype(int)
    y_predict_test = (mlpModel.predict(X_test) > 0.5).astype(int)

    training_f1 = round(f1_score(y_train, y_predict_train), 2)
    testing_f1 = round(f1_score(y_test, y_predict_test), 2)
    train_accuracy = round(accuracy_score(y_train, y_predict_train), 2)
    test_accuracy = round(accuracy_score(y_test, y_predict_test), 2)
    error_training = round((1 - train_accuracy) * 100, 2)
    error_testing = round((1 - test_accuracy) * 100, 2)
    acc_gap = train_accuracy - test_accuracy
    f1_gap = training_f1 - testing_f1

    print("=" * 75)
    print("\nTraining results\n")
    print(f"Training accuracy: {train_accuracy * 100}%")
    print(f"Training error {error_training}%")
    print(f"Training F1 score: {training_f1 * 100}%")
    print("\nTesting results\n")
    print(f"Testing accuracy: {test_accuracy * 100}%")
    print(f"Testing error rate {error_testing}%")
    print(f"Testing F1 score: {testing_f1 * 100}%")
    print("\nShowing the gap between train and test\n")
    print(f"Accuracy gap {acc_gap * 100:.2f}%")
    print(f"F1 gap {f1_gap * 100:.2f}%")
    print("=" * 75)

    plot_metrics(history, f1_callback)
    
    return mlpModel

# Run the baseline model
if __name__ == "__main__":
    print("Running baseline MLP model...")
    model = KerasMLP()
    
    # Uncomment below to run hyperparameter tuning
    # print("\nStarting hyperparameter tuning...")
    # tuner = tunerSearch()