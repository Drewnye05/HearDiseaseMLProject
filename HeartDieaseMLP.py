# Keras imports 
from keras.utils import set_random_seed
import keras.initializers
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import EarlyStopping,Callback
from keras.optimizers import Adam
from keras.regularizers import l1,l2
import keras_tuner as kt
from keras.metrics import Accuracy,F1Score
# sklearn imports 
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score
# data handling file 
import DataProcessing as dp
#plotting imports 
import matplotlib.pyplot as plt
import numpy as np 

SEED = set_random_seed(42)
print('working on it...')
# all of the data is being assigned here
pipeLine = dp.DataPipeline('heart_disease_uci.csv','num',test_size=0.2)
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
    batch_size = hp.Choice("batch size",[16,32,64,128])

    # Layer 1
    hp_units1 = hp.Int("units1", min_value=16, max_value=128, step=16)
    model.add(Dense(
        hp_units1, 
        activation="swish",
        input_shape=(X_train.shape[1],),
        kernel_regularizer=l2(hp.Choice("l2_1", [1e-6, 1e-5, 1e-4, 1e-3]))
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float("drop1", 0.0, 0.4, step=0.1)))

    # Layer 2
    hp_units2 = hp.Int("units2", min_value=8, max_value=64, step=8)
    model.add(Dense(
        hp_units2,
        activation="swish",
        kernel_regularizer=l2(hp.Choice("l2_2", [1e-6, 1e-5, 1e-4, 1e-3]))
    ))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float("drop2", 0.0, 0.4, step=0.1)))

    # Output
    model.add(Dense(1, activation="sigmoid"))

    # Learning rate
    lr = hp.Float("learning_rate", 1e-4, 5e-3, sampling="log")

    model.compile(
        optimizer=Adam(lr),
        loss="binary_crossentropy",
        batch_size = batch_size,
        metrics=["accuracy"]
    )
   
    return model

def tunerSearch():
    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=20,
        executions_per_trial=1,
        directory="tuner_results",
        project_name="heart_disease_mlp"
    )

    print("searching best hyperparameters...")

    tuner.search(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        ],
        verbose=1
    )
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
    
    mlpModel.add(Dense(32,activation="swish",input_shape=(X_train.shape[1],),kernel_regularizer=l2(0.0001)))
    mlpModel.add(BatchNormalization())
    mlpModel.add(Dropout(0.4))

    mlpModel.add(Dense(16,activation='swish',kernel_regularizer=l2(0.0001)))
    mlpModel.add(BatchNormalization())
    mlpModel.add(Dropout(0.3))

    mlpModel.add(Dense(1,activation='sigmoid'))

    mlpModel.compile(
        optimizer= Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    ealry_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    f1_callback = F1History(X_train, y_train, X_test, y_test)

    history =mlpModel.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=500,
        batch_size=64,
        callbacks=[ealry_stop,f1_callback],
        verbose = 1
    )


    y_predict_train = (mlpModel.predict(X_train)>0.5).astype(int)

    y_predict_test = (mlpModel.predict(X_test)>0.5).astype(int)

    training_f1 = round(f1_score(y_train,y_predict_train),2)

    testing_f1 = round(f1_score(y_test,y_predict_test),2)

    train_accuracy = round(accuracy_score(y_train,y_predict_train),2)

    test_accuracy = round(accuracy_score(y_test,y_predict_test),2)

    error_training = round((1 - train_accuracy)*100,2)

    error_testing = round((1-test_accuracy)*100,2)

    acc_gap = train_accuracy-test_accuracy

    f1_gap=training_f1-testing_f1


    print("="*75)

    print("\nTraining results\n")

    print(f"Training accuracy: {train_accuracy*100}%")

    print(f"Training error {error_training}%")

    print(f"Training F1 score: {training_f1*100}%")

    print("\nShowing training results\n")


    print(f"Testing accuracy: {test_accuracy*100}%")

    print(f"Testing error rate {error_testing}%")

    print(f"Testing F1 score: {testing_f1*100}%")

    print("\nshowing the gab between train and test\n")
    print(f"Accuracy gap {acc_gap*100:.2f}")
    print(f"F1 gap {f1_gap*100:.2f}")
    print("="*75)

    plot_metrics(history,f1_callback)
KerasMLP()

"""I know my model is overfitting right now but i will tune it after saturday and before our presentation"""