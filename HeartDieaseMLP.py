# Keras imports 
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l1,l2
# import keras_tuner as kt
from keras.metrics import Accuracy,F1Score
# sklearn imports 
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score
# data handling file 
import DataProcessing as dp

print('working on it...')
# all of the data is being assigned here
pipeLine = dp.DataPipeline('heart_disease_uci.csv','num',test_size=0.2)
pipeLine.run(binary = True)
X_train, X_test, y_train, y_test = pipeLine.getData()

def SklearnMLP():
    mlpModel = MLPClassifier(
        hidden_layer_sizes=(128,64),
        activation= 'relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=250,
        early_stopping=True,
        random_state=42
    )
    cv_scores = cross_val_score(mlpModel,X_train,y_train,cv=5)
    
    mlpModel.fit(X_train,y_train)

    y_predict_train = mlpModel.predict(X_train)

    y_predict_test = mlpModel.predict(X_test)

    training_f1 = round(f1_score(y_train,y_predict_train),2)

    testing_f1 = round(f1_score(y_test,y_predict_test),2)

    train_accuracy = round(accuracy_score(y_train,y_predict_train),2)

    test_accuracy = round(accuracy_score(y_test,y_predict_test),2)

    error_training = round((1 - train_accuracy)*100,2)

    error_testing = round((1-test_accuracy)*100,2)

    acc_gap = train_accuracy-test_accuracy

    f1_gap=training_f1-testing_f1

    print("="*75)

    print("Cross Validation\n")
    print("CV scores: " ,cv_scores)

    print("Mean CV accuracy:",cv_scores.mean())

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

def KerasMLP():
    mlpModel = Sequential()
    
    mlpModel.add(Dense(32,activation="swish",input_shape=(X_train.shape[1],),kernel_regularizer=l2(0.0001)))
    mlpModel.add(BatchNormalization())
    mlpModel.add(Dropout(0.3))

    mlpModel.add(Dense(16,activation='relu',kernel_regularizer=l2(0.001)))
    mlpModel.add(BatchNormalization())
    mlpModel.add(Dropout(0.2))

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

    history =mlpModel.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[ealry_stop],
        verbose = 0
    )


    y_predict_train = (mlpModel.predict(X_train)>0.5).astype(int)

    y_predict_test = (mlpModel.predict(X_test)>0.5).astype(int)

    training_f1 = round(f1_score(y_train,y_predict_train),2)

    testing_f1 = round(f1_score(y_test,y_predict_test),2)

    train_accuracy = round(accuracy_score(y_train,y_predict_train),2)

    test_accuracy = round(accuracy_score(y_test,y_predict_test),2)

    error_training = round((1 - train_accuracy)*100,2)

    error_testing = round((1-test_accuracy)*100,2)



    print(f"Training accuracy: {train_accuracy*100}%")

    print(f"Training error {error_training}%")

    print(f"Training F1 score: {training_f1*100}%")

    print(f"Testing accuracy: {test_accuracy*100}%")

    print(f"Testing error rate {error_testing}%")

    print(f"Testing F1 score: {testing_f1*100}%")

#KerasMLP()
SklearnMLP()