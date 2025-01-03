import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D

def train_sklearn_model(model, X_train, y_train, X_test, y_test, get_metrics):
    start = time.time()
    model.fit(X_train, y_train)
    total_time = time.time() - start
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = get_metrics(y_test, y_pred, y_prob)
    metrics["Training Time"] = total_time
    return metrics

def build_lstm(input_dim, max_len):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=50, input_length=max_len),
        LSTM(units=100),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def build_cnn(input_dim, max_len):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=50, input_length=max_len),
        Conv1D(128, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def train_keras_model(model, X_train, y_train, X_test, y_test, get_metrics, epochs=40, batch_size=16):
    start = time.time()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    total_time = time.time() - start
    results = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    metrics = get_metrics(y_test, y_pred, y_pred_prob)
    metrics["Training Time"] = total_time
    metrics["Loss"] = results[0]
    metrics["Accuracy"] = results[1]
    return metrics