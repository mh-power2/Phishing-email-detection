from data_preprocessing import prepare_data
from model_training import (
    train_sklearn_model, train_keras_model, build_lstm, build_cnn
)
from utils import get_metrics, save_results_to_excel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Prepare Data
X_train, X_test, y_train, y_test, tokenizer, le = prepare_data('Phishing_Email.csv')

# Models to Train
results = []
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(hidden_layer_sizes=(30, 40, 50, 20), solver="adam"),
}

for name, model in models.items():
    metrics = train_sklearn_model(model, X_train, y_train, X_test, y_test, get_metrics)
    metrics["Model"] = name
    results.append(metrics)

# Train LSTM
lstm_model = build_lstm(input_dim=len(tokenizer.word_index) + 1, max_len=150)
metrics = train_keras_model(lstm_model, X_train, y_train, X_test, y_test, get_metrics)
metrics["Model"] = "LSTM"
results.append(metrics)

# Train CNN
cnn_model = build_cnn(input_dim=len(tokenizer.word_index) + 1, max_len=150)
metrics = train_keras_model(cnn_model, X_train, y_train, X_test, y_test, get_metrics)
metrics["Model"] = "CNN"
results.append(metrics)

# Save Results
save_results_to_excel(results)
