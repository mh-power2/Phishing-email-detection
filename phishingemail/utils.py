import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, 
    balanced_accuracy_score, matthews_corrcoef
)

def get_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "F1-Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics

def save_results_to_excel(results, filename="model_metrics.xlsx"):
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")