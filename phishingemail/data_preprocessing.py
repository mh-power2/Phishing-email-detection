import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text).lower()
    return re.sub(r'\s+', ' ', text).strip()

def prepare_data(filepath, max_len=150):
    df = pd.read_csv(filepath)
    df["Email Text"] = df["Email Text"].apply(preprocess_text)
    le = LabelEncoder()
    df["Email Type"] = le.fit_transform(df["Email Type"])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["Email Text"])
    sequences = tokenizer.texts_to_sequences(df["Email Text"])
    vector = pad_sequences(sequences, padding="post", maxlen=max_len)

    x = np.array(vector)
    y = np.array(df["Email Type"])
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test, tokenizer, le