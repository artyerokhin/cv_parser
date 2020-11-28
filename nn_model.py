from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Dropout,
    Embedding,
    SpatialDropout1D,
    Bidirectional,
    concatenate,
)
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from eli5.lime import TextExplainer

import regex as re
import pandas as pd
import numpy as np
import pickle


# модель
class KerasTextClassifier(BaseEstimator, TransformerMixin):
    """Wrapper class for keras text classification models that takes raw text as input."""

    def __init__(
        self,
        max_words=30000,
        input_length=1000,
        emb_dim=20,
        n_classes=4,
        epochs=5,
        batch_size=32,
        model_path="neural_model.hdf5",
        tokenizer_path="tokenizer.pkl",
    ):
        self.max_words = max_words
        self.input_length = input_length
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.epochs = epochs
        self.bs = batch_size
        self.model_path = model_path
        self.model = self._get_model()
        self.tokenizer_path = tokenizer_path
        self.tokenizer = TfidfVectorizer(
            token_pattern="[a-zA-zА-яа-яёЁ]+",
            max_features=self.input_length,
            ngram_range=(1, 3),
        )

    def _get_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.input_length, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_classes, activation="softmax"))
        opt = Adam(learning_rate=0.01)
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def _get_sequences(self, texts):
        return self.tokenizer.transform(texts).toarray()

    def _preprocess(self, texts):
        return [re.sub(r"\d", "DIGIT", x) for x in texts]

    def fit(self, X, y):
        """
        Fit the vocabulary and the model.

        :params:
        X: list of texts.
        y: labels.
        """

        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.model_path,
            save_weights_only=False,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )

        self.tokenizer.fit(self._preprocess(X))
        with open(self.tokenizer_path, "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        seqs = self._get_sequences(self._preprocess(X))
        self.model.fit(
            seqs,
            y,
            batch_size=self.bs,
            epochs=self.epochs,
            validation_split=0.1,
            callbacks=[model_checkpoint_callback],
        )

    def predict_proba(self, X, y=None):
        seqs = self._get_sequences(self._preprocess(X))
        return self.model.predict(seqs)

    def predict(self, X, y=None):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def save(self):
        with open(self.tokenizer_path, "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.model.save(self.model_path)

    def load(self):
        with open(self.tokenizer_path, "rb") as handle:
            self.tokenizer = pickle.load(handle)
        self.model = load_model(self.model_path)


# top_k accuracy
def top_k_accuracy(y_pred, y_true, k=3):
    top_preds = np.argsort(y_pred, axis=1)[:, -k:]
    return np.mean([int(y_true[n]) in pred for n, pred in enumerate(top_preds)])
