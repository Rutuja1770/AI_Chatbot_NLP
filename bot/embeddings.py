from sentence_transformers import SentenceTransformer

import numpy as np

import pickle

import os


class Embedder:

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        os.makedirs("models", exist_ok=True)

        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):

        return np.array(
            self.model.encode(
                sentences,
                show_progress_bar=False),
            dtype=np.float32)

    def save(self, data, path):

        with open(path, "wb") as f:

            pickle.dump(data, f)

    def load(self, path):

        if not os.path.exists(path):

            raise FileNotFoundError(f"Embedding file not found: {path}")

        with open(path, "rb") as f:

            return pickle.load(f)
