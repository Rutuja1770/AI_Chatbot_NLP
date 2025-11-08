import pandas as pd

from sklearn.neighbors import NearestNeighbors

from transformers import AutoTokenizer, AutoModelForCausalLM

from .embeddings import Embedder

import torch

import os

class ChatBot:

    def __init__(self, kb_path="data/faq.csv"):

        # Load KB

        self.kb = pd.read_csv(kb_path, quotechar='"')

        print(f"Loaded {len(self.kb)} KB entries.")

        self.embedder = Embedder()

        # Load or compute embeddings

        self.embeddings_path = "models/kb_embeddings.pkl"

        if os.path.exists(self.embeddings_path):

            self.embeddings = self.embedder.load(self.embeddings_path)

            print("Loaded embeddings from cache.")

        else:

            self.embeddings = self.embedder.encode(self.kb["question"].tolist())

            self.embedder.save(self.embeddings, self.embeddings_path)

            print("Encoded and cached KB embeddings.")


# NearestNeighbors

        self.nn = NearestNeighbors(n_neighbors=1, metric="cosine")

        self.nn.fit(self.embeddings)

        # Load small language model

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    def retrieve(self, query):

        query_emb = self.embedder.encode([query])

        dist, idx = self.nn.kneighbors(query_emb)

        score = float(1 - dist[0][0])

        answer = self.kb.iloc[idx[0][0]]["answer"]

        print(f"DEBUG: Query='{query}', Score={score:.3f}, Answer='{answer}'")

        return score, answer

    def generate(self, query):

        input_ids = self.tokenizer.encode(query + self.tokenizer.eos_token, return_tensors="pt").to(self.device)

        output = self.model.generate(input_ids, max_length=150, pad_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

        return response
    def get_answer(self, query):

        score, answer = self.retrieve(query)

        if score > 0.5:  # threshold for KB match

            return f"(KB Answer) {answer}"

        else:

            return f"(AI Answer) {self.generate(query)}"













