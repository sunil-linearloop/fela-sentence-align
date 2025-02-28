import numpy as np
import os

from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps

class Encoder:
    def __init__(self, model_name):
        # Use the cache folder specified in the environment, default to /app/models
        cache_folder = os.environ.get("TRANSFORMERS_CACHE", "/app/models")
        # Try to load the model, fallback to default location if not found in cache
        try:
            self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
            print(f"Loaded {model_name} model from cache: {cache_folder}")
        except Exception as e:
            print(f"Warning: Could not load model from cache: {str(e)}")
            print(f"Trying to load model from default location...")
            self.model = SentenceTransformer(model_name)
        
        self.model_name = model_name

    def transform(self, sents, num_overlaps):
        overlaps = []
        for line in yield_overlaps(sents, num_overlaps):
            overlaps.append(line)

        sent_vecs = self.model.encode(overlaps)
        embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
        sent_vecs.resize(num_overlaps, len(sents), embedding_dim)

        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs)
        len_vecs.resize(num_overlaps, len(sents))

        return sent_vecs, len_vecs
