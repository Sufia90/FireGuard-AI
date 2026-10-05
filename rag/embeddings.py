"""TF-IDF embedder for the RAG pipeline.

Deliberately lightweight: no API calls, no downloads, works fully offline.
Produces sparse vectors that SimpleVectorStore ranks with cosine similarity.
Save/load support lets you persist the fitted vectorizer if you want to
reuse an index without rebuilding it.
"""
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:

    def __init__(self, **vectorizer_kwargs):
        kwargs = {"stop_words": "english"}
        kwargs.update(vectorizer_kwargs)
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.is_fitted = False

    def fit_transform(self, texts: list[str]):
        matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return matrix

    def transform(self, texts: list[str]):
        if not self.is_fitted:
            raise RuntimeError(
                "Embedder is not fitted yet — call fit_transform first.")
        return self.vectorizer.transform(texts)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TfidfEmbedder":
        embedder = cls()
        with open(Path(path), "rb") as f:
            embedder.vectorizer = pickle.load(f)
        embedder.is_fitted = True
        return embedder
