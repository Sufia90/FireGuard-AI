"""Simple TF-IDF vector database: chunk, index, and cosine-rank retrieval.

Each hit carries source filename, page number (for PDFs), chunk id, a text
excerpt, and the similarity score — everything the dashboard needs to show
"Source: SRD_BASS-II.pdf — page 7, chunk 42".
"""
from sklearn.metrics.pairwise import cosine_similarity

from rag.document_ingestion import chunk_text
from rag.embeddings import TfidfEmbedder


class SimpleVectorStore:
    def __init__(self, embedder=None):
        self.embedder = embedder or TfidfEmbedder()
        self.chunks = []
        self._matrix = None

    def add_documents(self, documents, chunk_size=400, overlap=50):
        """Chunk documents and (re)build the TF-IDF index."""
        for doc in documents:
            page_texts = doc.get("page_texts")
            if page_texts:
                for page_no, text in page_texts:
                    for chunk in chunk_text(text, chunk_size, overlap):
                        self.chunks.append({"source": doc["source"],
                                            "text": chunk, "page": page_no})
            else:
                for chunk in chunk_text(doc["text"], chunk_size, overlap):
                    self.chunks.append({"source": doc["source"],
                                        "text": chunk, "page": None})
        if not self.chunks:
            print("SimpleVectorStore: no chunks to index.")
            return 0
        self._matrix = self.embedder.fit_transform(
            [c["text"] for c in self.chunks])
        print(f"SimpleVectorStore: indexed {len(self.chunks)} chunk(s).")
        return len(self.chunks)

    def query(self, text, top_k=5):
        """Return the top_k chunks ranked by cosine similarity."""
        if self._matrix is None or not self.chunks:
            return []
        query_vec = self.embedder.transform([text])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i],
                        reverse=True)
        hits = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            hits.append({
                "source": self.chunks[i]["source"],
                "page": self.chunks[i].get("page"),
                "chunk_id": i,
                "excerpt": self.chunks[i]["text"][:300].strip(),
                "score": float(scores[i]),
            })
        return hits

    def __len__(self):
        return len(self.chunks)
