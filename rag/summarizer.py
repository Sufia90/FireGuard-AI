"""Extractive (TF-IDF) summarization — fully offline, no LLM needed.

Used by the Research Assistant tab to summarize an uploaded PDF the moment
it is indexed: the top-ranked sentences are shown as the "extractive
summary", clearly labeled as TF-IDF based so nobody mistakes it for an
LLM-written abstract.
"""
import re

from sklearn.feature_extraction.text import TfidfVectorizer

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_WORDS = 4


def split_sentences(text):
    """Split into sentences, dropping fragments shorter than 4 words."""
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return [s for s in parts if len(s.split()) >= _MIN_WORDS]


def extractive_summary(text, n_sentences=6):
    """Return the n_sentences highest TF-IDF-scoring sentences, in order."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= n_sentences:
        return sentences
    matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    scores = matrix.sum(axis=1).A1
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i],
                    reverse=True)
    top = sorted(ranked[:n_sentences])
    return [sentences[i] for i in top]
