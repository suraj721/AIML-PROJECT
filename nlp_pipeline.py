import io
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import spacy
from spacy.language import Language

from pypdf import PdfReader


_EN_CORE_MODEL = "en_core_web_sm"


def _load_spacy_model() -> Optional[Language]:
    try:
        return spacy.load(_EN_CORE_MODEL)
    except OSError:
        return None


@dataclass
class PreprocessingResources:
    nlp: Optional[Language]
    fallback_stopwords: set


def build_preprocessing_resources() -> PreprocessingResources:
    nlp = _load_spacy_model()
    fallback_stopwords = {
        "the",
        "and",
        "is",
        "in",
        "it",
        "of",
        "to",
        "a",
        "an",
        "for",
        "on",
        "with",
        "that",
        "this",
        "by",
        "from",
        "as",
        "at",
        "be",
        "are",
        "or",
        "we",
        "our",
        "their",
    }
    return PreprocessingResources(nlp=nlp, fallback_stopwords=fallback_stopwords)


def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    texts: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        texts.append(text)
    return "\n".join(texts)


def read_text_file(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def preprocess_document(text: str, resources: PreprocessingResources) -> str:
    text = text.lower()
    tokens: List[str] = []

    if resources.nlp is not None:
        doc = resources.nlp(text)
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            lemma = token.lemma_.strip()
            if not lemma or not lemma.isalpha():
                continue
            tokens.append(lemma)
    else:
        for raw in text.split():
            word = "".join(ch for ch in raw if ch.isalpha())
            if not word:
                continue
            if word in resources.fallback_stopwords:
                continue
            tokens.append(word)

    return " ".join(tokens)


def sentence_tokenize(text: str, resources: PreprocessingResources) -> List[str]:
    if resources.nlp is not None:
        doc = resources.nlp(text)
        return [normalize_whitespace(sent.text) for sent in doc.sents if sent.text.strip()]
    return [normalize_whitespace(s) for s in text.split(".") if s.strip()]


def build_tfidf_model(
    documents: List[str],
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.9,
) -> Tuple[TfidfVectorizer, np.ndarray]:
    n_docs = len(documents)
    # For very small corpora, relax min_df/max_df to avoid sklearn errors.
    if n_docs <= 3:
        min_df = 1
        max_df = 1.0

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix


def fit_topic_model(
    tfidf_matrix: np.ndarray, n_topics: int = 5, random_state: int = 42
) -> NMF:
    n_topics = max(2, min(n_topics, tfidf_matrix.shape[0])) if tfidf_matrix.shape[0] > 1 else 1
    model = NMF(
        n_components=n_topics,
        random_state=random_state,
        init="nndsvda",
        max_iter=400,
    )
    model.fit(tfidf_matrix)
    return model


def describe_topics(
    model: NMF, vectorizer: TfidfVectorizer, top_n: int = 10
) -> List[Dict[str, Any]]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    topics: List[Dict[str, Any]] = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[::-1][:top_n]
        words = feature_names[top_indices].tolist()
        weights = topic[top_indices].tolist()
        topics.append(
            {
                "topic_id": topic_idx,
                "top_terms": words,
                "weights": weights,
            }
        )
    return topics


def cluster_documents(
    tfidf_matrix: np.ndarray, n_clusters: int = 5, random_state: int = 42
) -> Tuple[np.ndarray, Optional[float]]:
    n_docs = tfidf_matrix.shape[0]
    n_clusters = max(2, min(n_clusters, n_docs)) if n_docs > 2 else min(n_docs, 2)
    if n_docs < 2:
        labels = np.zeros(n_docs, dtype=int)
        return labels, None

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(tfidf_matrix)

    score: Optional[float]
    try:
        score = silhouette_score(tfidf_matrix, labels) if n_clusters > 1 else None
    except Exception:
        score = None

    return labels, score


def cluster_top_terms(
    tfidf_matrix: np.ndarray,
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    top_n: int = 10,
) -> Dict[int, List[str]]:
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_terms: Dict[int, List[str]] = {}
    for cluster_id in np.unique(labels):
        cluster_docs = tfidf_matrix[labels == cluster_id]
        if cluster_docs.shape[0] == 0:
            cluster_terms[cluster_id] = []
            continue
        mean_tfidf = np.asarray(cluster_docs.mean(axis=0)).ravel()
        top_indices = mean_tfidf.argsort()[::-1][:top_n]
        cluster_terms[cluster_id] = feature_names[top_indices].tolist()
    return cluster_terms


def build_sentence_tfidf_model(
    sentences: List[str], max_features: int = 3000
) -> Tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(sentences)
    return vectorizer, matrix


def extractive_summary(
    text: str,
    resources: PreprocessingResources,
    max_sentences: int = 5,
) -> List[Tuple[str, float]]:
    sentences = sentence_tokenize(text, resources)
    if not sentences:
        return []

    vectorizer, matrix = build_sentence_tfidf_model(sentences)
    scores = np.asarray(matrix.sum(axis=1)).ravel()
    ranked_indices = scores.argsort()[::-1][:max_sentences]

    summary = [(sentences[i], float(scores[i])) for i in ranked_indices]
    summary.sort(key=lambda x: sentences.index(x[0]))
    return summary


def analyze_corpus(
    raw_documents: List[str],
    n_topics: int = 5,
    n_clusters: int = 5,
    summary_sentences: int = 5,
) -> Dict[str, Any]:
    resources = build_preprocessing_resources()

    preprocessed_docs = [preprocess_document(doc, resources) for doc in raw_documents]
    vectorizer, tfidf_matrix = build_tfidf_model(preprocessed_docs)

    topic_model = fit_topic_model(tfidf_matrix, n_topics=n_topics)
    topics = describe_topics(topic_model, vectorizer)

    cluster_labels, silhouette = cluster_documents(tfidf_matrix, n_clusters=n_clusters)
    cluster_terms = cluster_top_terms(tfidf_matrix, cluster_labels, vectorizer)

    summaries = [extractive_summary(doc, resources, max_sentences=summary_sentences) for doc in raw_documents]

    return {
        "preprocessed_docs": preprocessed_docs,
        "tfidf_vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "topics": topics,
        "cluster_labels": cluster_labels,
        "cluster_terms": cluster_terms,
        "silhouette_score": silhouette,
        "summaries": summaries,
    }

