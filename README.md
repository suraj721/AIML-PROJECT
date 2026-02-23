# Research Topic Analysis System – Milestone 1

Traditional NLP-based research analysis for academic documents (e.g., arXiv papers). This is **Milestone 1** of the “Intelligent Research Topic Analysis and Agentic AI Research Assistant” project, using only classical NLP and ML (no LLMs or agents).

---

## 1. Problem & Use-Case

Researchers often want a **quick analytical view** of a research area:

- **Input**: 
  - Research topic keywords (e.g., *graph neural networks*, *federated learning*).
  - Uploaded research documents (PDF or plain text articles).
- **Output**:
  - Key terms and themes.
  - Topic clusters or categories.
  - Extractive summaries of the uploaded content.
  - Optional analytical visualizations (topic-term distributions).

This system automates a **traditional NLP pipeline** to support literature review and topic exploration *without* LLMs.

---

## 2. Inputs & Outputs

- **Inputs**
  - Text query: research topic keywords.
  - One or more uploaded files:
    - `.txt` files (UTF-8 text).
    - `.pdf` research papers (basic text extraction).

- **Outputs**
  - **Preprocessed text** (tokenized, stop-word removed, lemmatized).
  - **Key terms** via TF-IDF statistics.
  - **Topic modeling** (NMF on TF-IDF) with top words per topic.
  - **Document clusters** (KMeans on TF-IDF) with top terms per cluster.
  - **Extractive summaries**: top-scoring sentences chosen from the documents.
  - **Basic evaluation**:
    - Silhouette score for clustering (interpretability proxy).

---

## 3. System Architecture (Traditional NLP Pipeline)

Conceptual pipeline:

```mermaid
flowchart TD
    A[User Topic Keywords] --> B[Collect & Upload Documents]
    B --> C[Document Loader<br/>(TXT/PDF to Raw Text)]
    C --> D[Preprocessing<br/>(tokenize, lower, stopwords, lemmatize)]
    D --> E[Feature Extraction<br/>(TF-IDF vectors)]
    E --> F[Topic Modeling<br/>(NMF)]
    E --> G[Clustering<br/>(KMeans)]
    D --> H[Sentence Scoring<br/>(TF-IDF-based)]
    F --> I[Key Themes & Topics]
    G --> J[Topic Clusters]
    H --> K[Extractive Summaries]
    I --> L[Streamlit UI]
    J --> L
    K --> L
```

**Implementation components**:

- `nlp_pipeline.py`
  - `load_documents(...)`: handle TXT/PDF uploads.
  - `preprocess_text(...)`: tokenization, stopword removal, lemmatization.
  - `build_tfidf_model(...)`: TF-IDF feature extraction.
  - `fit_topic_model(...)`: NMF topic modeling.
  - `cluster_documents(...)`: KMeans clustering and top terms per cluster.
  - `extractive_summary(...)`: sentence scoring and summary sentence selection.
- `app.py`
  - Streamlit UI to connect all pipeline components.

---

## 4. How to Run Locally

### 4.1. Install Dependencies

From the project directory:

```bash
cd research_topic_analysis_m1
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download SpaCy’s small English model (used for robust lemmatization and sentence splitting). If this fails, the code will fall back to a simpler NLTK-based pipeline:

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet
```

### 4.2. Run the Streamlit App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## 5. File Overview

- `app.py`
  - Streamlit UI for:
    - Entering topic keywords.
    - Uploading multiple documents.
    - Triggering analysis.
    - Viewing key terms, topics, clusters, and summaries.
- `nlp_pipeline.py`
  - Core NLP & ML functions (preprocessing, TF-IDF, topic modeling, clustering, summarization).
- `requirements.txt`
  - Python package dependencies for Milestone 1.
- `README.md`
  - Documentation for Milestone 1.

---

## 6. Short Report: Limitations of Traditional Approaches

**1. Dependence on surface-level statistics**  
TF-IDF and bag-of-words representations ignore **word order** and deeper semantics. Two sentences with different meanings but similar vocabulary may appear close in vector space. This limits the system’s ability to distinguish nuanced arguments or methodological differences across papers.

**2. Vocabulary mismatch and synonymy**  
Traditional models treat words as independent tokens. Synonyms (e.g., “GNNs” vs. “graph neural networks”), abbreviations, and domain-specific terminology fragment the representation. Without semantic embeddings, topics and clusters may split conceptually similar documents just because they use different surface forms.

**3. Sensitivity to preprocessing choices**  
Results are highly sensitive to stop-word lists, lemmatization quality, and tokenization. Domain-specific stop-words (e.g., “theorem”, “proposition” in theory papers) are not automatically removed. Poor preprocessing can lead to noisy topics and unstable clusters.

**4. Topic modeling interpretability and stability**  
Classical topic models like NMF or LDA:
- Require manually choosing the number of topics.
- Can be unstable with small or heterogeneous document sets.
- Often produce topics that are hard to interpret without human inspection.  
Coherence metrics are only approximate and may not align with human judgment for niche research areas.

**5. Limited discourse and document structure awareness**  
Extractive summarization here is based on TF-IDF sentence scoring. It does not model:
- Document sections (abstract, introduction, conclusion).
- Discourse relations (e.g., hypothesis vs. evidence).
- Citation context.  
As a result, summaries may overemphasize high-frequency technical terms while missing explanatory or comparative content important for literature review.

**6. Poor generalization to new styles and domains**  
Since the system relies on sparse TF-IDF vectors and clustering on top of them, it may struggle when:
- The number of documents is very small.
- The writing style or subfield changes (e.g., ML vs. physics papers).  
Unlike modern embedding-based methods, these models do not transfer well across domains without retuning and reinterpreting hyperparameters.

**7. No autonomous reasoning or planning**  
This Milestone 1 system is a **fixed pipeline**, not an agent:
- It does not autonomously search for related work.
- It cannot ask clarification questions or refine the analysis.
- It generates summaries and topics purely from local document statistics.  
This motivates Milestone 2, where LLMs and agentic workflows can provide semantic understanding, interactive exploration, and autonomous research assistance.

---

## 7. Next Steps Toward Milestone 2

- Replace sparse TF-IDF features with dense semantic embeddings (e.g., open-source sentence transformers).
- Use an agent framework (e.g., LangGraph) to:
  - Orchestrate web search, document retrieval, and iterative analysis.
  - Generate structured, human-readable reports.
- Host the final application on Hugging Face Spaces or Streamlit Community Cloud (non-local deployment).

