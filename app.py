import streamlit as st
from typing import List

from nlp_pipeline import (
    read_pdf,
    read_text_file,
    analyze_corpus,
)


st.set_page_config(
    page_title="Research Topic Analysis – Milestone 1",
    layout="wide",
)


def load_uploaded_documents(files) -> List[str]:
    documents: List[str] = []
    for uploaded in files:
        content = uploaded.read()
        name = uploaded.name.lower()
        if name.endswith(".pdf"):
            text = read_pdf(content)
        else:
            text = read_text_file(content)
        text = text.strip()
        if text:
            documents.append(text)
    return documents


def main() -> None:
    st.title("Intelligent Research Topic Analysis – Milestone 1")
    st.markdown(
        "Traditional NLP-based pipeline for analyzing research documents (no LLMs or agents)."
    )

    with st.sidebar:
        st.header("Configuration")
        topic_keywords = st.text_input(
            "Research topic keywords",
            placeholder="e.g., graph neural networks, federated learning",
        )
        n_topics = st.slider("Number of topics (NMF)", min_value=2, max_value=10, value=5)
        n_clusters = st.slider("Number of clusters (KMeans)", min_value=2, max_value=10, value=5)
        summary_len = st.slider(
            "Summary sentences per document", min_value=2, max_value=10, value=5
        )

        st.markdown("---")
        st.caption("Upload PDF or TXT research papers.")
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=["pdf", "txt"],
        )

        analyze_button = st.button("Run Analysis")

    if not analyze_button:
        st.info("Enter topic keywords, upload some documents, and click **Run Analysis**.")
        return

    if not uploaded_files:
        st.error("Please upload at least one document.")
        return

    with st.spinner("Loading and preprocessing documents..."):
        docs = load_uploaded_documents(uploaded_files)

    if not docs:
        st.error("No readable text content found in the uploaded files.")
        return

    st.success(f"Loaded {len(docs)} documents.")

    with st.spinner("Running topic modeling, clustering, and summarization..."):
        results = analyze_corpus(
            docs,
            n_topics=n_topics,
            n_clusters=n_clusters,
            summary_sentences=summary_len,
        )

    topics = results["topics"]
    cluster_labels = results["cluster_labels"]
    cluster_terms = results["cluster_terms"]
    silhouette = results["silhouette_score"]
    summaries = results["summaries"]

    col_topics, col_clusters = st.columns(2)

    with col_topics:
        st.subheader("Discovered Topics (NMF)")
        if topics:
            for topic in topics:
                terms = ", ".join(topic["top_terms"])
                st.markdown(f"**Topic {topic['topic_id']}**: {terms}")
        else:
            st.write("No topics could be extracted.")

    with col_clusters:
        st.subheader("Document Clusters (KMeans)")
        if silhouette is not None:
            st.caption(f"Silhouette score (higher is better): **{silhouette:.3f}**")
        else:
            st.caption("Silhouette score not available (too few documents or degenerate clusters).")

        for cluster_id, terms in cluster_terms.items():
            label = f"Cluster {cluster_id}"
            st.markdown(f"**{label}** – top terms: {', '.join(terms) if terms else 'N/A'}")

    st.markdown("---")
    st.subheader("Per-Document Analysis & Extractive Summaries")

    for idx, (doc, summary) in enumerate(zip(docs, summaries)):
        label = cluster_labels[idx] if len(cluster_labels) > idx else "N/A"
        with st.expander(f"Document {idx + 1} (Cluster {label})"):
            st.markdown("**Extractive summary:**")
            if summary:
                for sent, score in summary:
                    st.markdown(f"- {sent}  \n  <sub>score: {score:.4f}</sub>")
            else:
                st.write("No summary could be generated.")

            if topic_keywords:
                st.markdown("---")
                st.caption("Original content (truncated, filtered by keyword if provided).")
                lower_doc = doc.lower()
                if any(kw.strip().lower() in lower_doc for kw in topic_keywords.split(",")):
                    preview = (doc[:2000] + "...") if len(doc) > 2000 else doc
                    st.text(preview)
                else:
                    st.text("Document does not strongly match provided keywords; showing first 1000 chars.")
                    preview = (doc[:1000] + "...") if len(doc) > 1000 else doc
                    st.text(preview)


if __name__ == "__main__":
    main()

