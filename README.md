📚 Research Topic Analysis System



A traditional NLP-based research analysis system for academic documents (e.g., arXiv papers).
This project implements a classical Natural Language Processing and Machine Learning pipeline to analyze research documents without using Large Language Models (LLMs) or agentic AI systems.

🚀 Project Overview

Researchers often need a quick analytical overview of a research domain but face difficulty when reviewing multiple academic papers.

This system automates research document analysis by:

Extracting key terms

Identifying latent topics

Clustering similar documents

Generating extractive summaries

Providing clustering evaluation metrics

All using interpretable statistical NLP techniques.

🧠 Key Features

📄 Upload multiple .txt or .pdf research papers

🔎 TF-IDF based key term extraction

🧩 Topic modeling using Non-negative Matrix Factorization (NMF)

📊 Document clustering using KMeans

✂ Extractive summarization using sentence-level TF-IDF scoring

📈 Silhouette score for clustering evaluation

🌐 Interactive Streamlit web interface

🏗 System Architecture

The system follows a structured NLP pipeline:

Document Input

Research keywords

Uploaded TXT/PDF files

Text Preprocessing

Tokenization

Lowercasing

Stop-word removal

Lemmatization

Sentence segmentation

Feature Extraction

TF-IDF vectorization

Analysis

Topic modeling (NMF)

Document clustering (KMeans)

Summarization

TF-IDF-based sentence ranking

Output

Key terms

Topic clusters

Extractive summaries

Evaluation metrics

📂 Project Structure

research_topic_analysis/
│
├── app.py                # Streamlit user interface
├── nlp_pipeline.py       # Core NLP & ML processing logic
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone <your-repo-link>
cd research_topic_analysis
2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Download NLP Models
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet
▶️ Run the Application
streamlit run app.py

Open the local URL (typically http://localhost:8501) in your browser.

📊 Core Algorithms Used
Component	Technique
Feature Extraction	TF-IDF
Topic Modeling	NMF
Clustering	KMeans
Summarization	Sentence-level TF-IDF scoring
Evaluation	Silhouette Score
⚠️ Limitations

While effective and interpretable, traditional NLP approaches have limitations:

No semantic understanding of context

Ignores word order

Sensitive to preprocessing decisions

Requires manual topic selection

Limited generalization across domains

No autonomous reasoning or external knowledge retrieval

These limitations highlight opportunities for future integration of embedding-based models and intelligent workflows.

🔮 Future Enhancements

Replace sparse TF-IDF with dense semantic embeddings

Add visualization dashboards for topic distributions

Improve summarization using hybrid statistical methods

Integrate intelligent retrieval mechanisms

Deploy publicly on cloud platforms

🛠 Technologies Used

Python

Streamlit

scikit-learn

spaCy

NLTK

📌 License

This project is intended for academic and educational purposes.
