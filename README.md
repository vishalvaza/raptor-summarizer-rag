# Raptor-Summarizer-RAG 🦖

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)

An advanced **Retrieval-Augmented Generation (RAG)** system implementing the **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) architecture. This project enables high-performance summarization and semantic query answering for massive datasets by building a hierarchical knowledge tree.

## 🌟 Why RAPTOR?
Standard RAG systems often "lose the forest for the trees" because they only retrieve small, isolated text chunks. **RAPTOR** solves this by:
- **Recursive Summarization**: Building layers of summaries on top of raw data.
- **Thematic Clustering**: Grouping related information using Gaussian Mixture Models (GMM).
- **Global & Local Context**: Allowing the AI to answer broad thematic questions (using top-level summaries) or granular factual questions (using leaf nodes) simultaneously.

---

## 🏗️ Architecture
The system organizes your documents into a tree-like hierarchy:



1. **Leaf Nodes (Layer 0)**: Original document chunks.
2. **Clustering**: Semantic grouping of nodes using GMM and dimensionality reduction.
3. **Summarization**: LLM-generated abstractive summaries for each cluster.
4. **Recursive Indexing**: The process repeats until a global summary is achieved.



---

## 📂 Project Structure
```text
raptor-summarizer-rag/
├── src/
│   └── raptor_rag/
│       ├── engine.py       # Core RAPTOR tree-building & query logic
│       └── utils.py        # PDF extraction & text chunking utilities
├── config/
│   └── settings.yaml       # AI model and tree hyperparameters
├── data/                   # Directory for input PDF documents
├── tests/                  # Automated validation scripts
├── main.py                 # Interactive CLI & validation entry point
└── requirements.txt        # Project dependencies
