# Raptor-Summarizer-RAG 🦖

A high-performance Retrieval-Augmented Generation (RAG) system that implements the **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) architecture. 

Standard RAG struggles with large documents because it only looks at small, isolated chunks. This system solves that by building a "Tree of Knowledge," allowing the AI to understand both the forest and the trees.



## 🌟 Key Features
- **Recursive Summarization**: Automatically generates hierarchical layers of context.
- **Intelligent Clustering**: Uses Gaussian Mixture Models (GMM) to group semantically similar topics before summarizing.
- **Multi-Resolution Retrieval**: Retrieves specific facts (from bottom layers) or global themes (from top layers) dynamically.
- **Context-Aware**: Ideal for long-form documents like books, legal papers, or technical manuals.

## 🏗️ Architecture
1. **Layer 0**: Raw text chunks (Leaf nodes).
2. **Clustering**: Machine learning (GMM) identifies related chunks.
3. **Abstraction**: LLM summarizes clusters to create a higher-level layer.
4. **Recursion**: Step 2 and 3 repeat until a "Root Summary" is created.
5. **Indexing**: All layers are indexed in a Chroma Vector Database for hybrid retrieval.



## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- OpenAI API Key

### Installation
```bash
pip install langchain langchain-openai langchain-community chromadb scikit-learn numpy
