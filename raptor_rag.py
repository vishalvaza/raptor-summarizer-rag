import os
import numpy as np
from sklearn.mixture import GaussianMixture
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

class RaptorRAG:
    """
    A Large-Scale AI System implementation using RAPTOR:
    Recursive Abstractive Processing for Tree-Organized Retrieval.
    """
    def __init__(self, api_key, model="gpt-4o-mini"):
        os.environ["OPENAI_API_KEY"] = api_key
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def _cluster_and_summarize(self, texts, layer):
        """Groups similar texts and summarizes them to create the next tree layer."""
        # Convert text to vectors
        vectors = self.embeddings.embed_documents(texts)
        
        # Determine clusters (at least 1, or 1 per 3 chunks)
        num_clusters = max(1, len(texts) // 3)
        gm = GaussianMixture(n_components=num_clusters, random_state=42)
        labels = gm.fit_predict(vectors)

        summaries = []
        prompt = ChatPromptTemplate.from_template(
            "Summarize these related points from a larger document into a concise summary "
            "that captures the main themes:\n\n{context}"
        )
        chain = prompt | self.llm | StrOutputParser()

        for i in range(num_clusters):
            cluster_docs = [texts[j] for j, label in enumerate(labels) if label == i]
            if cluster_docs:
                summary = chain.invoke({"context": "\n".join(cluster_docs)})
                summaries.append(summary)
        return summaries

    def build_tree(self, chunks, max_layers=3):
        """Builds the recursive tree and indexes all nodes into ChromaDB."""
        all_docs = [Document(page_content=c, metadata={"layer": 0}) for c in chunks]
        current_layer_texts = chunks

        for layer in range(1, max_layers + 1):
            if len(current_layer_texts) <= 1:
                break
            print(f"Building Tree Layer {layer}...")
            summaries = self._cluster_and_summarize(current_layer_texts, layer)
            all_docs.extend([Document(page_content=s, metadata={"layer": layer}) for s in summaries])
            current_layer_texts = summaries

        # Persistence: This creates a local vector database
        self.vectorstore = Chroma.from_documents(
            documents=all_docs, 
            embedding=self.embeddings,
            collection_name="raptor_collection"
        )
        print(f"Tree built with {len(all_docs)} total nodes. Ready for queries.")

    def answer(self, query):
        """Performs retrieval across all layers of the tree."""
        if not self.vectorstore:
            return "Error: You must build the tree index first."
            
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, 
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5})
        )
        result = qa.invoke(query)
        return result["result"]

# --- SAMPLE EXECUTION ---
if __name__ == "__main__":
    # 1. Configuration
    API_KEY = "sk-your-openai-key-here" # Replace with your actual key
    
    # 2. Sample data representing a long document split into chunks
    sample_data = [
        "Project X started in 2022 with a budget of $5 million focused on solar efficiency.",
        "By 2023, the team discovered a new perovskite structure that boosted output by 12%.",
        "The lead scientist, Dr. Aris, noted that the stability of cells improved significantly.",
        "Market analysis shows a growing demand for solar tech in residential areas.",
        "Current challenges include high manufacturing costs and supply chain delays for glass.",
        "The board has approved an additional $2 million for 2025 to scale production.",
        "A new factory is planned in Arizona to minimize logistics costs.",
        "The goal is to reach 25% efficiency by the end of the decade."
    ]

    # 3. Initialize and Run
    app = RaptorRAG(api_key=API_KEY)
    app.build_tree(sample_data)
    
    print("\n" + "="*50)
    print("QUERY 1: What is the financial history and future of Project X?")
    print("RESPONSE:", app.answer("What is the financial history and future of Project X?"))
    
    print("\nQUERY 2: Who is the lead scientist and what was their main finding?")
    print("RESPONSE:", app.answer("Who is the lead scientist and what was their main finding?"))
    print("="*50)
