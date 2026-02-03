import os
import numpy as np
from sklearn.mixture import GaussianMixture
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

class RaptorSummarizerRAG:
    """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
    This system builds a hierarchical tree of summaries for deep document context.
    """
    def __init__(self, api_key, model="gpt-4o-mini"):
        if not api_key or "sk-" not in api_key:
            raise ValueError("A valid OpenAI API Key is required.")
        
        os.environ["OPENAI_API_KEY"] = api_key
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.layers_count = 0

    def _cluster_and_summarize(self, texts):
        """Clusters texts using GMM and returns abstractive summaries."""
        vectors = self.embeddings.embed_documents(texts)
        
        # Determine optimal clusters (rule of thumb: 1 cluster per 3-5 chunks)
        num_clusters = max(1, len(texts) // 3)
        gm = GaussianMixture(n_components=num_clusters, random_state=42)
        labels = gm.fit_predict(vectors)

        summaries = []
        prompt = ChatPromptTemplate.from_template(
            "Extract and summarize the core themes from these related document segments:\n\n{context}"
        )
        chain = prompt | self.llm | StrOutputParser()

        for i in range(num_clusters):
            cluster_docs = [texts[j] for j, label in enumerate(labels) if label == i]
            if cluster_docs:
                summary = chain.invoke({"context": "\n".join(cluster_docs)})
                summaries.append(summary)
        return summaries

    def build_tree(self, chunks, max_layers=3):
        """Builds the hierarchy and stores it in a Vector DB."""
        print(f"--- Building Knowledge Tree ---")
        all_docs = [Document(page_content=c, metadata={"layer": 0}) for c in chunks]
        current_layer_texts = chunks

        for layer in range(1, max_layers + 1):
            if len(current_layer_texts) <= 1:
                break
            
            print(f"Processing Layer {layer} (Summarizing {len(current_layer_texts)} nodes)...")
            summaries = self._cluster_and_summarize(current_layer_texts)
            all_docs.extend([Document(page_content=s, metadata={"layer": layer}) for s in summaries])
            current_layer_texts = summaries
            self.layers_count = layer

        # Initialize Vector Store
        self.vectorstore = Chroma.from_documents(
            documents=all_docs, 
            embedding=self.embeddings
        )
        print(f"✅ Success: Tree built with {len(all_docs)} nodes across {self.layers_count + 1} layers.\n")

    def query(self, question):
        """Performs RAG by searching across all layers of the hierarchy."""
        if not self.vectorstore:
            return "Index not found."
            
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, 
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4})
        )
        return qa.invoke(question)["result"]

# ==========================================
# TEST SUITE & VALIDATION
# ==========================================
def run_validation_test(api_key):
    # Sample Dataset: Information about a fictional deep-sea mission
    test_data = [
        "Mission 'Abyss-2026' launched in Jan 2026 to explore the Mariana Trench.",
        "The primary vessel is the 'Titan-9', a submersible capable of 12,000m depth.",
        "Lead biologist Dr. Sarah Chen is searching for bioluminescent microbes.",
        "Budget for the mission is $45 million, funded by the Global Ocean Institute.",
        "The mission found a new species of jelly-fish at 8,000m with metallic scales.",
        "Extreme pressure at 10,000m caused a minor leak in the external sensor array.",
        "Data indicates that deep-sea temperatures are rising faster than expected.",
        "Future missions plan to establish a permanent robotic outpost by 2030."
    ]

    engine = RaptorSummarizerRAG(api_key=api_key)
    
    # 1. Validate Tree Construction
    engine.build_tree(test_data, max_layers=2)
    
    # 2. Test Specific Retrieval (Leaf Node)
    print("Test 1: Specific Fact")
    ans1 = engine.query("What is the name of the submersible and its depth limit?")
    print(f"Q: Submersible name?\nA: {ans1}\n")

    # 3. Test General/Thematic Retrieval (Summary Node)
    print("Test 2: High-Level Summary")
    ans2 = engine.query("Provide a broad overview of the Abyss-2026 mission objectives and findings.")
    print(f"Q: Mission Overview?\nA: {ans2}\n")

if __name__ == "__main__":
    # Replace with your key to test
    MY_KEY = "sk-YOUR-KEY-HERE" 
    run_validation_test(MY_KEY)
