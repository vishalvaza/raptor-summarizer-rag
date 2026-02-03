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
    def __init__(self, api_key, model="gpt-4o-mini"):
        os.environ["OPENAI_API_KEY"] = api_key
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    def _cluster_and_summarize(self, texts):
        vectors = self.embeddings.embed_documents(texts)
        num_clusters = max(1, len(texts) // 3)
        gm = GaussianMixture(n_components=num_clusters, random_state=42)
        labels = gm.fit_predict(vectors)

        summaries = []
        prompt = ChatPromptTemplate.from_template(
            "Summarize these related points into a concise, thematic overview:\n\n{context}"
        )
        chain = prompt | self.llm | StrOutputParser()

        for i in range(num_clusters):
            cluster_docs = [texts[j] for j, label in enumerate(labels) if label == i]
            if cluster_docs:
                summary = chain.invoke({"context": "\n".join(cluster_docs)})
                summaries.append(summary)
        return summaries

    def build_tree(self, chunks, max_layers=3):
        print(f"[*] Building tree for {len(chunks)} chunks...")
        all_docs = [Document(page_content=c, metadata={"layer": 0}) for c in chunks]
        current_layer_texts = chunks
        for layer in range(1, max_layers + 1):
            if len(current_layer_texts) <= 1: break
            print(f"[*] Generating Layer {layer} summaries...")
            summaries = self._cluster_and_summarize(current_layer_texts)
            all_docs.extend([Document(page_content=s, metadata={"layer": layer}) for s in summaries])
            current_layer_texts = summaries
        
        self.vectorstore = Chroma.from_documents(all_docs, self.embeddings)
        print("[+] Knowledge tree indexed successfully.")

    def query(self, question):
        if not self.vectorstore: return "Engine not initialized."
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, 
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5})
        )
        return qa.invoke(question)["result"]
