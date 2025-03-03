from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_together import Together
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from config import FAISS_INDEX_PATH

class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.processed_hashes = set()
        
        if os.path.exists(FAISS_INDEX_PATH):
            self.vector_store = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings, allow_dangerous_deserialization=True)
            for doc in self.vector_store.docstore._dict.values():
                self.processed_hashes.add(doc.metadata.get("content_hash", ""))
        else:
            self.vector_store = None
        
        self.llm = Together(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            together_api_key="2357e751ff419dec84a6fab0e72dc3e2d877eaa47ac271d2506a43173d86e499",  # Replace with your key
            max_tokens=512,
            temperature=0.7,
            top_p=0.9
        )

    def has_content(self, content_hash):
        return content_hash in self.processed_hashes

    def add_documents(self, documents, content_hash):
        if content_hash in self.processed_hashes:
            print(f"Content hash {content_hash} already indexed, skipping.")
            return
        print(f"Indexing {len(documents)} documents...")
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            print("New FAISS index created.")
        else:
            self.vector_store.add_documents(documents)
            print("Added to existing FAISS index.")
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        self.vector_store.save_local(FAISS_INDEX_PATH)
        print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
        self.processed_hashes.add(content_hash)

    def rewrite_query(self, question):
        if "important points" in question.lower():
            return "Summarize the key sections of the document."
        elif "what is" in question.lower() and "section" in question.lower():
            section_part = question.split("section")[1].strip().split()[0]
            return f"Describe the content of Section {section_part}."
        return question

    def query(self, question):
        if self.vector_store is None:
            return "Please upload a PDF first."
        
        rewritten_question = self.rewrite_query(question)
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, say so—don’t guess.

        Context: {context}

        Question: {rewritten_question}

        Answer:"""
        prompt = PromptTemplate.from_template(template)
        
        docs_with_scores = self.vector_store.similarity_search_with_score(rewritten_question, k=10)
        filtered_docs = [doc for doc, score in docs_with_scores if score > 0.7] or [doc for doc, _ in docs_with_scores[:5]]
        context = "\n".join([doc.page_content for doc in filtered_docs])
        
        chain = (
            {"context": lambda x: context, "rewritten_question": lambda x: rewritten_question}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({})