"""
Simple RAG (Retrieval Augmented Generation) Knowledge Base
Stores unstructured business context and retrieves relevant sections based on queries.
Wrapped as a LangChain Tool.
"""
from langchain_core.tools import tool

class BusinessKnowledgeBase:
    def __init__(self):
        self.documents = {
            "background": "ShopEase is a mid-size E-commerce Platform in India operating D2C + Marketplace models. Categories: Electronics, Fashion, Home & Kitchen, Beauty.",
            "problem": "Leadership is concerned about increasing customer churn and declining repeat purchase rates, particularly from paid channels.",
            "hypotheses": "Leadership suspects poor delivery experience, discount dependency, and low post-purchase engagement are contributing factors.",
            "constraints": "Marketing budget is capped. Heavy discounting is discouraged. Focus on operational or engagement improvements."
        }

    def retrieve_context(self, query: str) -> str:
        """
        Simulates identifying relevant documents based on a query.
        """
        query = query.lower()
        relevant_docs = []
        
        # Simple keyword matching simulation of Semantic Search
        if "churn" in query or "problem" in query:
            relevant_docs.append(self.documents["problem"])
        if "delivery" in query or "discount" in query or "suspect" in query:
            relevant_docs.append(self.documents["hypotheses"])
        if "budget" in query or "cost" in query:
            relevant_docs.append(self.documents["constraints"])
            
        # Default fallback
        if not relevant_docs:
            relevant_docs.append(self.documents["background"])
            
        return "\n".join(relevant_docs)

# Global singleton
_RAG_INSTANCE = BusinessKnowledgeBase()

@tool
def get_business_context(query: str):
    """
    Retrieves qualitative business context, history, and hypotheses from company documents.
    Use this to understand 'Why' something is happening or what the business suspects.
    """
    return _RAG_INSTANCE.retrieve_context(query)
