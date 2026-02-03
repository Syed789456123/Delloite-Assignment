# ShopEase Agentic Data Analyst

## 1. Problem Understanding
ShopEase, a mid-size e-commerce platform, faces increasing customer churn despite strong acquisition. The goal is to identify drivers of churn (suspected: delivery experience, discount dependency) and recommend actionable strategies.

This repository contains an **Agentic AI System** designed to:
1.  **Ingest Data**: Combined structured (CSV) and unstructured (Business Brief) data.
2.  **Reason**: Use an orchestrator to select the right analysis tool based on the user's query.
3.  **Execute**: Run Python-based data analysis.
4.  **Synthesize**: Combine context and results into recommendations.

## 2. Architecture
The system follows a modular agent pattern:

- **`src/agent.py`**: The Central Orchestrator. It receives queries and creates an execution plan.
- **`src/rag.py`**: The Context Provider. A simulated RAG (Retrieval Augmented Generation) module that searches the Business Brief for relevant context.
- **`src/tools.py`**: The Execution Layer. Contains analytical functions (EDA, Stats, ML) that the agent calls.
- **`data/`**: Raw CSV files (Customers, Orders, Engagement, Labels).

## 3. Agent Design Flow
1.  **User Query** -> `Agent`
2.  `Agent` -> `RAG` -> "Retrieve relevant business context"
3.  `Agent` -> **Planner** -> "Decide: Do I need to analyze delivery, channels, or train a model?"
4.  `Agent` -> `Tools` -> "Run the selected Python function"
5.  `Agent` -> **Synthesizer** -> "Combine Context + Data Result + Recommendation" -> **Output**

## 4. Assumptions & Limitations
- **Assumption**: The provided data represents a representative sample of the customer base.
- **Assumption**: "Agentic" reasoning is simulated via keyword-based routing for the purpose of this deterministic demo (in production, an LLM would make these decisions).
- **Limitation**: The RAG module uses simple keyword matching rather than vector embeddings.
- **Limitation**: The ML model is a baseline Random Forest; hyperparameter tuning is out of scope for this POC.


