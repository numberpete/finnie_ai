#Finnie AI: Financial Analysis AI Assistant Capstone Project for IK

**Author:** Peter Hanus  
**Date:** 2025-12-08  
**Version:** 1.2  

---

## 1. Overview

The Financial Analysis AI Assistant (FinnieAI) is a multi-agent AI system designed to provide comprehensive financial insights and assistance. It integrates multiple specialized sub-agents, a knowledge base, and external data sources to offer:

- **Q&A** over curated financial knowledge  
- **Portfolio analysis** and optimization  
- **Market analysis** via live financial APIs  
- **Goal planning** and scenario simulation  
- **Data visualization** for analysis outputs  

The system leverages **FAISS-based semantic search**, **OpenAI embeddings**, **graphing via matplotlib**, and a **Model Context Protocol (MCP)** for structured data retrieval and computation.

---

## 2. System Architecture

### 2.1 High-Level Architecture

![FinnieAI Architecture](./A_flowchart_illustrates_the_architecture_of_a_Fina.png)

**Explanation of Flow:**

1. User submits a query.  
2. **General Agent** routes the query to the relevant sub-agent(s).  
3. Sub-agents call **MCP functions** to retrieve structured data or perform calculations.  
4. **Sub-agents generate visualizations** (matplotlib charts) using MCP-provided data.  
5. Aggregated results (text + charts) are returned to the **user interface**.

---

### 2.2 Agent Responsibilities

| Agent                     | Responsibilities                                                                 |
|----------------------------|-------------------------------------------------------------------------------|
| **General Agent**          | Entry point for all user requests. Routes queries to specialized agents.      |
| **Q&A Agent**              | Responds to knowledge queries via FAISS-based semantic search.                |
| **Portfolio Analysis Agent** | Performs portfolio evaluation, risk analysis, optimization, and generates visualizations (matplotlib) from MCP-provided data. |
| **Market Analysis Agent**  | Fetches real-time and historical market data via MCP, produces trend charts.  |
| **Goal Planning Agent**    | Simulates scenarios for financial goal planning using MCP data and produces probability/goal charts. |

---

### 2.3 Data Flow

1. User sends a query to the General Agent.  
2. General Agent determines which sub-agent(s) should handle the request.  
3. Sub-agent uses MCP functions to:
   - Query FAISS indexes (Q&A Agent)  
   - Call financial APIs (Market Analysis Agent)  
   - Retrieve portfolio data and run simulations (Portfolio/Goal Planning Agents)  
4. **Sub-agent generates visualizations using matplotlib** based on MCP-provided data.  
5. Sub-agent returns structured results and optional charts to the General Agent.  
6. General Agent aggregates responses and presents them to the user.

---

## 3. Technical Components

### 3.1 FAISS-based Knowledge Retrieval

- **Purpose:** Answer general financial questions using pre-embedded Bogleheads articles and other resources.  
- **Embedding Model:** OpenAI embedding (`text-embedding-3-small` or `text-embedding-3-large`) depending on desired precision vs cost.  
- **Index:** FAISS index storing vectorized articles/documents.  
- **MCP Function:**  
```python
query_faiss_index(query: str) -> List[Answer]

### 3.2 Model Context Protocol (MCP)
Expose tools as MCP Server functions such that they can be called independent of FinnieAI.

