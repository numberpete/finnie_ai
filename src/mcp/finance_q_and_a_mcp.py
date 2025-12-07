import os
from pathlib import Path
from typing import List, Optional
from fastmcp import FastMCP
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_tavily import TavilySearch

# Initialize your resources
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VECTOR_STORE_PATH_ARTICLES = PROJECT_ROOT / "src" / "data" / "financial_articles"
VECTOR_STORE_PATH_BOGLEHEADS = PROJECT_ROOT / "src" / "data" / "bogleheads"

# Create MCP server
mcp = FastMCP("Financial Q&A Server")

# Load the FAISS vector store for financial articles
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", max_retries=3)
basic_vector = FAISS.load_local(
    str(VECTOR_STORE_PATH_ARTICLES), 
    embeddings,
    allow_dangerous_deserialization=True
)
#search_tavily = TavilySearch(max_results=3)

advanced_vector = FAISS.load_local(
    str(VECTOR_STORE_PATH_BOGLEHEADS), 
    embeddings,
    allow_dangerous_deserialization=True
)

@mcp.tool()
def list_categories() -> List[str]:
    """
    Tool to retrieve the full list of available categories for the basic financial knowledge base. Use this when the user asks a domain-specific question and you need to select one or more precise category strings (e.g., 'Retirement') to pass to the basic_query tool.
    
    Returns:
        A list of unique category names.
    """
    categories = set()
    categories.add('Investing')  
    categories.add('Financial Planning')  
    categories.add('Retirement Planning')  

    for doc in basic_vector.get_all_documents():
        category = doc.metadata.get('category')
        if category:
            categories.add(category)

    return list(categories)

adv_categories = set()
adv_categories.add('Investing')  
adv_categories.add('Financial Planning')  
adv_categories.add('Retirement Planning')  

@mcp.tool()
def list_advanced_categories() -> List[str]:
    """
    Tool to retrieve the list of available categories for the advanced financial knowledge base. Use this when the user asks a highly technical or advanced question and you are preparing to call the advanced_query tool.
    
    Returns:
        A list of unique category names.
    """
    return list(adv_categories)


@mcp.tool()
def basic_query(query: str, categories: Optional[List[str]] = None) -> str:
    """
    Search financial articles for information about investing, taxes, retirement, etc.
    Also, can search for general definitions of financial terms by supplying category='Glossary
    
    Args:
        query: The search query
        categories: A list of relevant categories to limit the search 
                    (e.g., ['Retirement', 'Tax']). Only one of these must match.    
    Returns:
        Formatted search results with sources
    """
    search_kwargs={
        "k": 3,      # The final number of documents to return
        "fetch_k": 20, # Number of *initial* documents to fetch for re-ranking 
        "lambda_mult": 0.7 # Trade-off between relevance (1.0) and diversity (0.0)
    }

    if categories and len(categories) > 0:
        if len(categories) == 1:
            # Simple Filter: {"primary_category": "Retirement"}
            search_kwargs["filter"] = {"primary_category": categories[0]}
        else:
            # OR Filter: {"$or": [condition1, condition2, ...]}
            or_conditions = [{"primary_category": cat} for cat in categories]
            search_kwargs["filter"] = {"$or": or_conditions}
    
    retriever = basic_vector.as_retriever(
        search_type="mmr", 
        search_kwargs=search_kwargs
    )

    results = retriever.invoke(query)
    
    if not results:
        return "No relevant information found."
    
    formatted_results = []
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get('title', 'Unknown')
        url = doc.metadata.get('url', '')
        category = doc.metadata.get('category', 'General')
        
        result = f"""**Result {i}:**
{doc.page_content}

**Source**: {title}
**URL**: {url}
**Category**: {category}
"""
        formatted_results.append(result)
    
    return "\n---\n".join(formatted_results)

@mcp.tool()
def advanced_query(query: str, categories: Optional[List[str]] = None) -> str:
    """
    Search full Bogleheads' articles for information about investing, financial planning, and retirement planning.
    basic_query should be used first, advanced_query is for more detail than basic_query can provide.
    
    Args:
        query: The search query
        categories: A list of relevant categories to limit the search 
                    (e.g., ['Retirement', 'Tax']). Only one of these must match.    
    Returns:
        Formatted search results with sources
    """
    search_kwargs={
        "k": 3,      # The final number of documents to return
        "fetch_k": 20, # Number of *initial* documents to fetch for re-ranking 
        "lambda_mult": 0.7 # Trade-off between relevance (1.0) and diversity (0.0)
    }

    if categories and len(categories) > 0:
        if len(categories) == 1:
            # Simple Filter: {"primary_category": "Retirement"}
            search_kwargs["filter"] = {"primary_category": categories[0]}
        else:
            # OR Filter: {"$or": [condition1, condition2, ...]}
            or_conditions = [{"primary_category": cat} for cat in categories]
            search_kwargs["filter"] = {"$or": or_conditions}
    
    retriever = advanced_vector.as_retriever(
        search_type="mmr", 
        search_kwargs=search_kwargs
    )

    results = retriever.invoke(query)
    
    if not results:
        return "No relevant information found."
    
    formatted_results = []
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get('title', 'Unknown')
        url = doc.metadata.get('url', '')
        category = doc.metadata.get('category', 'General')
        
        result = f"""**Result {i}:**
{doc.page_content}

**Source**: {title}
**URL**: {url}
**Category**: {category}
"""
        formatted_results.append(result)
    
    return "\n---\n".join(formatted_results)

if __name__ == "__main__":
    mcp.run()

