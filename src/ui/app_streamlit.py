import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from langsmith import uuid7
import asyncio
import os
import warnings
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.router import RouterAgent
from src.agents.response import AgentResponse

# --- SESSION CHECKPOINTER ---
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()

warnings.filterwarnings("ignore", category=DeprecationWarning)
CHART_URL = os.getenv("CHART_URL", "http://localhost:8010/chart/")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Finnie AI Financial Assistant",
                   page_icon="🤖", layout="wide")

# --- CSS TO HIDE EXPANDERS WHEN PRINTING ---
st.markdown(
    """
    <style>
    @media print {
        .stExpander, .stAppHeader, .st-key-clear_session, .st-key-print_pdf, [role="tablist"], .stChatInput {
            display: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- ASYNC HELPER ---
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# --- AGENT ---
def get_agent():
    return RouterAgent(checkpointer=st.session_state.checkpointer)

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid7())

for key in ["chat_history", "market_history", "portfolio_history", "goals_history"]:
    if key not in st.session_state:
        st.session_state[key] = []

# --- MAIN UI ---
st.title("🤖 Finnie AI Financial Assistant")

# --- CLEAR SESSION BUTTON ---
col1, col2, col3 = st.columns([5, 1, 1])
with col3:
    if st.button("🗑️ Clear Session", key="clear_session"):
        AGENT = get_agent()
        run_async(AGENT.cleanup())
        for key in ["chat_history", "market_history", "portfolio_history", "goals_history"]:
            st.session_state[key] = []
        st.session_state.session_id = str(uuid7())
        st.rerun()

# --- CALCULATE COUNTS FOR BADGES ---
chat_count = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
market_count = len(st.session_state.market_history)
portfolio_count = len(st.session_state.portfolio_history)
goals_count = len(st.session_state.goals_history)

# Show counts only for non-chat tabs
tab_labels = [
    "💬 Chat",
    f"📈 Market" + (f" ({market_count})" if market_count > 0 else ""),
    f"💼 Portfolio" + (f" ({portfolio_count})" if portfolio_count > 0 else ""),
    f"🎯 Goals" + (f" ({goals_count})" if goals_count > 0 else "")
]

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

# --- HELPER TO RENDER RESPONSE (No controls - for Chat tab) ---
def render_response(resp: AgentResponse):
    """Render AgentResponse with two-column layout if charts exist."""
    if hasattr(resp, "charts") and resp.charts:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(f"**{resp.agent}:** {resp.message}")
        with col2:
            for chart in resp.charts:
                st.image(f"{CHART_URL}{chart.filename}", caption=chart.title)
    else:
        st.markdown(f"**{resp.agent}:** {resp.message}")

# --- HELPER TO RENDER RESPONSE WITH CONTROLS (For agent tabs) ---
def render_response_with_controls(resp: AgentResponse, index: int, history_key: str):
    """Render AgentResponse with delete and reorder controls in an expander."""
    
    # Render the response content first
    if hasattr(resp, "charts") and resp.charts:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(f"**{resp.agent}:** {resp.message}")
        with col2:
            for chart in resp.charts:
                st.image(f"{CHART_URL}{chart.filename}", caption=chart.title)
    else:
        st.markdown(f"**{resp.agent}:** {resp.message}")
    
    # Controls in a collapsed expander
    with st.expander("⚙️ Actions", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Delete", key=f"delete_{history_key}_{index}", use_container_width=True):
                st.session_state[history_key].pop(index)
                st.rerun()
        
        with col2:
            if index > 0:
                if st.button("⬆️ Move Up", key=f"up_{history_key}_{index}", use_container_width=True):
                    st.session_state[history_key][index], st.session_state[history_key][index-1] = \
                        st.session_state[history_key][index-1], st.session_state[history_key][index]
                    st.rerun()
            else:
                st.button("⬆️ Move Up", key=f"up_disabled_{history_key}_{index}", disabled=True, use_container_width=True)
        
        with col3:
            if index < len(st.session_state[history_key]) - 1:
                if st.button("⬇️ Move Down", key=f"down_{history_key}_{index}", use_container_width=True):
                    st.session_state[history_key][index], st.session_state[history_key][index+1] = \
                        st.session_state[history_key][index+1], st.session_state[history_key][index]
                    st.rerun()
            else:
                st.button("⬇️ Move Down", key=f"down_disabled_{history_key}_{index}", disabled=True, use_container_width=True)

# --- TAB 1: CHAT (No controls) ---
with tab1:
    for entry in st.session_state.chat_history:
        role = entry["role"]
        content = entry["content"]
        with st.chat_message(role):
            if role == "user":
                st.markdown(content)
            else:
                render_response(content)

# --- TAB 2: MARKET (With controls) ---
with tab2:
    st.subheader("📈 Market Analysis History")
    if st.session_state.market_history:
        for i, entry in enumerate(st.session_state.market_history):
            render_response_with_controls(entry, i, "market_history")
            st.divider()
    else:
        st.info("No market analysis history yet. Ask a market-related question in the chat!")

# --- TAB 3: PORTFOLIO (With controls) ---
with tab3:
    st.subheader("💼 Portfolio Management History")
    if st.session_state.portfolio_history:
        for i, entry in enumerate(st.session_state.portfolio_history):
            render_response_with_controls(entry, i, "portfolio_history")
            st.divider()
    else:
        st.info("No portfolio history yet. Start building your portfolio in the chat!")

# --- TAB 4: GOALS (With controls) ---
with tab4:
    st.subheader("🎯 Financial Goals & Simulations")
    if st.session_state.goals_history:
        for i, entry in enumerate(st.session_state.goals_history):
            render_response_with_controls(entry, i, "goals_history")
            st.divider()
    else:
        st.info("No goals analysis yet. Ask about future projections in the chat!")

# --- CHAT INPUT (Always visible at bottom) ---
user_input = st.chat_input("Ask a financial question...")

# --- DISCLAIMER ---
st.caption("FinnieAI can make mistakes, and answers are for educational purposes only.")

# --- PROCESS USER INPUT ---
if user_input:
    AGENT = get_agent()
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.rerun()

# --- HANDLE PENDING RESPONSE ---
if (st.session_state.chat_history and 
    st.session_state.chat_history[-1]["role"] == "user"):
    
    pending_question = st.session_state.chat_history[-1]["content"]
    
    with st.spinner("Thinking..."):
        AGENT = get_agent()
        response = run_async(AGENT.run_query(pending_question, st.session_state.session_id))
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    if response.agent == "FinanceMarketAgent":
        st.session_state.market_history.append(response)
    elif response.agent == "PortfolioAgent":
        st.session_state.portfolio_history.append(response)
    elif response.agent == "GoalsAgent":
        st.session_state.goals_history.append(response)

    st.rerun()