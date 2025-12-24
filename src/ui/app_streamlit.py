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

# Third-party menu
from streamlit_option_menu import option_menu

# --- SESSION CHECKPOINTER ---
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()

warnings.filterwarnings("ignore", category=DeprecationWarning)
CHART_URL = os.getenv("CHART_URL", "http://localhost:8010/chart/")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Finnie AI Financial Assistant",
                   page_icon="🤖", layout="wide")

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

# --- LEFT NAV MENU ---
menu_options = ["Chat", "Market", "Portfolio", "Goals"]
menu_icons = ["chat", "graph-up", "pie-chart", "bullseye"]

with st.sidebar:
    st.sidebar.header("🤖 Finnie AI")
    
    selected_tab = option_menu(
        menu_title=None,
        menu_icon=None,
        options=menu_options,
        icons=menu_icons,
        default_index=0,
        orientation="vertical"
    )

    st.sidebar.divider()
    if st.sidebar.button("Clear Session", use_container_width=True):
        AGENT = get_agent()
        run_async(AGENT.cleanup())
        for key in ["chat_history", "market_history", "portfolio_history", "goals_history"]:
            st.session_state[key] = []
        st.session_state.session_id = str(uuid7())
        st.session_state._rerun = not st.session_state.get("_rerun", False)

current_tab = selected_tab
st.session_state.selected_tab = current_tab

# --- HELPER TO RENDER RESPONSE ---
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

# --- RENDER PREVIOUS HISTORY ---
history_map = {
    "Chat": st.session_state.chat_history,
    "Market": st.session_state.market_history,
    "Portfolio": st.session_state.portfolio_history,
    "Goals": st.session_state.goals_history
}

if current_tab in history_map:
    for entry in history_map[current_tab]:
        if current_tab == "Chat":
            role = entry["role"]
            content = entry["content"]
            with st.chat_message(role):
                if role == "user":
                    st.markdown(content)
                else:
                    render_response(content)
        else:
            render_response(entry)

# --- STICKY INPUT + DISCLAIMER ---
st.markdown(
    """
    <style>
    div[data-baseweb="input"] > div > input {
        position: fixed !important;
        bottom: 40px !important;
        width: 95% !important;
        z-index: 1000 !important;
        background-color: white;
    }
    .stCaption {
        position: fixed;
        bottom: 5px !important;
        width: 95% !important;
        z-index: 1000 !important;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

user_input = st.chat_input("Ask a financial question...")
st.markdown(
    """
    <div style="
        position: fixed;
        bottom: 0px;
        width: 95%;
        background-color: white;
        padding: 10px 0;
        z-index: 1000;
    ">
        <div style='font-size:12px; color:gray; margin-top:4px;'>
            FinnieAI can make mistakes, and answers are for educational purposes only.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- PROCESS USER INPUT ---
if user_input:
    AGENT = get_agent()
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Render user immediately in Chat
    if current_tab == "Chat":
        with st.chat_message("user"):
            st.markdown(user_input)

    # Agent response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_async(AGENT.run_query(user_input, st.session_state.session_id))
        render_response(response)

    # Append to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Append to agent-specific tab histories
    if response.agent == "FinanceMarketAgent":
        st.session_state.market_history.append(response)
    elif response.agent == "PortfolioAgent":
        st.session_state.portfolio_history.append(response)
    elif response.agent == "GoalsAgent":
        st.session_state.goals_history.append(response)
