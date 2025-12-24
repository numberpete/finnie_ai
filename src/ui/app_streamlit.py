import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from langsmith import uuid7
import asyncio
import os
import warnings
from langgraph.checkpoint.memory import InMemorySaver

# --- AGENT IMPORTS ---
from src.agents.router import RouterAgent
from src.agents.response import AgentResponse
from src.utils.tracing import setup_tracing

setup_tracing(service_name="finnie-ui")

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = InMemorySaver()
    
warnings.filterwarnings("ignore", category=DeprecationWarning)

CHART_URL = os.getenv("CHART_URL", "http://localhost:8010/chart/")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Finnie AI Financial Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- ASYNC BRIDGE HELPER ---
def run_async(coro):
    """
    Robust helper to bridge Streamlit threads to an async loop.
    Prevents the 'attached to a different loop' error.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# --- INITIALIZE AGENT (runs once per session) ---
#@st.cache_resource
def get_agent():
    """Initialize the agent once and cache it across sessions."""
    # Importing inside the function ensures the loop isn't captured during module load
    return RouterAgent(checkpointer=st.session_state.checkpointer)

# --- INITIALIZE SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid7())

# Initialize individual tab histories
for hist_key in ["chat_history", "market_history", "portfolio_history", "goals_history"]:
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

# --- UI HELPER: RENDER MESSAGES ---
def render_chat_message(message):
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        if role == "user":
            st.markdown(content)
        else:
            # content is an AgentResponse object
            st.markdown(content.message)
            if hasattr(content, "charts") and content.charts:
                for chart in content.charts:
                    st.image(f"{CHART_URL}{chart.filename}", caption=chart.title, width="stretch")

# --- CORE AGENT LOGIC ---
async def call_agent(user_input: str) -> AgentResponse:
    """Wrapper with timeout to call the cached Agent."""
    try:
        return await asyncio.wait_for(
            AGENT.run_query(user_input, st.session_state.session_id),
            timeout=120
        )
    except asyncio.TimeoutError:
        return AgentResponse(agent="System", message="Request timed out.", charts=[])
    except Exception as e:
        return AgentResponse(agent="System", message=f"Execution error: {e}", charts=[])

# --- MAIN APP UI ---
st.title("🤖 Finnie AI Financial Assistant")

# Sidebar for Global Controls
with st.sidebar:
    st.header("Controls")
    if st.button("🗑️ Reset All Sessions", use_container_width=True):
        run_async(AGENT.cleanup()) # Ensure MCP connections close
        st.session_state.chat_history = []
        st.session_state.market_history = []
        st.session_state.portfolio_history = []
        st.session_state.goals_history = []
        st.session_state.session_id = str(uuid7())
        st.rerun()
    st.divider()
    st.caption(f"Thread ID: {st.session_state.session_id}")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📈 Market", "📊 Portfolio", "🎯 Goals"])

def handle_tab_input(input_text, history_key, spinner_text="Thinking..."):
    if input_text:
        # Save user input
        AGENT = get_agent()

        st.session_state[history_key].append({"role": "user", "content": input_text})
        
        # Display current interaction
        with st.chat_message("user"):
            st.write(input_text)
        
        with st.chat_message("assistant"):
            with st.spinner(spinner_text):
                # Use the bridge to call the async agent logic
                response = run_async(AGENT.run_query(input_text, st.session_state.session_id))
                st.write(response.message)
                if response.charts:
                    for chart in response.charts:
                        st.image(f"{CHART_URL}{chart.filename}", caption=chart.title)
        
        # Save response
        st.session_state[history_key].append({"role": "assistant", "content": response})

# Chat Tab
with tab1:
    for msg in st.session_state.chat_history:
        render_chat_message(msg)
    
    chat_input = st.chat_input("Ask a general financial question...")
    handle_tab_input(chat_input, "chat_history")

# Market Tab
with tab2:
    for msg in st.session_state.market_history:
        render_chat_message(msg)
    
    m_input = st.chat_input("Ask about market data...", key="m_input")
    handle_tab_input(m_input, "market_history", "Analyzing market...")

# Portfolio Tab
with tab3:
    for msg in st.session_state.portfolio_history:
        render_chat_message(msg)
    
    p_input = st.chat_input("Update your portfolio...", key="p_input")
    handle_tab_input(p_input, "portfolio_history", "Calculating allocations...")

# Goals Tab
with tab4:
    for msg in st.session_state.goals_history:
        render_chat_message(msg)
    
    g_input = st.chat_input("Simulate your financial future...", key="g_input")
    handle_tab_input(g_input, "goals_history", "Running simulations...")

st.divider()
st.caption("FinnieAI can make mistakes, and answers are for educational purposes only.")