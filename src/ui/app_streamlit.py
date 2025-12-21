import streamlit as st
from langsmith import uuid7
import asyncio
import os
import sys
from pathlib import Path
from src.agents.router import RouterAgent
from src.agents.response import AgentResponse, ChartArtifact
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

CHART_URL = os.getenv("CHART_URL", "http://localhost:8010/chart/")

# --- PAGE CONFIG (must be first Streamlit command) ---
st.set_page_config(
    page_title="Finnie AI Financial Assistant",
    page_icon="🤖",
    layout="wide"
)

# --- INITIALIZE AGENT (runs once per session) ---
@st.cache_resource
def get_agent():
    """Initialize the agent once and cache it across sessions"""
    return RouterAgent()

def run_async(coro):
    """Helper to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


AGENT = get_agent()

# --- INITIALIZE SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid7())

# Initialize chat history for each tab
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "market_history" not in st.session_state:
    st.session_state.market_history = []

if "portfolio_history" not in st.session_state:
    st.session_state.portfolio_history = []

if "goals_history" not in st.session_state:
    st.session_state.goals_history = []

# --- HELPER FUNCTION ---
async def get_agent_response(user_input: str, session_id: str) -> AgentResponse:
    """Wrapper to call the async agent"""
    try:
        response = await asyncio.wait_for(
            AGENT.run_query(user_input, session_id),
            timeout=120  # 2 minutes timeout
        )
        return response
    except asyncio.TimeoutError:
        return AgentResponse(
            agent="UI",
            message="The request timed out. Please try again or simplify your query.",
            charts=[]
        )
    except Exception as e:
        return AgentResponse(
            agent="UI",
            message=f"An error occurred during agent execution: {e}",
            charts=[]
        )

# --- MAIN APP ---
st.title("🤖 Finnie AI Financial Assistant")

with st.container():
    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📈 Market", "📊 Portfolio", "🎯 Goals"])
    with tab1:
        st.markdown("#### Ask your financial question (with history maintained)")

        # Display chat history
        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):
                #hmm, this is a bit of a hack to get the user and assistant messages to display correctly
                if message["role"] == "user":
                    st.write(message["content"])
                else :
                    st.write(message["content"].message)
                    chart_slot = st.empty()

                    if getattr(message["content"], "charts", None):
                        for chart in message["content"].charts:
                            st.image(f"{CHART_URL}{chart.filename}", caption=chart.title, width="stretch")

        # Chat input
        if user_input := st.chat_input("e.g., What is an IRA and how does it relate to tax?"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
 
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Run async function in event loop
                    response = run_async(get_agent_response(user_input, st.session_state.session_id))
                    if getattr(response, "charts"):
                        for chart in response.charts:
                            st.image(f"{CHART_URL}{chart.filename}", caption=chart.title, width="stretch")



            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.session_id = str(uuid7())
            st.rerun()
        pass

    with tab2:
        st.markdown("### Market Data")
        st.info("Market data and visualization coming soon...")
        
        # Display market chat history
        for message in st.session_state.market_history:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else :
                    st.write(message["content"].message)
        
        # Market chat input
        if market_input := st.chat_input("Ask about market data...", key="market_input"):
            st.session_state.market_history.append({"role": "user", "content": market_input})
            with st.chat_message("user"):
                st.write(market_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing market data..."):
                    response = asyncio.run(get_agent_response(market_input, st.session_state.session_id))
                    st.write(response.message)
            
            st.session_state.market_history.append({"role": "assistant", "content": response})
            pass

    with tab3:
        st.markdown("### Portfolio Analysis")
        st.info("Portfolio data and visualization coming soon...")
        
        # Display portfolio chat history
        for message in st.session_state.portfolio_history:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else :
                    st.write(message["content"].message)
        
        # Portfolio chat input
        if portfolio_input := st.chat_input("Ask about your portfolio...", key="portfolio_input"):
            st.session_state.portfolio_history.append({"role": "user", "content": portfolio_input})
            with st.chat_message("user"):
                st.write(portfolio_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing portfolio..."):
                    response = asyncio.run(get_agent_response(portfolio_input, st.session_state.session_id))
                    st.write(response.message)
            
            st.session_state.portfolio_history.append({"role": "assistant", "content": response})
            pass

    with tab4:
        st.markdown("### Financial Goals")
        st.info("Goals data and visualization coming soon...")
        
        # Display goals chat history
        for message in st.session_state.goals_history:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.write(message["content"])
                else :
                    st.write(message["content"].message)
        
        # Goals chat input
        if goals_input := st.chat_input("Ask about your financial goals...", key="goals_input"):
            st.session_state.goals_history.append({"role": "user", "content": goals_input})
            with st.chat_message("user"):
                st.write(goals_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Setting goals..."):
                    response = asyncio.run(get_agent_response(goals_input, st.session_state.session_id))
                    st.write(response.message)
            
            st.session_state.goals_history.append({"role": "assistant", "content": response})
            pass
