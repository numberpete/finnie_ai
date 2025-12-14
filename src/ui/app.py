import gradio as gr
from langsmith import uuid7 # Used for session ID
import asyncio # Used to run the async agent function
import os
import sys
from pathlib import Path


# --- FIX START ---
# 1. Get the absolute path of the current script's directory (src/ui)
current_dir = Path(__file__).resolve().parent 
# 2. Get the path to the parent directory (src)
root_src_dir = current_dir.parent 
# 3. Add the 'src' directory to Python's search path
sys.path.insert(0, str(root_src_dir))
# --- FIX END ---


# Now the import should work, because 'src' is on the path, 
# and 'agents' is a sub-module of 'src'.
# You may need to change the import to use the top-level package name:
from agents.finance_q_and_a import FinanceQandAAgent


# --- GRADIO SETUP ---

# 1. Initialize the Agent (This only runs once when the app starts)
# This is equivalent to Streamlit's @st.cache_resource for long-lived objects.
# We run this outside the main Gradio function to keep it instantiated.
AGENT = FinanceQandAAgent()


# 2. Gradio-facing wrapper function
async def chat_wrapper(user_input: str, session_id: str):
    """Gradio calls this function; it wraps the async agent call."""
    if not user_input:
        return "Please enter a question."
    
    # Delegate the work to the decoupled agent object
    try:
        # Use asyncio.wait_for to add a timeout for safety
        response = await asyncio.wait_for(
            AGENT.run_query(user_input, session_id), 
            timeout=120  # 2 minutes timeout
        )
        return response
    except asyncio.TimeoutError:
        return "The request timed out. Please try again or simplify your query."
    except Exception as e:
        return f"An error occurred during agent execution: {e}"


# --- GRADIO INTERFACE ---

with gr.Blocks() as app:
    gr.Markdown("# 🤖 Finnie AI Financial Assistant")
    
    # We use a state component to hold the unique session ID
    session_state = gr.State(value=str(uuid7()))

    with gr.Tab("💬 Chat"):
        gr.Markdown("#### Ask your financial question (with history maintained)")
        
        with gr.Row():
            input_box = gr.Textbox(
                label="Enter your query:", 
                placeholder="e.g., What is an IRA and how does it relate to tax?",
                scale=4
            )
            submit_button = gr.Button("Submit", scale=1)
        
        output_box = gr.Textbox(label="Response:", lines=10, interactive=False)

        # The submission links the UI components to the wrapper function
        submit_button.click(
            chat_wrapper, 
            inputs=[input_box, session_state], 
            outputs=output_box
        )
        input_box.submit(
            chat_wrapper, 
            inputs=[input_box, session_state], 
            outputs=output_box
        )

    # Placeholder for other tabs (Portfolio, Market, Goals)
    with gr.Tab("📈 Portfolio"):
        gr.Markdown("Portfolio data and visualization coming soon...")
    
    # Add other tabs...


# Launch the Gradio app
# Note: Since the agent uses aiohttp internally, running in the main thread is fine.
app.launch(debug=True, share=True, server_name="0.0.0.0", server_port=7860)