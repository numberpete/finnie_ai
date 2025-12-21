import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt

# --- Configuration ---
CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True, parents=True)

# --- Utility to save chart ---
def save_chart(fig: plt.Figure, chart_id: str) -> Path:
    """
    Save a matplotlib figure to the charts directory.
    Returns the Path to the saved file.
    """
    filepath = CHART_DIR / f"{chart_id}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return filepath

# --- Generate chart dynamically ---
def generate_chart(chart_id: str) -> Path:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [10, 20, 15, 25], marker="o")
    ax.set_title(f"Dynamic Chart: {chart_id}")
    return save_chart(fig, chart_id)

# --- Main Streamlit App ---
st.title("Dynamic Chart Demo")

# Button to regenerate chart
if st.button("Generate New Chart"):
    chart_path = generate_chart("test_chart")
    st.session_state["latest_chart"] = chart_path

# Use previous chart if exists
chart_path = st.session_state.get("latest_chart")
if chart_path is None:
    st.info("Click the button to generate a chart.")
else:
    # Debug / existence check
    if chart_path.exists():
        st.success(f"File exists and is readable: {chart_path}")
    else:
        st.error(f"File does not exist: {chart_path}")

    # Display chart
    st.image(str(chart_path), use_column_width=True)
