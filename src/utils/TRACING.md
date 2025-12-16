# OpenTelemetry Tracing Setup Guide

## Installation

First, install the required packages:

```bash
pip install opentelemetry-api opentelemetry-sdk
pip install opentelemetry-instrumentation-fastapi
pip install opentelemetry-instrumentation-httpx
pip install opentelemetry-instrumentation-requests
```

Or add to your `requirements.txt`:
```
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-instrumentation-fastapi>=0.41b0
opentelemetry-instrumentation-httpx>=0.41b0
opentelemetry-instrumentation-requests>=0.41b0
```

## Usage

### 1. In your MCP Server (`finance_q_and_a_mcp.py`)

```python
from src.utils.tracing import setup_tracing, setup_logger_with_tracing

# Initialize tracing at the top of your file
setup_tracing("mcp-server", enable_console_export=False)

# Use the tracing-enabled logger
logger = setup_logger_with_tracing(__name__)

# Your existing code
@mcp.tool()
async def basic_query(query: str, categories: list = None):
    logger.info(f"basic_query called: query='{query}', categories={categories}")
    # ... your code
    logger.info(f"Found {len(results)} results")
    return results
```

### 2. In your FinanceQandAAgent (`finance_q_and_a.py`)

```python
from src.utils.tracing import setup_tracing, setup_logger_with_tracing, get_tracer

# Initialize tracing
setup_tracing("finance-agent", enable_console_export=False)

# Use the tracing-enabled logger
logger = setup_logger_with_tracing(__name__)

# Get tracer for manual spans
tracer = get_tracer(__name__)

class FinanceQandAAgent:
    def __init__(self):
        logger.info("Initializing FinanceQandAAgent")
        # ... your code
    
    async def run_query(self, history, session_id):
        # Create a span for this query
        with tracer.start_as_current_span("agent_run_query"):
            logger.info(f"Processing query for session {session_id}")
            # ... your code
            logger.info("Query completed")
            return response
```

### 3. In your Supervisor (`supervisor_agent.py`)

```python
from src.utils.tracing import setup_tracing, setup_logger_with_tracing, get_tracer

# Initialize tracing
setup_tracing("supervisor", enable_console_export=False)

# Use the tracing-enabled logger
logger = setup_logger_with_tracing(__name__)
tracer = get_tracer(__name__)

class SupervisorAgent:
    async def run_query(self, user_input: str, session_id: str) -> str:
        # Create a root span for the entire request
        with tracer.start_as_current_span("supervisor_run_query") as span:
            # Add useful attributes to the span
            span.set_attribute("user_input", user_input[:50])
            span.set_attribute("session_id", session_id)
            
            logger.info(f"Processing query: {user_input[:50]}...")
            
            # ... your code
            
            logger.info("Query completed")
            return result
```

### 4. In your Gradio App (`app_chatbot.py`)

```python
from src.utils.tracing import setup_tracing, setup_logger_with_tracing

# Initialize tracing once at startup
setup_tracing("gradio-app", enable_console_export=False)

logger = setup_logger_with_tracing(__name__)

# Your existing code
async def chat_wrapper(user_input, chat_history, session_id):
    logger.info(f"Received query: {user_input[:50]}...")
    # ... your code
```

## What You'll See

With OpenTelemetry enabled, your logs will now include trace IDs:

```
INFO:     [a1b2c3d4:5678abcd] supervisor_agent.py:145 - Processing query: What is an IRA?
INFO:     [a1b2c3d4:9012efgh] finance_q_and_a.py:89 - Agent processing query
INFO:     [a1b2c3d4:ijkl3456] finance_q_and_a_mcp.py:94 - basic_query called: query='IRA'
INFO:     [a1b2c3d4:ijkl3456] finance_q_and_a_mcp.py:103 - Found 5 results
INFO:     [a1b2c3d4:9012efgh] finance_q_and_a.py:125 - Query completed
INFO:     [a1b2c3d4:5678abcd] supervisor_agent.py:158 - Query completed
```

**Key points:**
- First part (`a1b2c3d4`) is the **trace_id** - same across ALL logs for one request
- Second part (`5678abcd`) is the **span_id** - unique for each operation
- The trace_id automatically propagates across HTTP calls!

## Advanced: Console Export for Debugging

To see detailed span information during development, enable console export:

```python
setup_tracing("mcp-server", enable_console_export=True)
```

This will print detailed span information showing the hierarchy of operations.

## Benefits

✅ Automatically traces across process boundaries (HTTP calls)  
✅ No manual ECID passing needed  
✅ Industry-standard distributed tracing  
✅ Can be extended to export to Jaeger, Zipkin, or other tracing backends  
✅ Works with async Python  
✅ Minimal code changes required