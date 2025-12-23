# src/utils/tracing.py

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import logging
import time

# Import your logging setup
from .logging import ColoredFormatter

logger = logging.getLogger(__name__)


class TracingFormatter(ColoredFormatter):
    """Extends ColoredFormatter to include OpenTelemetry trace and span IDs."""
    
    # Additional colors
    DARK_GREEN = '\033[32m'
    DARKER_GREEN = '\033[2;32m'  # ✅ Dimmed/darker green
    BLUE = '\033[34m'
    
    def __init__(self, *args, service_name: str = "unknown", **kwargs):
        """
        Initialize formatter with service name.
        
        Args:
            service_name: Name of the service for logging
        """
        super().__init__(*args, **kwargs)
        self.service_name = service_name
    
    def format(self, record):
        # Get current span context
        span = trace.get_current_span()
        span_context = span.get_span_context()
        
        if span_context.is_valid:
            # Format trace_id and span_id as short hex strings (first 8 chars)
            trace_id = format(span_context.trace_id, '032x')[:8]
            span_id = format(span_context.span_id, '016x')[:8]
            record.trace_id = f"[{trace_id}:{span_id}]"
        else:
            record.trace_id = ""
        
        # Add colored service name to record
        record.service_name = f"{self.BLUE}{self.service_name}{self.RESET}"
        
        # Add color to level name
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        return super(ColoredFormatter, self).format(record)
    
    def formatTime(self, record, datefmt=None):
        """
        Override formatTime to add dark green color and brackets around timestamp.
        """
        # Get the formatted time with milliseconds
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime(self.default_time_format, ct)
        
        # Add milliseconds
        s = self.default_msec_format % (s, record.msecs)
        
        # Add dark green color and brackets
        return f"{self.DARKER_GREEN}[{s}]{self.RESET}"


def setup_tracing(service_name: str, enable_console_export: bool = False):
    """
    Initialize OpenTelemetry tracing for the application.
    
    Args:
        service_name: Name of the service (e.g., "supervisor", "finance-agent", "mcp-server")
        enable_console_export: If True, prints spans to console (useful for debugging)
    
    Example:
        # In your main app file
        setup_tracing("supervisor")
    """
    # Create a resource identifying this service
    resource = Resource(attributes={
        "service.name": service_name
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add console exporter if enabled (for debugging)
    if enable_console_export:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    # Set as global tracer provider
    trace.set_tracer_provider(provider)
    
    # Auto-instrument FastAPI (for MCP server)
    FastAPIInstrumentor().instrument()
    
    # Auto-instrument HTTP clients (for agent -> MCP calls)
    HTTPXClientInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    
    logger.info(f"OpenTelemetry tracing initialized for service: {service_name}")


def get_tracer(name: str):
    """
    Get a tracer for creating spans.
    
    Args:
        name: Name of the tracer (usually __name__)
    
    Returns:
        Tracer instance
    
    Example:
        tracer = get_tracer(__name__)
        
        with tracer.start_as_current_span("my_operation"):
            # Your code here
            logger.info("This log will include trace_id and span_id")
    """
    return trace.get_tracer(name)


def setup_logger_with_tracing(name: str, level: int = logging.INFO, service_name: str = "unknown") -> logging.Logger:
    """
    Creates a logger with both colored output and OpenTelemetry trace IDs.
    
    Args:
        name: Name of the logger (usually __name__)
        level: Logging level (default: INFO)
        service_name: Name of the service for log messages (default: "unknown")
    
    Returns:
        Configured logger instance
    
    Example:
        from src.utils.tracing import setup_logger_with_tracing
        logger = setup_logger_with_tracing(__name__, service_name="supervisor")
        logger.info("This message includes trace_id and service name")
    """
    import sys
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter with tracing support, timestamp, and service name
    # Format: [YYYY-MM-DD HH:MM:SS.mmm] [service] LEVEL [trace:span] file:line - message
    formatter = TracingFormatter(
        fmt='%(asctime)s [%(service_name)s] %(levelname)s:    %(trace_id)s %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        service_name=service_name
    )
    
    # Configure time formatting
    formatter.default_time_format = '%Y-%m-%d %H:%M:%S'
    formatter.default_msec_format = '%s.%03d'
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Convenience function for creating traced async functions
def traced(span_name: str = None):
    """
    Decorator to automatically create a span for a function.
    
    Args:
        span_name: Optional custom span name. If None, uses function name.
    
    Example:
        @traced("process_query")
        async def run_query(self, input: str):
            # This function is automatically traced
            logger.info("Processing...")
            return result
    """
    def decorator(func):
        import functools
        from opentelemetry import trace
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            name = span_name or func.__name__
            
            with tracer.start_as_current_span(name):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator