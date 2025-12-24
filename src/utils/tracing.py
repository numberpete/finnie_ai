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
        super().__init__(*args, **kwargs)
        self.service_name = service_name
    
    def format(self, record):
        span = trace.get_current_span()
        span_context = span.get_span_context()
        
        if span_context.is_valid:
            trace_id = format(span_context.trace_id, '032x')[:8]
            span_id = format(span_context.span_id, '016x')[:8]
            record.trace_id = f"[{trace_id}:{span_id}]"
        else:
            record.trace_id = ""
        
        record.service_name = f"{self.BLUE}{self.service_name}{self.RESET}"
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        return super(ColoredFormatter, self).format(record)
    
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        s = time.strftime(datefmt, ct) if datefmt else time.strftime(self.default_time_format, ct)
        s = self.default_msec_format % (s, record.msecs)
        return f"{self.DARKER_GREEN}[{s}]{self.RESET}"


def setup_tracing(service_name: str, enable_console_export: bool = False):
    """
    Initialize OpenTelemetry tracing for the application.
    Includes guards to prevent 'Overriding of current TracerProvider' warnings.
    """
    # 1. GUARD: Check if provider is already an SDK TracerProvider
    current_provider = trace.get_tracer_provider()
    
    # If the current provider has 'get_span_processor', it's already a real SDK provider.
    if hasattr(current_provider, 'get_span_processor'):
        logger.debug(f"Tracing already initialized for {service_name}, skipping setup.")
        return

    # Create a resource identifying this service
    resource = Resource(attributes={
        "service.name": service_name
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    if enable_console_export:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    # Set as global tracer provider
    try:
        trace.set_tracer_provider(provider)
    except ValueError:
        # Fallback in case of race condition
        pass
    
    # 2. GUARD: Auto-instrument ONLY if not already instrumented
    # FastAPIInstrumentor doesn't have a simple is_instrumented check, 
    # but we can wrap it in a try-except or check the global state.
    
    try:
        # FastAPI
        FastAPIInstrumentor().instrument()
        # HTTP clients
        HTTPXClientInstrumentor().instrument()
        RequestsInstrumentor().instrument()
    except Exception as e:
        # Silence "Already instrumented" errors
        if "already instrumented" not in str(e).lower():
            logger.warning(f"Instrumentation warning: {e}")
    
    logger.info(f"OpenTelemetry tracing initialized for service: {service_name}")


def get_tracer(name: str):
    return trace.get_tracer(name)


def setup_logger_with_tracing(name: str, level: int = logging.INFO, service_name: str = "unknown") -> logging.Logger:
    import sys
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    formatter = TracingFormatter(
        fmt='%(asctime)s [%(service_name)s] %(levelname)s:    %(trace_id)s %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        service_name=service_name
    )
    
    formatter.default_time_format = '%Y-%m-%d %H:%M:%S'
    formatter.default_msec_format = '%s.%03d'
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


def traced(span_name: str = None):
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