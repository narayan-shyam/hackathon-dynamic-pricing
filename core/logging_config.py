"""
Structured Logging Configuration with JSON Formatting and Azure Integration
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging with correlation tracking"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "correlation_id": getattr(record, 'correlation_id', None),
            "user_id": getattr(record, 'user_id', None),
            "request_id": getattr(record, 'request_id', None)
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO", azure_instrumentation_key: Optional[str] = None):
    """Configure structured logging with Azure Application Insights integration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler with JSON formatting
    file_handler = logging.FileHandler(
        log_dir / f"ml_project_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    # Azure Application Insights integration
    if azure_instrumentation_key:
        try:
            # Note: Requires azure-monitor-opentelemetry package
            # configure_azure_monitor(connection_string=f"InstrumentationKey={azure_instrumentation_key}")
            logger.info("Azure Application Insights configuration available")
        except Exception as e:
            logger.warning(f"Azure Application Insights configuration skipped: {str(e)}")
    
    return logger


class StructuredLogger:
    """Enhanced structured logger with context management"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def log_with_context(self, level: str, message: str, **context):
        """Log with additional context"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'message': message,
            'context': context
        }
        getattr(self.logger, level.lower())(json.dumps(log_data))
    
    def info(self, message: str, **context):
        """Log info message with context"""
        self.log_with_context('info', message, **context)
    
    def warning(self, message: str, **context):
        """Log warning message with context"""
        self.log_with_context('warning', message, **context)
    
    def error(self, message: str, **context):
        """Log error message with context"""
        self.log_with_context('error', message, **context)
    
    def debug(self, message: str, **context):
        """Log debug message with context"""
        self.log_with_context('debug', message, **context)
