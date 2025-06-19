"""
Utility Classes and Functions for Production ML Project
"""

import time
import functools
from pathlib import Path
from typing import Dict, List, Any, Callable
import logging

from .exceptions import ProjectSetupError

logger = logging.getLogger(__name__)


# Circuit Breaker Pattern Implementation
class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise ProjectSetupError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise e
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
    
    def reset(self):
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'


# Retry Mechanism Decorator with Exponential Backoff
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (backoff ** attempt))
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        continue
                    else:
                        logger.error(f"All {max_attempts} attempts failed. Last error: {str(e)}")
                        raise last_exception
            return None
        return wrapper
    return decorator


# Rate Limiter Utility with Token Bucket Algorithm
class RateLimiter:
    """Token bucket rate limiter implementation for API protection"""
    
    def __init__(self, max_tokens: int = 100, refill_period: float = 60.0):
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_period = refill_period
        self.last_refill = time.time()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket"""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = int(elapsed * (self.max_tokens / self.refill_period))
        
        if tokens_to_add > 0:
            self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
            self.last_refill = now


# Comprehensive Data Validation Utilities
class DataValidator:
    """Comprehensive data validation utilities with security features"""
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path exists and is accessible"""
        try:
            path = Path(file_path)
            return path.exists() and path.is_file()
        except Exception:
            return False
    
    @staticmethod
    def validate_directory_path(dir_path: str) -> bool:
        """Validate directory path exists and is accessible"""
        try:
            path = Path(dir_path)
            return path.exists() and path.is_dir()
        except Exception:
            return False
    
    @staticmethod
    def validate_json_schema(data: Dict[str, Any], required_fields: List[str]) -> tuple[bool, List[str]]:
        """Validate JSON data against required schema"""
        missing_fields = []
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
        
        return len(missing_fields) == 0, missing_fields
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks"""
        # Remove directory traversal patterns
        filename = filename.replace('../', '').replace('..\\', '')
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        return filename


# Security Utilities
class SecurityUtils:
    """Security and encryption utilities"""
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate secure random token"""
        import secrets
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_string(data: str) -> str:
        """Hash string using SHA-256"""
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """Sanitize user input"""
        # Basic sanitization - remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'"]
        sanitized = user_input
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        return sanitized.strip()


# Performance Monitoring Utilities
class PerformanceMonitor:
    """Simple performance monitoring utilities"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_time = time.time()
        self.current_operation = operation
    
    def end_timer(self) -> float:
        """End timing and return duration"""
        if self.start_time is None:
            return 0.0
        
        duration = time.time() - self.start_time
        if hasattr(self, 'current_operation'):
            self.metrics[self.current_operation] = duration
            logger.info(f"Operation '{self.current_operation}' completed in {duration:.2f} seconds")
        
        self.start_time = None
        return duration
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all collected metrics"""
        return self.metrics.copy()
