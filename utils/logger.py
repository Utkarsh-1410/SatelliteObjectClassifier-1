"""
Logging utilities for Satellite Classifier Application
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from utils.config import (
    LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, 
    MAX_LOG_FILE_SIZE, LOG_BACKUP_COUNT,
    DEFAULT_LOG_DIR, DEFAULT_LOG_FILENAME
)

# Global logger instance
_logger = None

def setup_logger():
    """
    Setup and configure the application logger.
    
    Returns:
        Logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # Create logger
    _logger = logging.getLogger('SatelliteClassifier')
    _logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    
    # Prevent duplicate handlers
    if _logger.handlers:
        return _logger
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)
    
    try:
        # Ensure log directory exists
        if not os.path.exists(DEFAULT_LOG_DIR):
            os.makedirs(DEFAULT_LOG_DIR)
        
        # File handler with rotation
        log_file_path = os.path.join(DEFAULT_LOG_DIR, DEFAULT_LOG_FILENAME)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=MAX_LOG_FILE_SIZE,
            backupCount=LOG_BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)
        
    except (OSError, PermissionError) as e:
        # If file logging fails, continue with console only
        _logger.warning(f"Could not setup file logging: {e}")
    
    # Log startup information
    _logger.info("=" * 60)
    _logger.info("Satellite Classifier Application Started")
    _logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _logger.info(f"Python Version: {sys.version}")
    _logger.info(f"Log Level: {LOG_LEVEL}")
    _logger.info("=" * 60)
    
    return _logger

def get_logger():
    """
    Get the application logger instance.
    
    Returns:
        Logger instance
    """
    if _logger is None:
        return setup_logger()
    return _logger

class GuiLogHandler(logging.Handler):
    """Custom logging handler that can send logs to GUI components."""
    
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                          '%H:%M:%S'))
    
    def emit(self, record):
        """Emit a log record to the GUI callback."""
        if self.callback:
            try:
                msg = self.format(record)
                self.callback(msg)
            except Exception:
                pass  # Ignore errors in GUI logging

def add_gui_handler(callback):
    """
    Add a GUI handler to the logger.
    
    Args:
        callback: Function to call with log messages
    """
    logger = get_logger()
    
    # Remove existing GUI handlers
    logger.handlers = [h for h in logger.handlers if not isinstance(h, GuiLogHandler)]
    
    # Add new GUI handler
    gui_handler = GuiLogHandler(callback)
    gui_handler.setLevel(logging.INFO)
    logger.addHandler(gui_handler)

def remove_gui_handlers():
    """Remove all GUI handlers from the logger."""
    logger = get_logger()
    logger.handlers = [h for h in logger.handlers if not isinstance(h, GuiLogHandler)]

class ContextualLogger:
    """Logger wrapper that adds contextual information."""
    
    def __init__(self, context=""):
        self.logger = get_logger()
        self.context = context
    
    def _format_message(self, message):
        """Format message with context."""
        if self.context:
            return f"[{self.context}] {message}"
        return message
    
    def debug(self, message, *args, **kwargs):
        """Log debug message with context."""
        self.logger.debug(self._format_message(message), *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        """Log info message with context."""
        self.logger.info(self._format_message(message), *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """Log warning message with context."""
        self.logger.warning(self._format_message(message), *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """Log error message with context."""
        self.logger.error(self._format_message(message), *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        """Log critical message with context."""
        self.logger.critical(self._format_message(message), *args, **kwargs)

def log_performance(func):
    """Decorator to log function performance."""
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__name__
        
        start_time = datetime.now()
        logger.debug(f"Starting {func_name}")
        
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.debug(f"Completed {func_name} in {duration:.2f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"Failed {func_name} after {duration:.2f}s: {e}")
            raise
    
    return wrapper

def log_exception(logger_instance=None):
    """Decorator to log exceptions with full traceback."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logger_instance or get_logger()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, operation_name, total_steps, context=""):
        self.logger = ContextualLogger(context)
        self.operation_name = operation_name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        
        self.logger.info(f"Starting {operation_name} ({total_steps} steps)")
    
    def step(self, message="", increment=1):
        """Log progress step."""
        self.current_step += increment
        
        if self.total_steps > 0:
            progress = (self.current_step / self.total_steps) * 100
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            if self.current_step > 0:
                estimated_total = elapsed * self.total_steps / self.current_step
                remaining = estimated_total - elapsed
                
                self.logger.info(
                    f"{self.operation_name} progress: {self.current_step}/{self.total_steps} "
                    f"({progress:.1f}%) - ETA: {remaining:.0f}s"
                    + (f" - {message}" if message else "")
                )
            else:
                self.logger.info(f"{self.operation_name} progress: {self.current_step}/{self.total_steps}")
        else:
            self.logger.info(f"{self.operation_name} step {self.current_step}" + (f" - {message}" if message else ""))
    
    def complete(self, message=""):
        """Log completion."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"Completed {self.operation_name} in {elapsed:.2f}s"
            + (f" - {message}" if message else "")
        )

def create_session_logger():
    """Create a logger for the current session with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_context = f"Session_{timestamp}"
    return ContextualLogger(session_context)

# Memory usage logging utilities
def log_memory_usage(logger_instance=None):
    """Log current memory usage if psutil is available."""
    logger = logger_instance or get_logger()
    
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.debug(f"Memory usage: {memory_mb:.1f} MB")
        
        # Warning if memory usage is high
        if memory_mb > 500:  # 500 MB threshold
            logger.warning(f"High memory usage detected: {memory_mb:.1f} MB")
            
    except ImportError:
        # psutil not available, skip memory logging
        pass
    except Exception as e:
        logger.debug(f"Could not get memory usage: {e}")

def setup_error_logging():
    """Setup global exception handler for unhandled exceptions."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger = get_logger()
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception

def shutdown_logger():
    """Shutdown the logger and close all handlers."""
    global _logger
    
    if _logger:
        _logger.info("Satellite Classifier Application Shutdown")
        _logger.info("=" * 60)
        
        # Close all handlers
        for handler in _logger.handlers[:]:
            handler.close()
            _logger.removeHandler(handler)
        
        _logger = None

# Initialize error logging when module is imported
setup_error_logging()
