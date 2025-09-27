from dataclasses import dataclass, field
import logging
import inspect
import os
from typing import Optional, Dict, Any, ClassVar
import colorlog
import pytz
from datetime import datetime
from functools import lru_cache
from pathlib import Path

@dataclass
class Logger:
    '''
    Logger class to log messages to the console with colored output and timestamps.
    Supports file logging, custom formatting, and Cairo timezone.
    '''
    logger_name: str = "Default Logger"
    log_level: int = logging.INFO
    logger: logging.Logger = field(init=False)
    _instances: Dict[str, 'Logger'] = field(default_factory=dict, init=False)
    
    # Add class variable for singleton pattern
    _class_instances: ClassVar[Dict[str, 'Logger']] = {}
    
    def __post_init__(self):
        """Initialize logger with colored output and timestamp filter."""
        self.logger = logging.getLogger(self.logger_name)
        
        # CRITICAL: Clear any existing handlers first
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Set up the console handler
        self._setup_console_handler()
        self.logger.addFilter(self.TimingFilter())
        
        # Set level and disable propagation to prevent duplicate logs
        self.logger.setLevel(self.log_level)
        self.logger.propagate = False

    def _setup_console_handler(self) -> None:
        """Set up console handler with colored output."""
        # Double-check: Only add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                colorlog.ColoredFormatter(
                    "%(log_color)s%(time)s - [%(name)s.%(funcName)s:%(lineno)d] %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                    secondary_log_colors={},
                    style='%'
                )
            )
            self.logger.addHandler(handler)

    class TimingFilter(logging.Filter):
        """Custom logging filter to include Cairo timezone in log records."""
        def __init__(self):
            super().__init__()
            self.cairo_tz = pytz.timezone('Africa/Cairo')

        def filter(self, record: logging.LogRecord) -> bool:
            record.time = datetime.now(self.cairo_tz).strftime('%Y-%m-%d %H:%M:%S')
            return True

    @classmethod
    def get_logger(cls, module_name: Optional[str] = None, include_function: bool = True) -> 'Logger':
        """
        Get or create a logger instance using module name.
        Uses proper singleton pattern to prevent duplicate instances.
        """
        
        if not module_name:
            # Inspect the caller frame
            caller_frame = inspect.stack()[1]
            module = inspect.getmodule(caller_frame[0])
            function_name = caller_frame.function
            if module:
                file_path = Path(inspect.getfile(module))
                try:
                    # Try to compute the relative path
                    relative_path = file_path.relative_to(Path.cwd())
                    module_name = f"{module.__name__}.{relative_path.as_posix().replace('/', '.')}"
                except ValueError:
                    module_name = f"{module.__name__}.{file_path.as_posix().replace('/', '.')}"
            else:
                module_name = "unnamed_module"

            # Optionally append the function name
            if include_function and function_name != "<module>":
                module_name = f"{module_name}.{function_name}"

        # Create a unique cache key for proper singleton behavior
        cache_key = f"{module_name}_{include_function}"
        
        # Check if instance already exists in class variable
        if cache_key in cls._class_instances:
            return cls._class_instances[cache_key]
        
        # Create new instance and store in cache
        instance = cls(logger_name=module_name)
        cls._class_instances[cache_key] = instance
        
        return instance

    def add_file_handler(
        self,
        file_path: str,
        level: int = logging.INFO,
        rotation_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> None:
        """
        Add a rotating file handler with specified parameters.
        
        Args:
            file_path: Path to log file
            level: Logging level for file handler
            rotation_size: Max file size before rotation in bytes
            backup_count: Number of backup files to keep
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=rotation_size,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(time)s - %(name)s.%(funcName)s:%(lineno)d - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(self.TimingFilter())
        self.logger.addHandler(file_handler)

    def _log(self, level: int, message: Any, stacklevel: int = 3, **kwargs) -> None:
        """Internal logging method with proper caller information."""
        self.logger.findCaller(stack_info=True)
        self.logger.log(level, message, stacklevel=stacklevel, **kwargs)

    def info(self, message: Any, disable: bool = False, **kwargs) -> None:
        """Log message with INFO level."""
        if not disable:
            self._log(logging.INFO, message, **kwargs)

    def error(self, message: Any, **kwargs) -> None:
        """Log message with ERROR level."""
        self._log(logging.ERROR, message, **kwargs)

    def exception(self, message: Any, **kwargs) -> None:
        """Log message with ERROR level including exception information."""
        self._log(logging.ERROR, message, exc_info=True, **kwargs)

    def warning(self, message: Any, **kwargs) -> None:
        """Log message with WARNING level."""
        self._log(logging.WARNING, message, **kwargs)

    def debug(self, message: Any, **kwargs) -> None:
        """Log message with DEBUG level."""
        self._log(logging.DEBUG, message, **kwargs)

    def critical(self, message: Any, **kwargs) -> None:
        """Log message with CRITICAL level."""
        self._log(logging.CRITICAL, message, **kwargs)