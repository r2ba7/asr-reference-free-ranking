import time
import functools
from typing import Any, Callable

from stt_benchmarking.utils import logger

LOGGER = logger.Logger.get_logger(module_name=__name__)

class Decorators:
    
    @staticmethod
    @staticmethod
    def calculate_execution_time(func):
        @functools.wraps(func)
        def execution_time_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            minutes = int(execution_time // 60)
            seconds = int(execution_time % 60)
            LOGGER.info(f"Execution time for function {func.__name__}: "
                        f"{minutes} minutes {seconds} seconds ({execution_time:.1f}s)")
            if args and hasattr(args[0], "report"):
                t = args[0].report.setdefault("timings_seconds", {})
                t[func.__name__] = round(max(execution_time, t.get(func.__name__, 0.0)), 1)
            return result
        return execution_time_wrapper
    
    @staticmethod
    def timeout_with_retry(func: Callable) -> Callable:
        @functools.wraps(func)
        def timeout_wrapper(*args: Any, **kwargs: Any) -> Any:
            retry_seconds = 3
            max_retries = 4
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        LOGGER.error(f"Function '{func.__name__}' failed after {max_retries} retries. Error: {str(e)}")
                        raise e
                    
                    LOGGER.warning(f"Attempt {attempt} failed with error: {str(e)}. Retrying in {retry_seconds} seconds...")
                    time.sleep(retry_seconds)
                    continue
            
            return None
        return timeout_wrapper