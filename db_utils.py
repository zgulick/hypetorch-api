# db_utils.py
import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar, cast
import time
import functools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_utils')

# Type variable for function return
T = TypeVar('T')

def with_retry(max_retries: int = 3, retry_delay_base: float = 1.0) -> Callable:
    """
    Decorator to retry database operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay_base: Base delay time in seconds (will be multiplied by 2^attempt)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = retry_delay_base * (2 ** attempt)
                        operation_name = func.__name__
                        logger.warning(
                            f"Database operation '{operation_name}' failed (attempt {attempt+1}/{max_retries+1}): {str(e)}. "
                            f"Retrying in {delay:.2f} seconds."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Database operation '{func.__name__}' failed after {max_retries+1} attempts: {str(e)}")
                        raise
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            # This makes the type checker happy
            return cast(T, None)
        return wrapper
    return decorator

def with_connection(conn_func):
    """
    Decorator to handle database connections safely.
    
    This ensures connections are always properly closed, even if errors occur.
    
    Args:
        conn_func: Function that returns a database connection
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            conn = None
            try:
                # Get connection
                conn = conn_func()
                # Add connection as first arg to function
                result = func(conn, *args, **kwargs)
                return result
            except Exception as e:
                # Log error
                logger.error(f"Database error in {func.__name__}: {str(e)}")
                # Re-raise the exception
                raise
            finally:
                # Ensure connection is closed
                if conn:
                    try:
                        conn.close()
                        logger.debug("Database connection closed successfully")
                    except Exception as e:
                        logger.warning(f"Error closing database connection: {str(e)}")
        return wrapper
    return decorator

def timing_decorator(func):
    """Decorator to time function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ {func.__name__} executed in {(end_time - start_time) * 1000:.2f}ms")
        return result
    return wrapper

def transactional(func):
    """
    Decorator to make a database operation transactional.
    
    Automatically commits if successful, rolls back if an error occurs.
    
    Note: Must be used with functions that take a connection as their first argument.
    """
    @wraps(func)
    def wrapper(conn, *args, **kwargs):
        try:
            result = func(conn, *args, **kwargs)
            conn.commit()
            return result
        except Exception as e:
            logger.error(f"Transaction error in {func.__name__}: {str(e)}")
            if conn:
                try:
                    conn.rollback()
                    logger.info("Transaction rolled back")
                except Exception as rollback_error:
                    logger.warning(f"Error rolling back transaction: {str(rollback_error)}")
            raise
    return wrapper