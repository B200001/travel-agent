"""LangGraph checkpoint helpers (SQLite-backed).

This module provides a wrapper for managing LangGraph's SQLite-based
checkpoint system, which enables persistent conversation memory across sessions.
"""

import logging
from pathlib import Path
from typing import Any, Optional

# Set up a logger for this module to track warnings and errors
logger = logging.getLogger(__name__)

# Try to import the SQLite checkpointer from LangGraph
# This is wrapped in try/except because it's an optional dependency
try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    # Flag to track whether the checkpointer is available
    SQLITE_CHECKPOINTER_AVAILABLE = True
except ImportError:
    # If the import fails, set the flag to False and set SqliteSaver to None
    # This allows the code to run without the optional dependency
    SQLITE_CHECKPOINTER_AVAILABLE = False
    SqliteSaver = None  # type: ignore[assignment]


class SqliteCheckpointerManager:
    """Owns the lifecycle of a LangGraph SQLite checkpointer.
    
    This class manages creating, using, and cleaning up a SQLite-backed
    checkpoint system that persists conversation state to disk.
    """

    def __init__(self, db_path: Path):
        """Initialize the manager with a database path.
        
        Args:
            db_path: Path where the SQLite database file will be stored
        """
        # Store the database file path
        self._db_path = db_path
        
        # Will hold the context manager returned by SqliteSaver.from_conn_string
        self._cm: Optional[Any] = None
        
        # Will hold the actual checkpointer instance (what we get from __enter__)
        self.checkpointer: Optional[Any] = None

    def setup(self) -> Optional[Any]:
        """Create a checkpointer instance if sqlite checkpoint package is available.
        
        This method:
        1. Creates the parent directory for the database if needed
        2. Checks if the SQLite checkpointer is available
        3. Initializes the checkpointer using a context manager pattern
        
        Returns:
            The checkpointer instance if successful, None otherwise
        """
        # Create the directory structure for the database file if it doesn't exist
        # parents=True creates all parent directories as needed
        # exist_ok=True prevents errors if the directory already exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if the optional dependency is installed
        if not SQLITE_CHECKPOINTER_AVAILABLE:
            # Log a warning but continue running (just without persistence)
            logger.warning(
                "langgraph.checkpoint.sqlite not available; running without persistent thread memory."
            )
            return None

        try:
            # Create a SqliteSaver context manager connected to our database file
            # from_conn_string() returns a context manager that handles
            # opening/closing the database connection properly
            self._cm = SqliteSaver.from_conn_string(str(self._db_path))
            
            # Enter the context manager to get the actual checkpointer instance
            # This is equivalent to the object you'd get in: with SqliteSaver... as checkpointer
            self.checkpointer = self._cm.__enter__()
            
            # Return the checkpointer for use
            return self.checkpointer
            
        except Exception as e:
            # If anything goes wrong during initialization, log it and continue
            # without a checkpointer (graceful degradation)
            logger.warning("Failed to initialize LangGraph SQLite checkpointer: %s", e)
            self._cm = None
            self.checkpointer = None
            return None

    def close(self) -> None:
        """Close checkpointer resources if opened via context manager.
        
        This method properly cleans up the SQLite connection by exiting
        the context manager. It's safe to call even if setup() failed.
        """
        # If we never created a context manager, nothing to close
        if self._cm is None:
            return
        
        try:
            # Exit the context manager (closes database connection)
            # The three None arguments represent (exception_type, exception_value, traceback)
            # Passing None means "exit normally without an exception"
            self._cm.__exit__(None, None, None)
            
        except Exception:
            # Silently ignore any errors during cleanup
            # This is important during interpreter shutdown where resources
            # might already be cleaned up or in an inconsistent state
            pass
            
        finally:
            # Always reset our references to None, whether exit succeeded or not
            # This ensures we don't try to use a closed checkpointer
            self._cm = None
            self.checkpointer = None