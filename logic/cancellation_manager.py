# logic/cancellation_manager.py
import threading
import subprocess
from typing import Optional
import logging

class CancelledError(Exception):
    """Custom exception to signal a graceful cancellation."""
    pass

class CancellationManager:
    def __init__(self):
        self._cancel_event = threading.Event()
        self._active_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._logger = logging.getLogger(__name__)

    def request_cancellation(self):
        """Signal that a cancellation has been requested."""
        self._logger.warning("Cancellation requested via CancellationManager")
        
        # If already cancelled, return False to indicate no new cancellation
        if self._cancel_event.is_set():
            self._logger.debug("Cancellation already in progress")
            return False
            
        self._cancel_event.set()
        with self._lock:
            if self._active_process:
                try:
                    self._logger.info(f"Terminating active process: {self._active_process.pid}")
                    self._active_process.terminate()
                    # Give it a moment to terminate gracefully
                    try:
                        self._active_process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._logger.warning("Process didn't terminate gracefully, killing it")
                        self._active_process.kill()
                except Exception as e:
                    self._logger.error(f"Error terminating process: {e}")
                    
        return True  # Successfully initiated cancellation

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_event.is_set()

    def check_cancel(self, context: str = ""):
        """Check for cancellation and raise CancelledError if requested."""
        if self.is_cancelled():
            msg = f"Cancellation check triggered{' in ' + context if context else ''} - raising CancelledError"
            self._logger.info(msg)
            raise CancelledError("Processing was cancelled by the user.")

    def check_cancel_with_timeout(self, timeout_seconds: float = 0.1, context: str = ""):
        """
        Check for cancellation with a small timeout to allow more responsive cancellation
        during long-running operations that can't be directly interrupted.
        """
        if self._cancel_event.wait(timeout_seconds):
            msg = f"Cancellation detected{' in ' + context if context else ''} - raising CancelledError"
            self._logger.info(msg)
            raise CancelledError("Processing was cancelled by the user.")

    def reset(self):
        """Reset the cancellation state for a new operation."""
        was_cancelled = self._cancel_event.is_set()
        self._cancel_event.clear()
        with self._lock:
            self._active_process = None
        if was_cancelled:
            self._logger.info("Cancellation manager state reset (was previously cancelled)")
        else:
            self._logger.debug("Cancellation manager state reset")

    def set_active_process(self, process: subprocess.Popen):
        """Store the currently running subprocess."""
        with self._lock:
            self._active_process = process
            self._logger.debug(f"Set active process: {process.pid if process else None}")

    def clear_active_process(self):
        """Clear the stored subprocess."""
        with self._lock:
            old_pid = self._active_process.pid if self._active_process else None
            self._active_process = None
            self._logger.debug(f"Cleared active process: {old_pid}")

# Global instance to be used across the application
cancellation_manager = CancellationManager() 