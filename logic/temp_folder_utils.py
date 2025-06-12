import os
import shutil
import tempfile
from typing import Optional


# ---------------------------------------------------------------------------
# Helper functions for locating and managing temporary directories.
# We consider *all* common temp locations (Python's gettempdir and the TEMP/TMP
# environment variables).  Some systems may map these to different drives, so
# we handle them as a set of unique, existing paths.
# ---------------------------------------------------------------------------

def _collect_temp_paths() -> list[str]:
    """Return a de-duplicated list of temp directories that exist on disk."""
    paths = {tempfile.gettempdir()}

    for env_var in ("TMP", "TEMP", "TMPDIR"):
        val = os.environ.get(env_var)
        if val:
            paths.add(val)

    # Ensure they actually exist (some env vars might be bogus)
    existing_paths = [p for p in paths if os.path.isdir(p)]
    return existing_paths


def get_temp_folder_path() -> str:
    """Return *one* representative temp directory (first of the collected set)."""
    paths = _collect_temp_paths()
    return paths[0] if paths else tempfile.gettempdir()


def _safe_human_readable_size(num_bytes: int) -> str:
    """Convert a size in bytes to a human-readable string formatted to two decimals."""
    if num_bytes <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} EB"  # extremely large, unlikely


def calculate_temp_folder_size(logger: Optional[object] = None) -> int:
    """Return cumulative size (bytes) of all recognised temp directories."""
    total = 0
    for temp_dir in _collect_temp_paths():
        for root, _dirs, files in os.walk(temp_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    total += os.path.getsize(fpath)
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not access size of {fpath}: {e}")
    return total


def format_temp_folder_size(logger: Optional[object] = None) -> str:
    """Return human-readable total size of temp directory (e.g. '1.33 GB')."""
    return _safe_human_readable_size(calculate_temp_folder_size(logger))


def clear_temp_folder(logger: Optional[object] = None) -> bool:
    """Remove contents of all recognised temp directories. Returns True if no fatal errors."""
    success = True
    for temp_dir in _collect_temp_paths():
        for entry in os.listdir(temp_dir):
            entry_path = os.path.join(temp_dir, entry)
            try:
                if os.path.isfile(entry_path) or os.path.islink(entry_path):
                    os.remove(entry_path)
                else:
                    shutil.rmtree(entry_path, ignore_errors=True)
            except Exception as e:
                success = False
                if logger:
                    logger.error(f"Failed to delete '{entry_path}': {e}")
    return success 