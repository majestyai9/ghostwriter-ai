"""
File operations utilities for atomic saves and safe file handling.
"""

import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict

from exceptions import FileOperationError


class FileOperations:
    """Handle atomic file operations and safe file handling."""

    def __init__(self) -> None:
        """Initialize the file operations handler."""
        self.logger = logging.getLogger(__name__)

    def save_json_atomically(self, data: Dict[str, Any], file_path: str) -> None:
        """
        Save JSON data atomically using temp file and rename.

        This ensures that the file is either completely written or not written
        at all, preventing partial writes that could corrupt the data.

        Args:
            data: Dictionary data to save as JSON
            file_path: Target path for the JSON file

        Raises:
            FileOperationError: If the save operation fails
        """
        # Create temp file in same directory for atomic rename
        dir_path = os.path.dirname(file_path)
        
        # Ensure directory exists
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=dir_path,
            suffix='.tmp',
            delete=False
        ) as tmp_file:
            try:
                json.dump(data, tmp_file, indent=4, ensure_ascii=False)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())  # Ensure data is written to disk
                tmp_path = tmp_file.name
            except Exception as e:
                os.unlink(tmp_file.name)
                raise FileOperationError(f"Failed to write temp file: {e}")

        # Atomic rename
        try:
            self._atomic_rename(tmp_path, file_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise FileOperationError(f"Failed to save file atomically: {e}")

    def _atomic_rename(self, source: str, destination: str) -> None:
        """
        Perform atomic rename operation, handling platform differences.

        Args:
            source: Source file path
            destination: Destination file path
        """
        if os.name == 'nt':  # Windows
            # Windows doesn't support atomic rename if target exists
            if os.path.exists(destination):
                backup_path = f"{destination}.backup"
                shutil.copy2(destination, backup_path)
            shutil.move(source, destination)
        else:  # Unix-like
            os.rename(source, destination)

    def load_json_safely(self, file_path: str) -> Dict[str, Any]:
        """
        Safely load JSON file with error handling.

        Args:
            file_path: Path to the JSON file

        Returns:
            Loaded JSON data as dictionary

        Raises:
            FileOperationError: If loading fails
        """
        try:
            with open(file_path, encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise FileOperationError(f"Invalid JSON format in {file_path}: {e}")
        except OSError as e:
            raise FileOperationError(f"Failed to read {file_path}: {e}")