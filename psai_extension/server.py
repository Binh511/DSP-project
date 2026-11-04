"""Compatibility imports for the local extension backend.

Use the focused modules for new implementation. Existing scripts can continue
to import JobManager, create_server, constants, and main from this module.
"""
# Keep the shared stdlib module available for existing subprocess test patches.
# pipeline.py imports this same module object, so its Popen is patched as well.
import subprocess as subprocess

from .cli import main
from .config import MAX_ACTIVE_JOBS, MAX_REQUEST_BYTES, MAX_UPLOAD_BYTES, ROOT
from .http import create_server
from .jobs import JobManager, QueueFullError

__all__ = ['JobManager', 'QueueFullError', 'create_server', 'main',
           'MAX_ACTIVE_JOBS', 'MAX_REQUEST_BYTES', 'MAX_UPLOAD_BYTES', 'ROOT']
