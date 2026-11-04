"""Shared limits and routing rules for the loopback extension backend."""
from pathlib import Path
import re
import secrets


ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_REQUEST_BYTES = 4 * ((MAX_UPLOAD_BYTES + 2) // 3) + 65536
MAX_ACTIVE_JOBS = 4
JOB_ID_PATTERN = re.compile(r'[0-9a-f]{32}\Z')
EXTENSION_ORIGIN_PATTERN = re.compile(r'chrome-extension://[a-p]{32}\Z')
RESULT_FILES = {'report': 'report.html', 'workspace': 'workspace.html', 'analysis': 'analysis.json'}


def same_secret(left: str, right: str) -> bool:
    """Compare credentials safely, including malformed non-ASCII input."""
    return secrets.compare_digest(left.encode('utf-8'), right.encode('utf-8'))
