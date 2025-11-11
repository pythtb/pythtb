import sys
import os

# Ensure local pythtb.py is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
import pythtb  # noqa: E402


def test_version_exists_and_format():
    # __version__ should be defined and be a non-empty string
    assert hasattr(pythtb, "__version__"), "pythtb must define a __version__"
    version = pythtb.__version__
    assert isinstance(version, str) and version, (
        "__version__ should be a non-empty string"
    )
