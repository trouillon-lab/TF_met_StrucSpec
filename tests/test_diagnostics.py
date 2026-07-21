import os
import pytest
from scripts.diagnose_environment import run_diagnostics

def test_run_diagnostics():
    # Execute diagnostics in current environment
    status = run_diagnostics()
    # Should complete without throwing exceptions
    assert isinstance(status, bool)
