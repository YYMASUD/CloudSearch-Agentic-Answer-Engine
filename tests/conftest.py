"""
conftest.py — ensures the repo root is on sys.path so all service imports resolve
without needing to install every package.
"""
import sys
import os

# Add project root to path so `services`, `packages` and `apps` are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Also ensure packages/python-shared is importable
SHARED = os.path.join(ROOT, "packages", "python-shared")
if SHARED not in sys.path:
    sys.path.insert(0, SHARED)
