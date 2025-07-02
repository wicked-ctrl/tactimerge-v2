import os

"""
Scan all .py files under the project for null bytes (\x00) and report their locations.
Usage:
    python find_nulls.py

# Extended detectors
# 1) Null bytes (