#!/usr/bin/env python
"""Verify the four retrieved Run 009 attempts and Run 004 matches."""

from verification import verify_run


if __name__ == "__main__":
    result = verify_run()
    print(result["status"], result["condition_count"])
