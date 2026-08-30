"""Verify the six retrieved Run 004 attempts."""

from verification import verify_run


if __name__ == "__main__":
    result = verify_run()
    print(result["status"], result["condition_count"])

