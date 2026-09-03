#!/usr/bin/env python
"""Apply Run 020's predeclared training-only learning-rate rule."""

from selection import select_learning_rate


if __name__ == "__main__":
    result = select_learning_rate()
    print(result["status"], result["selected_condition_id"], result["next_action"])
