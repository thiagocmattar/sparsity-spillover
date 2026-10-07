# Launcher namespace retry

Tag002 stopped during imports, before any timing/validation/diagnostic workload.
The retry launcher was named profile.py. Python's cProfile imported it instead
of the standard-library profile module, recursively starting the same attempt.
The immutable failed002 result and controller snapshot were retrieved/hash-
verified (96 cumulative files) before this fresh tag003 cohort.

measure.py has the same untimed shape adapter and removes its own launch folder
from sys.path before library imports. This prevents standard-library shadowing.
The original 02_profile.py, stages.py, production and clocked CUDA sources,
weights/gates, timers, seeds, sample/validation coverage and numerical bounds
remain unchanged. All six fresh processes repeat under tag003 on the same
physical RTX5090 and original two-hour deadline; failed001/002 stay immutable.
