"""Run tsl_bench.main with the CalJEPA model registered."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tsl_jepa  # noqa: F401  registers CalJEPA-TCN
import tsl_bench

if __name__ == '__main__':
    tsl_bench.main()
