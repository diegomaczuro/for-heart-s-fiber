#import cython
from petsc_2d import *
#%load_ext line_profiler
import line_profiler
import numpy as np


def cumulative_sum():
    a = np.random.rand(10, 10)
    return np.sum(a)

print cumulative_sum()
#Print profiling statistics using the `line_profiler` API
profile = line_profiler.LineProfiler(cumulative_sum)
profile.runcall(cumulative_sum)
profile.print_stats()
