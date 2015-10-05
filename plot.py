#!/usr/bin/env python
"""
This script plots results from the turbinesFoam cross-flow turbine actuator
line tutorial.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from py_unh_rvat_turbinesfoam.plotting import *
from pxl.styleplot import set_sns


if __name__ == "__main__":
    set_sns()
    plt.rcParams["axes.grid"] = True

    if len(sys.argv) > 1:
        if sys.argv[1] == "wake":
            plot_meancontquiv()
            plot_kcont()
        elif sys.argv[1] == "perf":
            plot_cp()
        elif sys.argv[1] == "blade":
            plot_blade_perf()
        elif sys.argv[1] == "strut":
            plot_strut_perf()
        elif sys.argv[1] == "perf-curves":
            plot_perf_curves(exp=False)
        elif sys.argv[1] == "perf-curves-exp":
            plot_perf_curves(exp=True)
    else:
        plot_cp()
    plt.show()
