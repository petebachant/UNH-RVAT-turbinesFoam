#!/usr/bin/env python
"""
This script plots results from the UNH-RVAT turbinesFoam case.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from py_unh_rvat_turbinesfoam.plotting import *
from pxl.styleplot import set_sns
import argparse


if __name__ == "__main__":
    set_sns()
    plt.rcParams["axes.grid"] = True

    parser = argparse.ArgumentParser(description="Generate plots.")
    parser.add_argument("plot", nargs="*", help="What to plot", default="perf",
                        choices=["perf", "wake", "blade-perf", "strut-perf",
                                 "perf-curves", "perf-curves-exp"])
    parser.add_argument("--all", "-A", help="Generate all figures",
                        default=False, action="store_true")
    parser.add_argument("--save", "-s", help="Save to `figures` directory",
                        default=False, action="store_true")
    parser.add_argument("--noshow", help="Do not call matplotlib show function",
                        default=False, action="store_true")
    args = parser.parse_args()

    if "wake" in args.plot or args.all:
        plot_meancontquiv(save=args.save)
        plot_kcont(save=args.save)
    if "perf" in args.plot or args.all:
        plot_cp(save=args.save)
    if "blade-perf" in args.plot or args.all:
        plot_blade_perf(save=args.save)
    if "strut-perf" in args.plot or args.all:
        plot_strut_perf(save=args.save)
    if "perf-curves" in args.plot or args.all:
        plot_perf_curves(exp=False, save=args.save)
    if "perf-curves-exp" in args.plot or args.all:
        plot_perf_curves(exp=True, save=args.save)

    if not args.noshow:
        plt.show()
