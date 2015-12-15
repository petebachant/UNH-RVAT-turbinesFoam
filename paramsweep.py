#!/usr/bin/env python
"""
Run multiple simulations varying a single parameter.
"""

from __future__ import division, print_function
import foampy
from foampy.dictionaries import replace_value
import numpy as np
from subprocess import call
import os
import pandas as pd
from pyurtf import processing as pr


def set_tsr_fluc(val=0.0):
    """Set TSR fluctuation amplitude to zero."""
    replace_value("system/fvOptions", "tsrAmplitude", val)


def set_tsr(val):
    """Set mean tip speed ratio."""
    print("Setting TSR to", val)
    replace_value("system/fvOptions", "tipSpeedRatio", val)


def log_perf(param="tsr", append=True):
    """Log performance to file."""
    if not os.path.isdir("processed"):
        os.mkdir("processed")
    fpath = "processed/{}_sweep.csv".format(param)
    if append and os.path.isfile(fpath):
        df = pd.read_csv(fpath)
    else:
        df = pd.DataFrame(columns=["tsr", "cp", "cd"])
    df = df.append(pr.calc_perf(t1=3.0), ignore_index=True)
    df.to_csv(fpath, index=False)


def set_blockmesh_resolution(nx=32, ny=None, nz=None):
    """Set mesh resolution in `blockMeshDict`.

    If only `nx` is provided, the default resolutions for other dimensions are
    scaled proportionally.
    """
    defaults = {"nx": 32, "ny": 32, "nz": 24}
    if ny is None:
        ny = nx
    if nz is None:
        nz = int(nx*defaults["nz"]/defaults["nx"])
    resline = "({nx} {ny} {nz})".format(nx=nx, ny=ny, nz=nz)
    blocks = """blocks
(
    hex (0 1 2 3 4 5 6 7)
    {}
    simpleGrading (1 1 1)
);
""".format(resline)
    foampy.dictionaries.replace_value("system/blockMeshDict",
                                      "blocks", blocks)


def set_timestep(dt=0.01):
    """Set `deltaT` in `controlDict`."""
    dt = str(dt)
    foampy.dictionaries.replace_value("system/controlDict", "deltaT", dt)


def tsr_sweep(start=0.4, stop=3.4, step=0.5, append=False):
    """Run over multiple TSRs. `stop` will be included."""
    if not append and os.path.isfile("processed/tsr_sweep.csv"):
        os.remove("processed/tsr_sweep.csv")
    tsrs = np.arange(start, stop + 0.5*step, step)
    set_tsr_fluc(0.0)
    cp = []
    cd = []
    for tsr in tsrs:
        set_tsr(tsr)
        if tsr == tsrs[0]:
            call("./Allclean")
            call("./Allrun")
        else:
            print("Running pimpleFoam")
            call("pimpleFoam > log.pimpleFoam", shell=True)
        os.rename("log.pimpleFoam", "log.pimpleFoam." + str(tsr))
        log_perf(append=True)
    # Set TSR parameters back to default
    set_tsr(1.9)
    set_tsr_fluc(0.19)


if __name__ == "__main__":
    tsr_sweep(0.4, 3.4, 0.5, append=False)
