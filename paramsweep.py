#!/usr/bin/env python
"""
Run multiple simulations varying a single parameter.
"""

from __future__ import division, print_function
import foampy
from foampy.dictionaries import replace_value
import numpy as np
from subprocess import call, check_output
import os
import shutil
import pandas as pd
from pyurtf import processing as pr


def get_mesh_dims():
    """Get mesh dimensions by grepping `blockMeshDict`."""
    raw = check_output("grep blocks system/blockMeshDict -A3",
                       shell=True).decode().split("\n")[3]
    raw = raw.replace("(", "").replace(")", "").split()
    return {"nx": int(raw[0]), "ny": int(raw[1]), "nz": int(raw[2])}


def get_timestep():
    """Read `deltaT` from `controlDict`."""
    return foampy.dictionaries.read_single_line_value("controlDict", "deltaT")


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
        df = pd.DataFrame(columns=["nx", "ny", "nz", "dt", "tsr", "cp", "cd"])
    d = pr.calc_perf(t1=3.0)
    d.update(get_mesh_dims())
    d["dt"] = get_timestep()
    df = df.append(d, ignore_index=True)
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
    resline = "({nx} {ny} {nz})".format(nx=int(nx), ny=int(ny), nz=int(nz))
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


def run_solver(parallel=True):
    """Run `pimpleFoam`."""
    if parallel:
        call("mpirun -np 2 pimpleFoam -parallel > log.pimpleFoam", shell=True)
    else:
        call("pimpleFoam > log.pimpleFoam", shell=True)


def param_sweep(param="tsr", start=None, stop=None, step=None, dtype=float,
                append=False, parallel=True):
    """Run multiple simulations, varying `quantity`.

    `step` is not included.
    """
    print("Running {} sweep".format(param))
    fpath = "processed/{}_sweep.csv".format(param)
    if not append and os.path.isfile(fpath):
        os.remove(fpath)
    if param == "nx":
        dtype = int
    param_list = np.arange(start, stop, step, dtype=dtype)
    if param == "tsr":
        set_tsr_fluc(0.0)
    for p in param_list:
        print("Setting {} to {}".format(param, p))
        if param == "tsr":
            set_tsr(p)
        elif param == "nx":
            set_blockmesh_resolution(nx=p)
        elif param == "timestep":
            set_timestep(p)
        if p == param_list[0] or param == "nx":
            call("./Allclean")
            print("Running blockMesh")
            call("blockMesh > log.blockMesh", shell=True)
            print("Running snappyHexMesh")
            call("snappyHexMesh -overwrite > log.snappyHexMesh", shell=True)
            print("Running topoSet")
            call("topoSet > log.topoSet", shell=True)
            shutil.copytree("0.org", "0")
            if parallel:
                print("Running decomposePar")
                call("decomposePar > log.decomposePar", shell=True)
                call("ls -d processor* | xargs -I {} rm -rf ./{}/0", shell=True)
                call("ls -d processor* | xargs -I {} cp -r 0.org ./{}/0",
                     shell=True)
            print("Running pimpleFoam")
            run_solver(parallel=parallel)
        else:
            print("Running pimpleFoam")
            run_solver(parallel=parallel)
        os.rename("log.pimpleFoam", "log.pimpleFoam." + str(p))
        log_perf(param=param, append=True)
    # Set parameters back to defaults
    if param == "tsr":
        set_tsr(1.9)
        set_tsr_fluc(0.19)
    elif param == "nx":
        set_blockmesh_resolution()
    elif param == "timestep":
        set_timestep()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run mulitple simulations, "
                                     "varying a single parameter.")
    parser.add_argument("parameter", default="tsr", help="Parameter to vary",
                        nargs="?")
    parser.add_argument("start", default=0.4, type=float, nargs="?")
    parser.add_argument("stop", default=3.5, type=float, nargs="?")
    parser.add_argument("step", default=0.5, type=float, nargs="?")
    parser.add_argument("--parallel", default=True, type=bool)
    parser.add_argument("--append", "-a", default=False, action="store_true")

    args = parser.parse_args()

    param_sweep(args.parameter, args.start, args.stop, args.step,
                append=args.append, parallel=args.parallel)
