#!/usr/bin/env python
"""Script to run UNH-RVAT ALM case."""

import argparse
import os
import subprocess
from subprocess import call, check_output
import numpy as np
import pandas as pd
import glob
import foampy
from foampy.dictionaries import replace_value
import numpy as np
import shutil
from pyurtf import processing as pr


def get_mesh_dims():
    """Get mesh dimensions by grepping `blockMeshDict`."""
    raw = check_output("grep blocks system/blockMeshDict -A3",
                       shell=True).decode().split("\n")[3]
    raw = raw.replace("(", "").replace(")", "").split()
    return {"nx": int(raw[0]), "ny": int(raw[1]), "nz": int(raw[2])}


def get_dt():
    """Read ``deltaT`` from ``controlDict``."""
    return foampy.dictionaries.read_single_line_value("controlDict",
                                                      keyword="deltaT")


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
    d["dt"] = get_dt()
    df = df.append(d, ignore_index=True)
    df.to_csv(fpath, index=False)


def set_blockmesh_resolution(nx=48, ny=None, nz=None):
    """Set mesh resolution in ``blockMeshDict``.

    If only ``nx`` is provided, the default resolutions for other dimensions are
    scaled proportionally.
    """
    defaults = {"nx": 48, "ny": 48, "nz": 32}
    if ny is None:
        ny = nx
    if nz is None:
        nz = int(nx*defaults["nz"]/defaults["nx"])
    print("Setting blockMesh resolution to ({} {} {})".format(nx, ny, nz))
    foampy.fill_template("system/blockMeshDict.template", nx=nx, ny=ny, nz=nz)


def set_dt(dt=0.005, tsr=None, tsr_0=1.9):
    """Set ``deltaT`` in ``controlDict``. Will scale proportionally if ``tsr``
    and ``tsr_0`` are supplied, such that steps-per-rev is consistent with
    ``tsr_0``.
    """
    if tsr is not None:
        dt = dt*tsr_0/tsr
        print("Setting deltaT = dt*tsr_0/tsr = {:.3f}".format(dt))
    dt = str(dt)
    foampy.fill_template("system/controlDict.template", dt=dt)


def set_talpha(val=6.25):
    """Set `TAlpha` value for the Leishman--Beddoes SGC dynamic stall model."""
    foampy.dictionaries.replace_value("system/fvOptions", "TAlpha", str(val))


def gen_sets_file():
    """Generate ``sets`` file for post-processing."""
    # Input parameters
    setformat = "raw"
    interpscheme = "cellPoint"
    fields = ["UMean", "UPrime2Mean", "kMean"]
    x = 1.0
    ymax = 1.5
    ymin = -1.5
    ny = 51
    zmax = 1.125
    zmin = -1.125
    nz = 19
    z_array = np.linspace(zmin, zmax, nz)
    txt = "\ntype sets;\n"
    txt +='libs ("libsampling.so");\n'
    txt += "setFormat " + setformat + ";\n"
    txt += "interpolationScheme " + interpscheme + ";\n\n"
    txt += "sets \n ( \n"
    for z in z_array:
        # Fix interpolation issues if directly on a face
        if z == 0.0:
            z += 1e-5
        txt += "    " + "profile_" + str(z) + "\n"
        txt += "    { \n"
        txt += "        type        uniform; \n"
        txt += "        axis        y; \n"
        txt += "        start       (" + str(x) + " " + str(ymin) + " " \
            + str(z) + ");\n"
        txt += "        end         (" + str(x) + " " + str(ymax) + " " \
            + str(z) + ");\n"
        txt += "        nPoints     " + str(ny) + ";\n    }\n\n"
    txt += ");\n\n"
    txt += "fields \n(\n"
    for field in fields:
        txt += "    " + field + "\n"
    txt += "); \n\n"
    txt += "//\
     *********************************************************************** //\
     \n"
    with open("system/sets", "w") as f:
        f.write(txt)


def post_process(parallel=False, tee=False, overwrite=True):
    """Execute all post-processing."""
    gen_sets_file()
    foampy.run("postProcess", args="-func -vorticity", parallel=parallel,
               logname="log.vorticity", tee=tee, overwrite=overwrite)
    foampy.run("postProcess", args="-dict system/controlDict.recovery "
               " -latestTime", parallel=parallel, logname="log.recovery",
               tee=tee, overwrite=overwrite)
    foampy.run("funkyDoCalc", args="system/funkyDoCalcDict -latestTime",
               parallel=parallel, tee=tee, overwrite=overwrite)


def param_sweep(param="tsr", start=None, stop=None, step=None, dtype=float,
                append=False, parallel=True, tee=False, **kwargs):
    """Run multiple simulations, varying ``quantity``.

    ``step`` is not included.
    """
    print("Running {} sweep".format(param))
    fpath = "processed/{}_sweep.csv".format(param)
    if not append and os.path.isfile(fpath):
        os.remove(fpath)
    if param == "nx":
        dtype = int
    param_list = np.arange(start, stop, step, dtype=dtype)
    for p in param_list:
        print("Running with {} = {}".format(param, p))
        if param == "talpha":
            set_talpha(p)
        if p == param_list[0] or param == "nx":
            foampy.clean(remove_zero=True)
            mesh = True
        else:
            mesh = False
        # Update kwargs for this value
        kwargs.update({param: p})
        run(parallel=parallel, tee=tee, mesh=mesh, reconstruct=False,
            post=False, **kwargs)
        os.rename("log.pimpleFoam", "log.pimpleFoam." + str(p))
        log_perf(param=param, append=True)
    # Set parameters back to defaults
    if param == "talpha":
        set_talpha()


def set_turbine_op_params(tsr=1.9, tsr_amp=0.0, tsr_phase=1.4):
    """Write file defining turbine operating parameters.

    ``tsr_phase`` is in radians.
    """
    txt="""
tipSpeedRatio   {tsr};
tsrAmplitude    {tsr_amp};
tsrPhase        {tsr_phase};
    """.format(tsr=tsr, tsr_amp=tsr_amp, tsr_phase=tsr_phase)
    with open("system/turbineOperatingParams", "w") as f:
        f.write(txt)


def run(tsr=1.9, tsr_amp=0.0, tsr_phase=1.4, nx=48, mesh=True, parallel=False,
        dt=0.005, tee=False, reconstruct=True, overwrite=False, post=True):
    """Run simulation once."""
    print("Setting TSR to", tsr)
    set_turbine_op_params(tsr=tsr, tsr_amp=tsr_amp, tsr_phase=tsr_phase)
    set_dt(dt=dt, tsr=tsr)
    if mesh:
        # Create blockMeshDict
        set_blockmesh_resolution(nx=nx)
        foampy.run("blockMesh", tee=tee)
    # Copy over initial conditions
    subprocess.call("cp -rf 0.orig 0 > /dev/null 2>&1", shell=True)
    if parallel and not glob.glob("processor*"):
        foampy.run("decomposePar", tee=tee)
        subprocess.call("for PROC in processor*; do cp -rf 0.orig/* $PROC/0; "
                        " done", shell=True)
    if mesh:
        foampy.run("snappyHexMesh", args="-overwrite", tee=tee,
                   parallel=parallel)
        foampy.run("topoSet", parallel=parallel, tee=tee)
        if parallel:
            foampy.run("reconstructParMesh", args="-constant -time 0", tee=tee)
    foampy.run("pimpleFoam", parallel=parallel, tee=tee, overwrite=overwrite)
    if parallel and reconstruct:
        foampy.run("reconstructPar", tee=tee, overwrite=overwrite)
    if post:
        post_process(overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNH-RVAT ALM case")
    parser.add_argument("--tsr", "-t", default=1.9, type=float, help="TSR")
    parser.add_argument("--nx", "-x", default=48, type=int, help="Number of "
                        "cells in the x-direction for the base mesh")
    parser.add_argument("--dt", default=0.005, type=float, help="Time step")
    parser.add_argument("--leave-mesh", "-l", default=False,
                        action="store_true", help="Leave existing mesh")
    parser.add_argument("--post", "-P", default=False, action="store_true",
                        help="Run post-processing (done by default at end of "
                             " run)")
    parser.add_argument("--param-sweep", "-p",
                        help="Run multiple simulations varying a parameter",
                        choices=["tsr", "nx", "dt", "talpha"])
    parser.add_argument("--start", default=0.4, type=float)
    parser.add_argument("--stop", default=3.5, type=float)
    parser.add_argument("--step", default=0.5, type=float)
    parser.add_argument("--serial", "-S", default=False, action="store_true")
    parser.add_argument("--append", "-a", default=False, action="store_true")
    parser.add_argument("--tee", "-T", default=False, action="store_true",
                        help="Print log files to terminal while running")
    args = parser.parse_args()

    if args.param_sweep:
        param_sweep(args.param_sweep, args.start, args.stop, args.step,
                    append=args.append, parallel=not args.serial,
                    tee=args.tee, nx=args.nx, dt=args.dt, tsr=args.tsr)
    elif not args.post:
        run(tsr=args.tsr, nx=args.nx, dt=args.dt, parallel=not args.serial,
            tee=args.tee, mesh=not args.leave_mesh, overwrite=args.leave_mesh)
    if args.post:
        post_process()
