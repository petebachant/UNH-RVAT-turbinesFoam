"""Processing functions for UNH-RVAT turbinesFoam case."""

from __future__ import division, print_function
import matplotlib.pyplot as plt
import re
import numpy as np
import os
import sys
import foampy
from pxl import fdiff, timeseries as ts
import pandas as pd

# Some constants
R = 0.5
U = 1.0
H = 1.0
D = 1.0
A = H*D
rho = 1000.0
nu = 1e-6
A_c = 3.66*2.44 # Tow tank cross-section

ylabels = {"meanu" : r"$U/U_\infty$",
           "stdu" : r"$\sigma_u/U_\infty$",
           "meanv" : r"$V/U_\infty$",
           "meanw" : r"$W/U_\infty$",
           "meanuv" : r"$\overline{u'v'}/U_\infty^2$"}


class WakeMap(object):
    """
    Object that represents a wake map or statistics.
    """
    def __init__(self):
        self.load()

    def load_single_time(self, time):
        """
        Loads data from a single time step.
        """
        timedir = "postProcessing/sets/{}".format(time)


def loadwake(time):
    """Loads wake data and returns y/R and statistics."""
    # Figure out if time is an int or float
    if not isinstance(time, str):
        if time % 1 == 0:
            folder = str(int(time))
        else:
            folder = str(time)
    else:
        folder = time
    flist = os.listdir("postProcessing/sets/"+folder)
    data = {}
    for fname in flist:
        fpath = "postProcessing/sets/"+folder+"/"+fname
        z_H = float(fname.split("_")[1])
        data[z_H] = np.loadtxt(fpath, unpack=True)
    return data


def calcwake(t1=0.0):
    times = os.listdir("postProcessing/sets")
    times = [float(time) for time in times]
    times.sort()
    times = np.asarray(times)
    data = loadwake(times[0])
    z_H = np.asarray(sorted(data.keys()))
    y_R = data[z_H[0]][0]/R
    # Find first timestep from which to average over
    t = times[times >= t1]
    # Assemble 3-D arrays, with time as first index
    u = np.zeros((len(t), len(z_H), len(y_R)))
    v = np.zeros((len(t), len(z_H), len(y_R)))
    w = np.zeros((len(t), len(z_H), len(y_R)))
    xvorticity = np.zeros((len(t), len(z_H), len(y_R)))
    # Loop through all times
    for n in range(len(t)):
        data = loadwake(t[n])
        for m in range(len(z_H)):
            u[n,m,:] = data[z_H[m]][1]
            v[n,m,:] = data[z_H[m]][2]
            w[n,m,:] = data[z_H[m]][3]
            try:
                xvorticity[n,m,:] = data[z_H[m]][4]
            except IndexError:
                pass
    meanu = u.mean(axis=0)
    meanv = v.mean(axis=0)
    meanw = w.mean(axis=0)
    xvorticity = xvorticity.mean(axis=0)
    return {"meanu" : meanu,
            "meanv" : meanv,
            "meanw" : meanw,
            "xvorticity" : xvorticity,
            "y/R" : y_R,
            "z/H" : z_H}


def load_u_profile(z_H=0.0):
    """
    Loads data from the sampled mean velocity and returns it as a pandas
    `DataFrame`.
    """
    z_H = float(z_H)
    timedirs = os.listdir("postProcessing/sets")
    latest_time = max(timedirs)
    fname = "profile_{}_UMean.xy".format(z_H)
    data = np.loadtxt(os.path.join("postProcessing", "sets", latest_time,
                      fname), unpack=True)
    df = pd.DataFrame()
    df["y_R"] = data[0]/R
    df["u"] = data[1]
    return df


def load_vel_map(component="u"):
    """
    Loads all mean streamwise velocity profiles. Returns a `DataFrame` with
    `z_H` as the index and `y_R` as columns.
    """
    # Define columns in set raw data file
    columns = dict(u=1, v=2, w=3)
    sets_dir = os.path.join("postProcessing", "sets")
    latest_time = max(os.listdir(sets_dir))
    data_dir = os.path.join(sets_dir, latest_time)
    flist = os.listdir(data_dir)
    z_H = []
    for fname in flist:
        if "UMean" in fname:
            z_H.append(float(fname.split("_")[1]))
    z_H.sort()
    z_H.reverse()
    vel = []
    for zi in z_H:
        fname = "profile_{}_UMean.xy".format(zi)
        rawdata = np.loadtxt(os.path.join(data_dir, fname), unpack=True)
        vel.append(rawdata[columns[component]])
    y_R = rawdata[0]/R
    vel = np.array(vel).reshape((len(z_H), len(y_R)))
    df = pd.DataFrame(vel, index=z_H, columns=y_R)
    return df


def load_k_profile(z_H=0.0):
    """Load data from the sampled `UPrime2Mean` and `kMean` (if available) and
    returns it as a pandas `DataFrame`.
    """
    z_H = float(z_H)
    df = pd.DataFrame()
    timedirs = os.listdir("postProcessing/sets")
    latest_time = max(timedirs)
    fname_u = "profile_{}_UPrime2Mean.xy".format(z_H)
    fname_k = "profile_{}_turbulenceProperties:kMean.xy".format(z_H)
    data = np.loadtxt(os.path.join("postProcessing", "sets", latest_time,
                      fname_u), unpack=True)
    df["y_R"] = data[0]/R
    df["k_resolved"] = 0.5*(data[1] + data[4] + data[6])
    try:
        data = np.loadtxt(os.path.join("postProcessing", "sets", latest_time,
                          fname_k), unpack=True)
        df["k_modeled"] = data[1]
        df["k_total"] = df.k_modeled + df.k_resolved
    except FileNotFoundError:
        df["k_modeled"] = np.zeros(len(df.y_R))*np.nan
        df["k_total"] = df.k_resolved
    return df


def load_k_map(amount="total"):
    """Load all TKE profiles. Returns a `DataFrame` with `z_H` as the index and
    `y_R` as columns.
    """
    sets_dir = os.path.join("postProcessing", "sets")
    latest_time = max(os.listdir(sets_dir))
    data_dir = os.path.join(sets_dir, latest_time)
    flist = os.listdir(data_dir)
    z_H = []
    for fname in flist:
        if "UPrime2Mean" in fname:
            z_H.append(float(fname.split("_")[1]))
    z_H.sort()
    z_H.reverse()
    k = []
    for z_H_i in z_H:
        dfi = load_k_profile(z_H_i)
        k.append(dfi["k_" + amount].values)
    y_R = dfi.y_R.values
    k = np.array(k).reshape((len(z_H), len(y_R)))
    df = pd.DataFrame(k, index=z_H, columns=y_R)
    return df


def calc_perf(t1=3.0):
    """Calculate mean turbine performance."""
    df = pd.read_csv("postProcessing/turbines/0/turbine.csv")
    df = df[df.time >= t1]
    return {"tsr": df.tsr.mean(), "cp": df.cp.mean(), "cd": df.cd.mean()}


def read_funky_log():
    """Read `log.funkyDoCalc` and parse recovery term averages."""
    with open("log.funkyDoCalc") as f:
        for line in f.readlines():
            try:
                line = line.replace("=", " ")
                line = line.split()
                if line[0] == "planeAverageAdvectionY":
                    y_adv = float(line[-1])
                elif line[0] == "weightedAverage":
                    z_adv = float(line[-1])
                elif line[0] == "planeAverageTurbTrans":
                    turb_trans = float(line[-1])
                elif line[0] == "planeAverageViscTrans":
                    visc_trans = float(line[-1])
                elif line[0] == "planeAveragePressureGradient":
                    pressure_trans = float(line[-1])
            except IndexError:
                pass
    return {"y_adv" : y_adv, "z_adv" : z_adv, "turb_trans" : turb_trans,
            "visc_trans" : visc_trans, "pressure_trans" : pressure_trans}


def load_exp_recovery():
    """Load recovery terms from experimental data. Must be run with IPython."""
    start_dir = os.getcwd()
    exp_dir = os.path.join(os.path.expanduser("~"), "Google Drive", "Research",
                           "Experiments", "RVAT Re dep")
    os.chdir(exp_dir)
    import pyrvatrd.plotting as exppl
    wm = exppl.WakeMap(1.0)
    dUdy = wm.dUdy
    dUdz = wm.dUdz
    tt = wm.ddy_upvp + wm.ddz_upwp
    d2Udy2 = wm.d2Udy2
    d2Udz2 = wm.d2Udz2
    meanu, meanv, meanw = wm.df.mean_u, wm.df.mean_v, wm.df.mean_w
    y_R, z_H = wm.y_R, wm.z_H
    os.chdir(start_dir)
    return {"y_adv": ts.average_over_area(-meanv*dUdy/meanu/U*D, y_R, z_H),
            "z_adv": ts.average_over_area(-meanw*dUdz/meanu/U*D, y_R, z_H),
            "turb_trans": ts.average_over_area(-tt/meanu/U*D, y_R, z_H),
            "visc_trans": ts.average_over_area(nu*(d2Udy2 + d2Udz2)/meanu/U*D,
                                               y_R, z_H),
            "pressure_trans": np.nan}


def load_exp_wake():
    """Load wake measurement data. Must be run with IPython."""
    exp_dir = os.path.join(os.path.expanduser("~"), "Google Drive", "Research",
                           "Experiments", "RVAT Re dep")
    df = pd.read_csv(os.path.join(exp_dir, "Data", "Processed", "Wake-1.0.csv"))
    return df


if __name__ == "__main__":
    pass
