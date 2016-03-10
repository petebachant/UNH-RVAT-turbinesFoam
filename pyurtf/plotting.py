"""Plotting functions."""

from .processing import *
import matplotlib.pyplot as plt


labels = {"y_adv": r"$-V \frac{\partial U}{\partial y}$",
          "z_adv": r"$-W \frac{\partial U}{\partial z}$",
          "turb_trans": r"Turb. trans.",
          "pressure_trans": r"$-\frac{\partial P}{\partial x}$",
          "visc_trans": r"Visc. trans.",
          "rel_vel_mag": "Relative velocity (m/s)",
          "cc": "$C_c$",
          "cm": "$C_m$",
          "cn": "$C_n$",
          "alpha_deg": "Angle of attack (degrees)",
          "meanu": r"$U/U_\infty$",
          "k": r"$k/U_\infty^2$"}


def plot_exp_lines(color="gray", linewidth=2):
    """Plots the outline of the experimental y-z measurement plane"""
    plt.hlines(0.625, -3, 3, linestyles="dashed", colors=color,
               linewidth=linewidth)
    plt.hlines(0.0, -3, 3, linestyles="dashed", colors=color,
               linewidth=linewidth)
    plt.vlines(-3.0, 0.0, 0.625, linestyles="dashed", colors=color,
               linewidth=linewidth)
    plt.vlines(3.0, 0.0, 0.625, linestyles="dashed", colors=color,
               linewidth=linewidth)


def plot_meancontquiv(save=False, show=False,
                      cb_orientation="vertical"):
    """Plot mean contours/quivers of velocity."""
    mean_u = load_vel_map("u")
    mean_v = load_vel_map("v")
    mean_w = load_vel_map("w")
    y_R = np.round(np.asarray(mean_u.columns.values, dtype=float), decimals=4)
    z_H = np.asarray(mean_u.index.values, dtype=float)
    plt.figure(figsize=(7.5, 4.5))
    # Add contours of mean velocity
    cs = plt.contourf(y_R, z_H, mean_u,
                      np.arange(0.15, 1.25, 0.05), cmap=plt.cm.coolwarm)
    if cb_orientation == "horizontal":
        cb = plt.colorbar(cs, shrink=1, extend="both",
                          orientation="horizontal", pad=0.14)
    elif cb_orientation == "vertical":
        cb = plt.colorbar(cs, shrink=1, extend="both",
                          orientation="vertical", pad=0.02)
    cb.set_label(r"$U/U_{\infty}$")
    plt.hold(True)
    # Make quiver plot of v and w velocities
    Q = plt.quiver(y_R, z_H, mean_v, mean_w, width=0.0022,
                   edgecolor="none", scale=3.0)
    plt.xlabel(r"$y/R$")
    plt.ylabel(r"$z/H$")
    if cb_orientation == "horizontal":
        plt.quiverkey(Q, 0.65, 0.26, 0.1, r"$0.1 U_\infty$",
                      labelpos="E",
                      coordinates="figure",
                      fontproperties={"size": "small"})
    elif cb_orientation == "vertical":
        plt.quiverkey(Q, 0.65, 0.07, 0.1, r"$0.1 U_\infty$",
                      labelpos="E",
                      coordinates="figure",
                      fontproperties={"size": "small"})
    plot_turb_lines()
    plot_exp_lines()
    ax = plt.axes()
    ax.set_aspect(2.0)
    plt.yticks(np.around(np.arange(-1.125, 1.126, 0.125), decimals=2))
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        figname = "meancontquiv"
        plt.savefig("figures/" + figname + ".pdf")
        plt.savefig("figures/" + figname + ".png", dpi=300)


def plot_kcont(cb_orientation="vertical", newfig=True, save=False):
    """Plot contours of TKE."""
    k = load_k_map(amount="total")
    y_R = np.round(np.asarray(k.columns.values, dtype=float), decimals=4)
    z_H = np.asarray(k.index.values, dtype=float)
    if newfig:
        plt.figure(figsize=(7.5, 1.9))
    cs = plt.contourf(y_R, z_H, k, 20, cmap=plt.cm.coolwarm,
                      levels=np.linspace(0, 0.09, num=19))
    plt.xlabel(r"$y/R$")
    plt.ylabel(r"$z/H$")
    if cb_orientation == "horizontal":
        cb = plt.colorbar(cs, shrink=1, extend="both",
                          orientation="horizontal", pad=0.3)
    elif cb_orientation == "vertical":
        cb = plt.colorbar(cs, shrink=1, extend="both",
                          orientation="vertical", pad=0.02)
    cb.set_label(r"$k/U_\infty^2$")
    plot_turb_lines(color="black")
    plt.ylim((0, 0.63))
    ax = plt.axes()
    ax.set_aspect(2)
    plt.yticks([0,0.13,0.25,0.38,0.5,0.63])
    plt.tight_layout()
    if save:
        figname = "kcont"
        plt.savefig("figures/" + figname + ".pdf")
        plt.savefig("figures/" + figname + ".png", dpi=300)


def plot_turb_lines(half=False, color="gray"):
    if half:
        plt.hlines(0.5, -1, 1, linestyles="solid", linewidth=2)
        plt.vlines(-1, 0, 0.5, linestyles="solid", linewidth=2)
        plt.vlines(1, 0, 0.5, linestyles="solid", linewidth=2)
    else:
        plt.hlines(0.5, -1, 1, linestyles="solid", colors=color,
                   linewidth=3)
        plt.hlines(-0.5, -1, 1, linestyles="solid", colors=color,
                   linewidth=3)
        plt.vlines(-1, -0.5, 0.5, linestyles="solid", colors=color,
                   linewidth=3)
        plt.vlines(1, -0.5, 0.5, linestyles="solid", colors=color,
                   linewidth=3)


def plot_cp(ax=None, angle0=540.0, save=False):
    df = pd.read_csv("postProcessing/turbines/0/turbine.csv")
    df = df.drop_duplicates("time", keep="last")
    if df.angle_deg.max() < angle0:
        angle0 = 0.0
    print("Performance from {:.1f}--{:.1f} degrees:".format(angle0,
          df.angle_deg.max()))
    print("Mean TSR = {:.2f}".format(df.tsr[df.angle_deg >= angle0].mean()))
    print("Mean C_P = {:.2f}".format(df.cp[df.angle_deg >= angle0].mean()))
    print("Mean C_D = {:.2f}".format(df.cd[df.angle_deg >= angle0].mean()))
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(df.angle_deg, df.cp)
    ax.set_xlabel("Azimuthal angle (degrees)")
    ax.set_ylabel("$C_P$")
    ymin, ymax = None, None
    if df.cp.max() > 2 or df.cp.min() < -0.5:
        ax.set_ylim((-0.1, 0.7))
    plt.tight_layout()
    if save:
        figname = "cp"
        plt.savefig("figures/" + figname + ".pdf")
        plt.savefig("figures/" + figname + ".png", dpi=300)


def plot_perf_curves(exp=False, save=False):
    """Plot performance curves."""
    df = pd.read_csv("processed/tsr_sweep.csv")
    if exp:
        df_exp = pd.read_csv("https://raw.githubusercontent.com/UNH-CORE/"
                             "RVAT-Re-dep/master/Data/Processed/Perf-1.0.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.5), nrows=1, ncols=2)
    ax[0].plot(df.tsr, df.cp, "-o", label="ALM")
    ax[0].set_ylabel(r"$C_P$")
    ax[1].plot(df.tsr, df.cd, "-o", label="ALM")
    ax[1].set_ylabel(r"$C_D$")
    for a in ax:
        a.set_xlabel(r"$\lambda$")
    if exp:
        ax[0].plot(df_exp.mean_tsr, df_exp.mean_cp, "^", label="Exp.",
                   markerfacecolor="none")
        ax[1].plot(df_exp.mean_tsr, df_exp.mean_cd, "^", label="Exp.",
                   markerfacecolor="none")
        ax[1].legend(loc="lower right")
    ax[1].set_ylim((0, None))
    fig.tight_layout()
    if save:
        figname = "perf-curves"
        plt.savefig("figures/" + figname + ".pdf")
        plt.savefig("figures/" + figname + ".png", dpi=300)


def plot_al_perf(name="blade1", theta1=0, theta2=None, remove_offset=False,
                 quantities=["alpha", "rel_vel_mag", "cc"]):
    if isinstance(quantities, str):
        quantities = [quantities]
    df_turb = pd.read_csv("postProcessing/turbines/0/turbine.csv")
    df_turb = df_turb.drop_duplicates("time", keep="last")
    df = pd.read_csv("postProcessing/actuatorLines/0/{}.csv".format(name))
    df = df.drop_duplicates("time", keep="last")
    df["angle_deg"] = df_turb.angle_deg
    df["cc"] = df.cl*np.sin(np.deg2rad(df.alpha_deg)) \
             - df.cd*np.cos(np.deg2rad(df.alpha_deg))
    df["cn"] = df.cl*np.cos(np.deg2rad(df.alpha_deg)) \
             - df.cd*np.sin(np.deg2rad(df.alpha_deg))
    df = df[df.angle_deg >= theta1]
    if theta2 is not None:
        df = df[df.angle_deg <= theta2]
    if remove_offset:
        offset = df.angle_deg.values[0]
        df.angle_deg -= offset
        theta1 -= offset
        if theta2 is not None:
            theta2 -= offset
    if len(quantities) > 1:
        fig, ax = plt.subplots(figsize=(7.5, 3.5), nrows=1,
                               ncols=len(quantities))
    else:
        fig, ax = plt.subplots()
        ax = [ax]
    for a, q in zip(ax, quantities):
        if q == "alpha":
            a.plot(df.angle_deg, df.alpha_deg, label="Actual")
            a.plot(df.angle_deg, df.alpha_geom_deg, label="Geometric")
            a.set_ylabel("Angle of attack (degrees)")
            a.legend(loc="best")
        else:
            a.plot(df.angle_deg, df[q])
            a.set_ylabel(labels[q])
        a.set_xlim((theta1, theta2))
        a.set_xlabel(r"$\theta$ (degrees)")
    fig.tight_layout()


def plot_blade_perf(theta1=0, theta2=None, remove_offset=False, save=False,
                    **kwargs):
    plot_al_perf("blade1", theta1, theta2, remove_offset, **kwargs)
    if save:
        figname = "blade-perf"
        plt.savefig("figures/" + figname + ".pdf")
        plt.savefig("figures/" + figname + ".png", dpi=300)


def plot_strut_perf(save=False, **kwargs):
    plot_al_perf("strut1", **kwargs)
    if save:
        figname = "strut-perf"
        plt.savefig("figures/" + figname + ".pdf")
        plt.savefig("figures/" + figname + ".png", dpi=300)


def make_recovery_bar_chart(ax=None, save=False):
    """Create a bar chart with x-labels for each recovery term and 5 different
    bars per term, corresponding to each CFD case and the experimental data.
    """
    A_exp = 3.0*0.625
    df = pd.DataFrame(index=["y_adv", "z_adv", "turb_trans", "pressure_trans",
                             "visc_trans"])
    df["ALM"] = pd.Series(read_funky_log(), name="ALM")*A_c
    # Results from blade-resolved CFD, added manually
    df["Blade-resolved SST"] = pd.Series({"pressure_trans": -0.02528183721,
                                          "turb_trans": 0.02868203482,
                                          "visc_trans": 1.031373523e-06,
                                          "y_adv": -0.001940439635,
                                          "z_adv": 0.009998671966})*A_c
    df["Blade-resolved SA"] = pd.Series({"pressure_trans": -0.0048568805,
                                         "turb_trans": 0.00515128319,
                                         "visc_trans": 1.073326703e-06,
                                         "y_adv": -0.006335581968,
                                         "z_adv": 0.004210650781})*A_c
    df["Exp."] = pd.Series(load_exp_recovery(), name="Exp.")*A_exp
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    else:
        fig = None
    df.index = [labels[k] for k in df.index.values]
    df.plot(kind="bar", ax=ax, rot=0)
    ax.hlines(0, -0.5, len(df) + 0.5, color="gray", lw=1)
    ax.set_ylabel(r"$\frac{U \, \mathrm{ transport} \times A_c}"
                  "{UU_\infty D^{-1}}$")
    if fig is not None:
        fig.tight_layout()
    if save and fig is not None:
        figname = "recovery-bar-chart"
        fig.savefig("figures/{}.pdf".format(figname))
        fig.savefig("figures/{}.png".format(figname), dpi=300)


def plot_wake_profiles(z_H=1e-5, exp=False, save=False):
    """Plot profiles of mean streamwise velocity and TKE."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5), nrows=1, ncols=2)
    dfu = load_u_profile(z_H=z_H)
    dfk = load_k_profile(z_H=z_H)
    ax[0].plot(dfu.y_R, dfu.u, "-o", label="ALM")
    ax[0].set_ylabel(labels["meanu"])
    ax[1].plot(dfk.y_R, dfk.k_total, "-o", label="ALM")
    ax[1].set_ylabel(labels["k"])
    for a in ax:
        a.set_xlabel("$y/R$")
    if exp:
        df = load_exp_wake()
        df = df[df.z_H == np.round(z_H, decimals=3)]
        ax[0].plot(df.y_R, df.mean_u, "^", markerfacecolor="none", label="Exp.")
        ax[1].plot(df.y_R, df.k, "^", markerfacecolor="none", label="Exp.")
        ax[0].legend(loc="lower left")
    fig.tight_layout()
    if save:
        savefig(fig, "wake-profiles")


def plot_verification(save=False):
    """Plot spatial and temporal grid dependence."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5), ncols=2)
    dt_fpath = "processed/dt_sweep.csv"
    nx_fpath = "processed/nx_sweep.csv"
    if os.path.isfile(dt_fpath):
        df_dt = pd.read_csv(dt_fpath)
        df_dt["steps_per_rev"] = 1.0/(df_dt.tsr/R*U/(2.0*np.pi))/df_dt.dt
        ax[0].plot(df_dt.steps_per_rev, df_dt.cp, "-o")
        ax[0].set_xlabel("Time steps per rev.")
        ax[0].set_ylabel(r"$C_P$")
    if os.path.isfile(nx_fpath):
        df_nx = pd.read_csv(nx_fpath)
        ax[1].plot(df_nx.nx, df_nx.cp, "o")
        ax[1].set_xlabel(r"$N_x$")
        ax[1].set_ylabel(r"$C_P$")
    fig.tight_layout()
    if save:
        savefig(fig, "verification")


def savefig(fig, figname):
    """Save figure in `figures` directory as PDF and PNG."""
    fig.savefig("figures/{}.pdf".format(figname))
    fig.savefig("figures/{}.png".format(figname), dpi=300)
