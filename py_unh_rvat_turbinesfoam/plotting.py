"""
Plotting functions
"""

from .processing import *
import matplotlib.pyplot as plt


labels = {"y_adv": r"$-V \frac{\partial U}{\partial y}$",
          "z_adv": r"$-W \frac{\partial U}{\partial z}$",
          "turb_trans": r"Turb. trans.",
          "pressure_trans": r"$-\frac{\partial P}{\partial x}$",
          "visc_trans": r"Visc. trans."}


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
        figname = "rvat-alm-meanconquiv"
        plt.savefig("figures/" + figname + ".pdf")
        plt.savefig("figures/" + figname + ".png", dpi=300)


def plot_kcont(cb_orientation="vertical", newfig=True, save=False):
    """Plot contours of TKE."""
    k = load_k_map()
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
        figname = "rvat-alm-kcont"
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


def plot_cp(angle0=540.0):
    df = pd.read_csv("postProcessing/turbines/0/turbine.csv")
    df = df.drop_duplicates("time", take_last=True)
    if df.angle_deg.max() < angle0:
        angle0 = 0.0
    print("Performance from {:.1f}--{:.1f} degrees:".format(angle0,
          df.angle_deg.max()))
    print("Mean TSR = {:.2f}".format(df.tsr[df.angle_deg >= angle0].mean()))
    print("Mean C_P = {:.2f}".format(df.cp[df.angle_deg >= angle0].mean()))
    print("Mean C_D = {:.2f}".format(df.cd[df.angle_deg >= angle0].mean()))
    plt.plot(df.angle_deg, df.cp)
    plt.xlabel("Azimuthal angle (degrees)")
    plt.ylabel("$C_P$")
    plt.tight_layout()


def plot_perf_curves(exp=False):
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
    fig.tight_layout()


def plot_al_perf(name="blade1", theta1=0, theta2=None, remove_offset=False):
    df_turb = pd.read_csv("postProcessing/turbines/0/turbine.csv")
    df_turb = df_turb.drop_duplicates("time", take_last=True)
    df = pd.read_csv("postProcessing/actuatorLines/0/{}.csv".format(name))
    df = df.drop_duplicates("time", take_last=True)
    df["angle_deg"] = df_turb.angle_deg
    df["ct"] = df.cl*np.sin(np.deg2rad(df.alpha_deg)) \
             - df.cd*np.cos(np.deg2rad(df.alpha_deg))
    df = df[df.angle_deg >= theta1]
    if theta2 is not None:
        df = df[df.angle_deg <= theta2]
    if remove_offset:
        offset = df.angle_deg.values[0]
        df.angle_deg -= offset
        theta1 -= offset
        if theta2 is not None:
            theta2 -= offset
    fig, ax = plt.subplots(figsize=(7.5, 3.5), nrows=1, ncols=3)
    ax[0].plot(df.angle_deg, df.alpha_deg, label="Actual")
    ax[0].plot(df.angle_deg, df.alpha_geom_deg, label="Geometric")
    ax[0].set_ylabel("Angle of attack (degrees)")
    ax[0].legend(loc="best")
    ax[1].plot(df.angle_deg, df.rel_vel_mag)
    ax[1].set_ylabel("Relative velocity (m/s)")
    ax[2].plot(df.angle_deg, df.ct)
    ax[2].set_ylabel("$C_T$")
    for a in ax:
        a.set_xlim((theta1, theta2))
        a.set_xlabel(r"$\theta$ (degrees)")
    fig.tight_layout()


def plot_blade_perf(theta1=360, theta2=720, remove_offset=False):
    plot_al_perf("blade1", theta1, theta2, remove_offset)


def plot_strut_perf():
    plot_al_perf("strut1")


def make_recovery_bar_chart(ax=None, save=False):
    """
    Create a bar chart with x-labels for each recovery term and 5 different
    bars per term, corresponding to each CFD case and the experimental data.
    """
    A_exp = 3.0*0.625
    df = pd.DataFrame(index=["y_adv", "z_adv", "turb_trans", "pressure_trans",
                             "visc_trans"])
    df["ALM"] = pd.Series(read_funky_log(), name="ALM")*A_c
    df["Exp."] = pd.Series(load_exp_recovery(), name="Exp.")*A_exp
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5))
    else:
        fig = None
    df.index = [labels[k] for k in df.index.values]
    df.plot(kind="bar", ax=ax, rot=0)
    ax.hlines(0, -0.5, len(df) + 0.5, color="gray")
    ax.set_ylabel(r"$\frac{U \, \mathrm{ transport} \times A_c}"
                  "{UU_\infty D^{-1}}$")
    if fig is not None:
        fig.tight_layout()
    if save and fig is not None:
        fig.savefig("figures/rvat-alm-recovery-bar-chart.pdf")
        fig.savefig("figures/rvat-alm-recovery-bar-chart.png", dpi=300)
