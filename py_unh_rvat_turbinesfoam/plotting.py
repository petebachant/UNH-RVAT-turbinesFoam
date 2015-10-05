"""
Plotting functions
"""

from .processing import *


def plot_meancontquiv():
    u = load_vel_map("u")
    v = load_vel_map("v")
    w = load_vel_map("w")
    y_R = np.round(np.asarray(u.columns.values, dtype=float), decimals=4)
    z_H = np.asarray(u.index.values, dtype=float)
    plt.figure(figsize=(7, 6.8))
    # Add contours of mean velocity
    cs = plt.contourf(y_R, z_H, u, 20, cmap=plt.cm.coolwarm)
    cb = plt.colorbar(cs, shrink=1, extend='both',
                      orientation='horizontal', pad=0.12)
                      #ticks=np.round(np.linspace(0.44, 1.12, 10), decimals=2))
    cb.set_label(r'$U/U_{\infty}$')
    plt.hold(True)
    # Make quiver plot of v and w velocities
    Q = plt.quiver(y_R, z_H, v, w, angles='xy', width=0.0022,
                   edgecolor="none", scale=3.0)
    plt.xlabel(r'$y/R$')
    plt.ylabel(r'$z/H$')
    #plt.ylim(-0.2, 0.78)
    #plt.xlim(-3.2, 3.2)
    plt.xlim(-3.66, 3.66)
    plt.ylim(-1.22, 1.22)
    plt.quiverkey(Q, 0.8, 0.22, 0.1, r'$0.1 U_\infty$',
               labelpos='E',
               coordinates='figure',
               fontproperties={'size': 'small'})
    plt.hlines(0.5, -1, 1, linestyles='solid', colors='gray',
               linewidth=3)
    plt.hlines(-0.5, -1, 1, linestyles='solid', colors='gray',
               linewidth=3)
    plt.vlines(-1, -0.5, 0.5, linestyles='solid', colors='gray',
               linewidth=3)
    plt.vlines(1, -0.5, 0.5, linestyles='solid', colors='gray',
               linewidth=3)
    ax = plt.axes()
    ax.set_aspect(2.0)
    plt.tight_layout()


def plot_kcont(cb_orientation="vertical", newfig=True):
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