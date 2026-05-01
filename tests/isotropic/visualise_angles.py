#!/usr/bin/env python3
"""
Visualise the SO(d) angle distributions saved by test_so_angle_sampler_save.

Usage:
    python visualise_angles.py [path/to/so_angle_samples.h5]

If no path is given a file-open dialog appears.
Requires: h5py, numpy, matplotlib, tkinter (stdlib).
"""

import sys
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required:  pip install h5py")

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError:
    sys.exit("matplotlib is required:  pip install matplotlib")

import tkinter as tk
from tkinter import ttk, filedialog


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(h5path: str):
    """
    Returns:
        groups -- list of dicts, sorted by alpha:
            {
              "alpha":   float,
              "tangent": ndarray [N, m],
              "hmc":     ndarray [N, m] | None,
            }
        d -- SO(d) dimension
        m -- number of angles
    """
    groups = []
    with h5py.File(h5path, "r") as f:
        d = int(f.attrs.get("d", 0))
        m = int(f.attrs.get("m", 0))
        for key in f.keys():
            grp   = f[key]
            alpha = float(grp.attrs["alpha"])
            tangent = grp["angles_tangent"][:]
            hmc     = grp["angles_hmc"][:] if "angles_hmc" in grp else None
            if m == 0:
                m = tangent.shape[1]
            groups.append({"alpha": alpha, "tangent": tangent, "hmc": hmc})

    groups.sort(key=lambda g: g["alpha"])
    return groups, d, m


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_alpha(alpha: float) -> str:
    if alpha >= 1e6:
        return f"α = {alpha:.2e}"
    elif alpha >= 10:
        return f"α = {int(alpha)}"
    else:
        return f"α = {alpha}"


def _plot_hist(ax, samples, label, color):
    ax.hist(
        samples,
        bins=60,
        density=True,
        color=color,
        edgecolor="none",
        alpha=0.80,
        label=label,
    )
    ax.set_xlim(samples.min(), samples.max())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

VIEW_HMC     = "hmc"
VIEW_TANGENT = "tangent"
VIEW_BOTH    = "both"


class AngleViewer(tk.Tk):
    def __init__(self, h5path: str):
        super().__init__()
        self.title("SO(d) Angle Distribution Viewer")
        self.resizable(True, True)

        self.groups, self.d, self.m = load_data(h5path)

        # ------------------------------------------------------------------
        # Control bar
        # ------------------------------------------------------------------
        ctrl = tk.Frame(self, pady=4, padx=8)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        # Angle selector
        angle_frame = tk.LabelFrame(ctrl, text="Angle", padx=4, pady=2)
        angle_frame.pack(side=tk.LEFT, padx=(0, 12))

        self.angle_var = tk.IntVar(value=0)
        for i in range(self.m):
            ttk.Radiobutton(
                angle_frame,
                text=f"θ{i + 1}",
                variable=self.angle_var,
                value=i,
                command=self.redraw,
            ).pack(side=tk.LEFT, padx=4)

        # Alpha selector
        alpha_frame = tk.LabelFrame(ctrl, text="Alpha", padx=4, pady=2)
        alpha_frame.pack(side=tk.LEFT, padx=(0, 12))

        self.alpha_var = tk.IntVar(value=0)
        for i, g in enumerate(self.groups):
            ttk.Radiobutton(
                alpha_frame,
                text=_fmt_alpha(g["alpha"]),
                variable=self.alpha_var,
                value=i,
                command=self._on_alpha_change,
            ).pack(side=tk.LEFT, padx=4)

        # View selector
        view_frame = tk.LabelFrame(ctrl, text="View", padx=4, pady=2)
        view_frame.pack(side=tk.LEFT)

        self.view_var = tk.StringVar(value=VIEW_TANGENT)
        self._view_btns = {}
        for val, label in [(VIEW_HMC, "HMC"), (VIEW_TANGENT, "Tangent"), (VIEW_BOTH, "Both")]:
            btn = ttk.Radiobutton(
                view_frame,
                text=label,
                variable=self.view_var,
                value=val,
                command=self.redraw,
            )
            btn.pack(side=tk.LEFT, padx=4)
            self._view_btns[val] = btn

        # ------------------------------------------------------------------
        # Matplotlib figure
        # ------------------------------------------------------------------
        self.fig = Figure(figsize=(8, 4), tight_layout=True)

        frame = tk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, frame)
        toolbar.update()

        self._on_alpha_change()

    def _on_alpha_change(self):
        """Disable HMC / Both when the selected alpha has no HMC data."""
        g       = self.groups[self.alpha_var.get()]
        has_hmc = g["hmc"] is not None

        state_hmc  = tk.NORMAL if has_hmc else tk.DISABLED
        self._view_btns[VIEW_HMC ].config(state=state_hmc)
        self._view_btns[VIEW_BOTH].config(state=state_hmc)

        # If the current view requires HMC but none is available, fall back.
        if not has_hmc and self.view_var.get() in (VIEW_HMC, VIEW_BOTH):
            self.view_var.set(VIEW_TANGENT)

        self.redraw()

    def redraw(self):
        angle_idx = self.angle_var.get()
        g         = self.groups[self.alpha_var.get()]
        view      = self.view_var.get()

        self.fig.clear()

        if view == VIEW_BOTH:
            axes = [self.fig.add_subplot(1, 2, 1),
                    self.fig.add_subplot(1, 2, 2)]
            datasets = [(g["hmc"],     "HMC",     "steelblue"),
                        (g["tangent"], "Tangent",  "darkorange")]
        elif view == VIEW_HMC:
            axes     = [self.fig.add_subplot(1, 1, 1)]
            datasets = [(g["hmc"], "HMC", "steelblue")]
        else:
            axes     = [self.fig.add_subplot(1, 1, 1)]
            datasets = [(g["tangent"], "Tangent", "darkorange")]

        alpha_str = _fmt_alpha(g["alpha"])
        angle_lbl = f"θ{angle_idx + 1}  (radians)"

        for ax, (data, label, color) in zip(axes, datasets):
            samples = data[:, angle_idx]
            _plot_hist(ax, samples, label, color)
            ax.set_xlabel(angle_lbl, fontsize=10)
            ax.set_ylabel("density", fontsize=10)
            ax.set_title(
                f"SO({self.d})  —  θ{angle_idx + 1},  {alpha_str}  [{label}]",
                fontsize=11,
            )

        self.canvas.draw()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        h5path = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()
        h5path = filedialog.askopenfilename(
            title="Open HDF5 angle file",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*")],
        )
        root.destroy()
        if not h5path:
            print("No file selected — exiting.")
            sys.exit(0)

    app = AngleViewer(h5path)
    app.mainloop()


if __name__ == "__main__":
    main()
