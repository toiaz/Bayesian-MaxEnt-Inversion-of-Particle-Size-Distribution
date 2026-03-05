#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI module for Bayesian-MaxEnt PSD inversion v3.2.1.
"""

import dataclasses
import os
import sys
import ctypes
import math
import importlib.util
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# High-DPI scaling
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

Array = np.ndarray

_CORE_VERSIONS = {
    "v3.2.1": "MaxEnt_core_v3.2.1.py",
}
if "_DEFAULT_CORE_VERSION" not in globals():
    _DEFAULT_CORE_VERSION = "v3.2.1"

_loaded_core_version = None
_maxent_core = None

STRUCTURE_FACTOR_MODELS = None
SF_REQUIRED_PARAMS = None
MaxEntConfig = None
OuterSearchConfig = None
PSDInversionConfig = None
invert_psd = None


def _load_core(version: str) -> None:
    global _loaded_core_version, _maxent_core
    global STRUCTURE_FACTOR_MODELS, SF_REQUIRED_PARAMS
    global MaxEntConfig, OuterSearchConfig, PSDInversionConfig, invert_psd

    filename = _CORE_VERSIONS.get(version)
    if filename is None:
        raise ValueError(f"Unknown core version: {version}. Available: {list(_CORE_VERSIONS)}")

    core_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(core_path):
        raise FileNotFoundError(f"Core file not found: {core_path}")

    code_dir = os.path.dirname(core_path)
    os.environ["MAXENT_CORE_FILE"] = core_path
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    mod_name = "MaxEnt_core"
    spec = importlib.util.spec_from_file_location(mod_name, core_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    _maxent_core = mod
    _loaded_core_version = version
    STRUCTURE_FACTOR_MODELS = mod.STRUCTURE_FACTOR_MODELS
    SF_REQUIRED_PARAMS      = mod.SF_REQUIRED_PARAMS
    MaxEntConfig            = mod.MaxEntConfig
    OuterSearchConfig       = mod.OuterSearchConfig
    PSDInversionConfig      = mod.PSDInversionConfig
    invert_psd              = mod.invert_psd


_load_core(_DEFAULT_CORE_VERSION)


# ---------------------------------------------------------------------
# Utility: reading / saving
# ---------------------------------------------------------------------
def load_sas_data(path, q_col=0, I_col=1, sigma_col=2, delimiter=None, skiprows=0):
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    q = np.asarray(data[:, q_col], dtype=float)
    I = np.asarray(data[:, I_col], dtype=float)
    sigma = None
    if sigma_col is not None and sigma_col < data.shape[1]:
        sigma = np.asarray(data[:, sigma_col], dtype=float)
    return q, I, sigma


def load_counts_data(path, q_col=0, counts_col=1, scale_col=2, delimiter=None, skiprows=0):
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
    q = np.asarray(data[:, q_col], dtype=float)
    counts = np.asarray(data[:, counts_col], dtype=float)
    scale_i = None
    if scale_col is not None and scale_col < data.shape[1]:
        scale_i = np.asarray(data[:, scale_col], dtype=float)
    return q, counts, scale_i


def save_psd_csv(path, edges, centers, psd_bins):
    r_low = edges[:-1]
    r_high = edges[1:]
    w = r_high - r_low
    w_log = np.log(r_high / np.maximum(r_low, 1e-300))
    psd_density = psd_bins / np.maximum(w, 1e-300)
    psd_density_logr = psd_bins / np.maximum(w_log, 1e-300)
    data = np.column_stack([r_low, r_high, centers, psd_bins, psd_density, psd_density_logr])
    header = "r_low,r_high,r_center,psd_bin_weight,psd_density_per_r,psd_density_per_log_r"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------
class MaxEntApp:
    def __init__(self, root):
        self.root = root
        self.root.resizable(True, True)
        self.style_setup()
        self.create_widgets()
        self._update_title()

    def style_setup(self):
        self.font_norm = ("Segoe UI", 10)
        self.font_bold = ("Segoe UI", 10, "bold")
        self.pad_x = 8
        self.pad_y = 5

    def _update_title(self):
        ver = self.core_version.get()
        self.root.title(f"MaxEnt PSD Inversion  [Core {ver}]")

    def _on_core_version_change(self, event=None):
        ver = self.core_version.get()
        try:
            _load_core(ver)
            self.sf_combo['values'] = list(STRUCTURE_FACTOR_MODELS.keys())
            self._update_title()
            self.update_sf_params()
        except Exception as e:
            messagebox.showerror("Core Load Error", str(e))

    def add_label(self, parent, text, row, col=0, font=None):
        if font is None:
            font = self.font_norm
        lbl = ttk.Label(parent, text=text, font=font)
        lbl.grid(row=row, column=col, sticky="e", padx=self.pad_x, pady=self.pad_y)
        return lbl

    def add_entry(self, parent, default, row, col=1, width=20):
        var = tk.StringVar(value=str(default))
        ent = ttk.Entry(parent, textvariable=var, width=width, font=self.font_norm)
        ent.grid(row=row, column=col, sticky="w", padx=self.pad_x, pady=self.pad_y)
        return var

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        main = ttk.Frame(self.notebook, padding=15)
        self.notebook.add(main, text="Inversion Calculation")

        row = 0
        ttk.Label(main, text="Bayesian MaxEnt PSD Inversion",
                  font=("Segoe UI", 12, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(0, 5), sticky="w")

        self.core_version = tk.StringVar(value=_loaded_core_version or _DEFAULT_CORE_VERSION)
        ttk.Label(main, text="v3.2.1 (Profile-α + Two-Stage Grid + Smooth Prior)",
                  font=("Segoe UI", 10, "italic")).grid(
            row=row, column=1, columnspan=2, sticky="e", padx=self.pad_x, pady=(0, 5))
        row += 1

        # 1) Data
        ttk.Label(main, text="1. Input Data (Measured)", font=self.font_bold).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=5)
        row += 1

        self.add_label(main, "Data File (Q, I, Sigma):", row)
        self.file_path = tk.StringVar()
        ttk.Entry(main, textvariable=self.file_path, width=30, font=self.font_norm).grid(
            row=row, column=1, sticky="w", padx=self.pad_x, pady=self.pad_y)
        ttk.Button(main, text="Browse", command=self.on_browse).grid(
            row=row, column=2, padx=self.pad_x, pady=self.pad_y)
        row += 1

        self.add_label(main, "Output Root:", row)
        default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PSD_results")
        self.output_root = tk.StringVar(value=default_out)
        ttk.Entry(main, textvariable=self.output_root, width=30, font=self.font_norm).grid(
            row=row, column=1, sticky="w", padx=self.pad_x, pady=self.pad_y)
        ttk.Button(main, text="Browse", command=self.on_browse_output).grid(
            row=row, column=2, padx=self.pad_x, pady=self.pad_y)
        row += 1

        # 2) Grid and structure factor
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1
        ttk.Label(main, text="2. Physical Parameters & structure factor",
                  font=self.font_bold).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=5)
        row += 1

        self.add_label(main, "R min (Å):", row)
        self.r_min = self.add_entry(main, "5.0", row)
        self.auto_r = tk.BooleanVar(value=False)
        ttk.Checkbutton(main, text="Auto (2.5/Q)", variable=self.auto_r).grid(
            row=row, column=2, sticky="w", padx=2)
        row += 1

        self.add_label(main, "R max (Å):", row)
        self.r_max = self.add_entry(main, "15000.0", row)
        row += 1

        self.add_label(main, "N bins:", row)
        self.n_r = self.add_entry(main, "90", row)
        row += 1

        self.add_label(main, "Structure Factor:", row)
        self.sf_model = tk.StringVar(value="hard_sphere_py")
        self.sf_combo = ttk.Combobox(main, textvariable=self.sf_model,
                                     values=list(STRUCTURE_FACTOR_MODELS.keys()),
                                     state="readonly", font=self.font_norm)
        self.sf_combo.grid(row=row, column=1, sticky="w", padx=self.pad_x, pady=self.pad_y)
        self.sf_combo.bind("<<ComboboxSelected>>", self.update_sf_params)
        row += 1

        self.sf_frame = ttk.Frame(main)
        self.sf_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=20)
        self.sf_vars = {}
        row += 1

        # 3) Model options
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1
        ttk.Label(main, text="3. Model Options", font=self.font_bold).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=5)
        row += 1

        self.inc_bg = tk.BooleanVar(value=True)
        ttk.Checkbutton(main, text="Include Background", variable=self.inc_bg).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=self.pad_x)
        row += 1

        self.inc_pl = tk.BooleanVar(value=True)
        ttk.Checkbutton(main, text="Include Powerlaw", variable=self.inc_pl).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=self.pad_x)
        self.add_label(main, "Exponent:", row, col=1)
        self.pl_exp = self.add_entry(main, "4.0", row, col=2, width=8)
        row += 1

        self.add_label(main, "Likelihood:", row)
        self.likelihood = tk.StringVar(value="poisson")
        self.likelihood_combo = ttk.Combobox(
            main, textvariable=self.likelihood, values=["gaussian", "poisson"],
            state="readonly", width=10)
        self.likelihood_combo.grid(row=row, column=1, sticky="w",
                                   padx=self.pad_x, pady=self.pad_y)
        self.likelihood_combo.bind("<<ComboboxSelected>>", self.on_likelihood_change)
        row += 1

        self.add_label(main, "P. Scale s0 (optional):", row)
        self.poisson_scale_str = self.add_entry(main, "", row, col=1, width=10)
        ttk.Label(main, text="(Only for Poisson)", font=self.font_norm).grid(
            row=row, column=2, sticky="w")
        row += 1

        self.poisson_use_counts = tk.BooleanVar(value=False)
        self.poisson_counts_chk = ttk.Checkbutton(
            main,
            text="Poisson counts-mode: file is (Q, counts[, scale_i]), use s0 to convert",
            variable=self.poisson_use_counts)
        self.poisson_counts_chk.grid(row=row, column=0, columnspan=3, sticky="w",
                                     padx=self.pad_x, pady=(0, self.pad_y))
        row += 1

        self.add_label(main, "Outer Search:", row)
        self.outer_mode = tk.StringVar(value="grid_search")
        ttk.Combobox(main, textvariable=self.outer_mode,
                     values=["none", "grid_search"],
                     state="readonly", width=15).grid(
            row=row, column=1, sticky="w", padx=self.pad_x, pady=self.pad_y)
        row += 1

        # Grid search controls (no log_alpha in outer bounds)
        self.add_label(main, "Coarse Grid (per Dim):", row)
        self.grid_points = self.add_entry(main, "20", row, col=1, width=15)
        row += 1

        self.add_label(main, "Grid Refine Top K:", row)
        self.grid_top_k = self.add_entry(main, "3", row, col=1, width=15)
        row += 1

        # Two-stage grid controls
        self.two_stage = tk.BooleanVar(value=True)
        ttk.Checkbutton(main, text="Two-Stage Grid (Coarse→Fine)",
                        variable=self.two_stage).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=self.pad_x)
        row += 1

        self.add_label(main, "Fine Grid (per Dim):", row)
        self.fine_grid_points = self.add_entry(main, "15", row, col=1, width=15)
        row += 1

        self.add_label(main, "Fine Region Factor:", row)
        self.fine_grid_factor = self.add_entry(main, "0.25", row, col=1, width=15)
        row += 1

        self.add_label(main, "Refine X Tol:", row)
        self.refine_xtol = self.add_entry(main, "1e-5", row, col=1, width=15)
        row += 1

        self.add_label(main, "Refine F Tol:", row)
        self.refine_ftol = self.add_entry(main, "1e-5", row, col=1, width=15)
        row += 1

        self.add_label(main, "Tolerance (Chi2 x):", row)
        self.chi2_multiplier = self.add_entry(main, "1.0", row, col=1, width=10)
        row += 1

        self.add_label(main, "Alpha Tol (log α):", row)
        self.alpha_tol = self.add_entry(main, "1e-3", row, col=1, width=10)
        row += 1

        self.add_label(main, "PSD Smooth λ:", row)
        self.smooth_lambda = self.add_entry(main, "0.1", row, col=1, width=10)
        row += 1

        self.add_label(main, "Alpha Min:", row)
        self.alpha_min = self.add_entry(main, "1e-6", row, col=1, width=10)
        self.add_label(main, "Alpha Max:", row, col=2)
        self.alpha_max = self.add_entry(main, "1e6", row, col=3, width=10)
        row += 1

        self.add_label(main, "Max Bisect Iter:", row)
        self.max_bisect_iter = self.add_entry(main, "40", row, col=1, width=10)
        self.fine_warm_start = tk.BooleanVar(value=False)
        ttk.Checkbutton(main, text="Fine Grid Warm Start", variable=self.fine_warm_start).grid(
            row=row, column=2, columnspan=2, sticky="w", padx=self.pad_x)
        row += 1

        self.add_label(main, "L-BFGS MaxIter:", row)
        self.opt_maxiter = self.add_entry(main, "2000", row, col=1, width=10)
        self.add_label(main, "opt fTol:", row, col=2)
        self.opt_ftol = self.add_entry(main, "1e-10", row, col=3, width=10)
        row += 1

        self.add_label(main, "opt gTol:", row)
        self.opt_gtol = self.add_entry(main, "1e-8", row, col=1, width=10)
        self.add_label(main, "eps_I:", row, col=2)
        self.eps_I = self.add_entry(main, "1e-10", row, col=3, width=10)
        row += 1

        self.add_label(main, "u Min:", row)
        self.u_min = self.add_entry(main, "-60", row, col=1, width=10)
        self.add_label(main, "u Max:", row, col=2)
        self.u_max = self.add_entry(main, "60", row, col=3, width=10)
        row += 1

        self.add_label(main, "Poisson Clip Min:", row)
        self.poisson_clip_min = self.add_entry(main, "1e-12", row, col=1, width=10)
        self.add_label(main, "Poisson Clip Max:", row, col=2)
        self.poisson_clip_max = self.add_entry(main, "1e12", row, col=3, width=10)
        row += 1

        self.add_label(main, "Sigma Rel Floor:", row)
        self.sigma_rel_floor = self.add_entry(main, "0.0", row, col=1, width=10)
        self.add_label(main, "Sigma Abs Floor:", row, col=2)
        self.sigma_abs_floor = self.add_entry(main, "0.0", row, col=3, width=10)
        row += 1

        # 4) Absolute porosity
        ttk.Separator(main, orient="horizontal").grid(
            row=row, column=0, columnspan=4, sticky="ew", pady=10)
        row += 1
        ttk.Label(main, text="4. Absolute Porosity & Volume",
                  font=self.font_bold).grid(
            row=row, column=0, columnspan=4, sticky="w", pady=5)
        row += 1

        self.add_label(main, "SLD Matrix (10⁻⁶ Å⁻²):", row)
        self.sld_matrix = self.add_entry(main, "3.98", row, col=1, width=15)
        row += 1

        self.add_label(main, "SLD Pore (10⁻⁶ Å⁻²):", row)
        self.sld_pore = self.add_entry(main, "0.0", row, col=1, width=15)
        row += 1

        self.add_label(main, "Sample Thickness (mm):", row)
        self.thickness = self.add_entry(main, "1.0", row, col=1, width=15)
        row += 1

        self.add_label(main, "Beam Spot Area (mm²):", row)
        self.beam_area = self.add_entry(main, "10.0", row, col=1, width=15)
        row += 1

        self.add_label(main, "Calc R Min (Å):", row)
        self.calc_r_min = self.add_entry(main, "10.0", row, col=1, width=15)
        row += 1

        self.add_label(main, "Calc R Max (Å):", row)
        self.calc_r_max = self.add_entry(main, "15000.0", row, col=1, width=15)
        row += 1

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=row, column=0, columnspan=4, pady=20)
        ttk.Button(btn_frame, text="Start Calculation", command=self.on_run).pack(
            side="left", padx=10)
        ttk.Button(btn_frame, text="Quit", command=self.root.destroy).pack(
            side="left", padx=10)

        self.update_sf_params()
        self.on_likelihood_change()

    def on_likelihood_change(self, event=None):
        is_poisson = (self.likelihood.get() == "poisson")
        if is_poisson:
            self.poisson_counts_chk.state(["!disabled"])
        else:
            self.poisson_use_counts.set(False)
            self.poisson_counts_chk.state(["disabled"])

    def update_sf_params(self, event=None):
        for widget in self.sf_frame.winfo_children():
            widget.destroy()
        self.sf_vars.clear()

        model = self.sf_model.get()
        req = SF_REQUIRED_PARAMS.get(model, [])
        if not req:
            return

        ttk.Label(self.sf_frame, text="SF Parameters:",
                  font=("Segoe UI", 9, "italic")).grid(row=0, column=0, sticky="w", pady=(0, 5))

        for i, param in enumerate(req):
            ttk.Label(self.sf_frame, text=f"{param}:").grid(
                row=i + 1, column=0, sticky="e", padx=5)
            val = "0.0"
            if param == "eta":
                val = "0.3"
            elif param == "R_hs":
                val = "500.0"
            elif param == "xi":
                val = "300.0"
            elif param == "b":
                val = "-1.0"
            elif param == "log_a":
                val = "0.0"

            var = tk.StringVar(value=val)
            ttk.Entry(self.sf_frame, textvariable=var, width=10).grid(
                row=i + 1, column=1, sticky="w", padx=5)
            self.sf_vars[param] = var

            step_val = "0.1"
            if param == "eta":
                step_val = "0.01"
            elif param == "R_hs":
                step_val = "0.145"
            elif param == "xi":
                step_val = "0.15"
            elif param == "b":
                step_val = "0.1"
            elif param == "log_a":
                step_val = "0.5"

            ttk.Label(self.sf_frame, text="step:").grid(
                row=i + 1, column=2, sticky="e", padx=(10, 2))
            var_step = tk.StringVar(value=step_val)
            ttk.Entry(self.sf_frame, textvariable=var_step, width=6).grid(
                row=i + 1, column=3, sticky="w", padx=2)
            self.sf_vars[param + "_step"] = var_step

    def on_browse(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Data Files", "*.txt;*.dat;*.csv"), ("All Files", "*.*")])
        if files:
            self.selected_files = files
            self.file_path.set(f"{len(files)} files selected")
            if self.auto_r.get() and len(files) > 0:
                try:
                    q, _, _ = load_sas_data(files[0])
                    if q.size > 0:
                        q_min_val = np.min(q[q > 0]) if np.any(q > 0) else 1e-3
                        q_max_val = np.max(q)
                        self.r_min.set(f"{2.5 / q_max_val:.2f}")
                        self.r_max.set(f"{2.5 / q_min_val:.2f}")
                except Exception as e:
                    print(f"Error estimating R from first file: {e}")

    def on_browse_output(self):
        current = self.output_root.get().strip()
        init_dir = current if current and os.path.isdir(current) else os.path.dirname(os.path.abspath(__file__))
        out_dir = filedialog.askdirectory(initialdir=init_dir)
        if out_dir:
            self.output_root.set(out_dir)

    def on_run(self):
        try:
            if hasattr(self, 'selected_files') and self.selected_files:
                files_to_process = self.selected_files
            else:
                manual_path = self.file_path.get()
                if manual_path and os.path.exists(manual_path):
                    files_to_process = [manual_path]
                else:
                    raise ValueError("No valid data files selected.")

            rmin = float(self.r_min.get())
            rmax = float(self.r_max.get())
            nr = int(self.n_r.get())

            def _parse_float(var, default):
                try:
                    x = float(var.get())
                except Exception:
                    return float(default)
                return float(x) if np.isfinite(x) else float(default)

            def _parse_int(var, default):
                try:
                    x = int(var.get())
                except Exception:
                    return int(default)
                return int(x)

            sf_params = {}
            sf_steps = {}
            for k, v in self.sf_vars.items():
                if k.endswith("_step"):
                    sf_steps[k.replace("_step", "")] = float(v.get())
                else:
                    sf_params[k] = float(v.get())

            processed_count = 0
            for fpath in files_to_process:
                if not os.path.exists(fpath):
                    print(f"Skipping missing file: {fpath}")
                    continue

                print(f"\nProcessing: {os.path.basename(fpath)}...")

                pscale_str = self.poisson_scale_str.get().strip()
                p_scale_user = float(pscale_str) if pscale_str else None
                is_poisson = (self.likelihood.get() == "poisson")
                data_mode = "q_i_sigma"
                used_scale_note = "n/a"
                sigma_fallback_used = False

                if is_poisson and self.poisson_use_counts.get():
                    q, counts, scale_i_file = load_counts_data(fpath)
                    data_mode = "counts_mode"
                    if p_scale_user is not None:
                        scale_used = float(p_scale_user)
                        used_scale_note = "GUI s0"
                    elif scale_i_file is not None:
                        scale_used = scale_i_file
                        used_scale_note = "file scale_i"
                        p_scale_user = scale_i_file
                    else:
                        raise ValueError(
                            "Counts-mode needs GUI s0 or 3rd column scale_i.")
                    counts = np.maximum(counts, 0.0)
                    I = counts / np.maximum(scale_used, 1e-30)
                    sigma = np.sqrt(np.maximum(counts, 1.0)) / np.maximum(scale_used, 1e-30)
                    print(f"[Poisson counts-mode] converted using {used_scale_note}.")
                else:
                    q, I, sigma = load_sas_data(fpath)

                if sigma is None or np.all(np.isnan(sigma)):
                    if is_poisson and p_scale_user is None:
                        messagebox.showerror(
                            "Error",
                            f"File '{os.path.basename(fpath)}' missing sigma.\n"
                            "Set 'P. Scale s0' for Poisson mode.")
                        continue
                    elif not is_poisson:
                        sigma = np.abs(I) * 0.05
                        sigma_fallback_used = True

                if self.auto_r.get() and q.size > 0:
                    q_min_val = np.min(q[q > 0]) if np.any(q > 0) else 1e-3
                    q_max_val = np.max(q)
                    curr_rmin = 2.5 / q_max_val
                    curr_rmax = 2.5 / q_min_val
                else:
                    curr_rmin, curr_rmax = rmin, rmax

                alpha_tol_val = _parse_float(self.alpha_tol, 1e-3)
                smooth_lambda_val = max(0.0, _parse_float(self.smooth_lambda, 0.1))
                alpha_min_val = max(1e-300, _parse_float(self.alpha_min, 1e-6))
                alpha_max_val = max(alpha_min_val * (1.0 + 1e-9), _parse_float(self.alpha_max, 1e6))
                max_bisect_iter_val = max(5, _parse_int(self.max_bisect_iter, 40))
                opt_maxiter_val = max(100, _parse_int(self.opt_maxiter, 2000))
                opt_ftol_val = max(1e-16, _parse_float(self.opt_ftol, 1e-10))
                opt_gtol_val = max(1e-16, _parse_float(self.opt_gtol, 1e-8))
                eps_i_val = max(1e-300, _parse_float(self.eps_I, 1e-10))
                u_min_val = _parse_float(self.u_min, -60.0)
                u_max_val = _parse_float(self.u_max, 60.0)
                if u_max_val <= u_min_val:
                    u_min_val, u_max_val = -60.0, 60.0
                pclip_min_val = max(1e-300, _parse_float(self.poisson_clip_min, 1e-12))
                pclip_max_val = max(pclip_min_val * (1.0 + 1e-9), _parse_float(self.poisson_clip_max, 1e12))
                sigma_rel_floor_val = max(0.0, _parse_float(self.sigma_rel_floor, 0.0))
                sigma_abs_floor_val = max(0.0, _parse_float(self.sigma_abs_floor, 0.0))

                maxent_kwargs = dict(
                    likelihood=self.likelihood.get(),
                    poisson_scale=p_scale_user,
                    alpha_tol=alpha_tol_val,
                    smooth_lambda=smooth_lambda_val,
                    alpha_bounds=(alpha_min_val, alpha_max_val),
                    max_bisect_iter=max_bisect_iter_val,
                    opt_maxiter=opt_maxiter_val,
                    opt_ftol=opt_ftol_val,
                    opt_gtol=opt_gtol_val,
                    eps_I=eps_i_val,
                    u_bounds=(u_min_val, u_max_val),
                    poisson_scale_clip=(pclip_min_val, pclip_max_val),
                    sigma_rel_floor=sigma_rel_floor_val,
                    sigma_abs_floor=sigma_abs_floor_val,
                )
                maxent_cfg = MaxEntConfig(**maxent_kwargs)

                try:
                    chi2_mult = float(self.chi2_multiplier.get())
                except ValueError:
                    chi2_mult = 1.0
                try:
                    grid_pts_val = int(self.grid_points.get())
                except ValueError:
                    grid_pts_val = 20
                try:
                    top_k_val = int(self.grid_top_k.get())
                except ValueError:
                    top_k_val = 3
                try:
                    refine_xtol_val = float(self.refine_xtol.get())
                except ValueError:
                    refine_xtol_val = 1e-5
                try:
                    refine_ftol_val = float(self.refine_ftol.get())
                except ValueError:
                    refine_ftol_val = 1e-5
                try:
                    fine_pts = int(self.fine_grid_points.get())
                except ValueError:
                    fine_pts = 15
                try:
                    fine_fac = float(self.fine_grid_factor.get())
                except ValueError:
                    fine_fac = 0.25

                # Set alpha_mode and targets
                if maxent_cfg.likelihood == "poisson":
                    I_arr = np.asarray(I, float)
                    mask_eff = (I_arr >= 0) & np.isfinite(I_arr)
                    if sigma is not None:
                        sig_arr = np.asarray(sigma, float)
                        if sig_arr.shape == I_arr.shape:
                            mask_eff &= np.isfinite(sig_arr) & (sig_arr > 0)
                    n_eff = int(np.sum(mask_eff))
                    maxent_cfg.alpha_mode = "deviance"
                    n_params = nr + (1 if self.inc_bg.get() else 0) + (1 if self.inc_pl.get() else 0)
                    nu_eff = min(float(n_eff) * 0.5, float(n_params))
                    maxent_cfg.D_target = max(1.0, float(n_eff) - nu_eff) * chi2_mult
                else:
                    maxent_cfg.alpha_mode = "chi2"
                    maxent_cfg.C_target = 0.5 * float(len(q)) * chi2_mult

                # Build SF bounds (no log_alpha; alpha is profiled out).
                sf_mod = self.sf_model.get()
                s_bounds = {}
                if sf_mod == "hard_sphere_py":
                    r_lo = max(0.5 * curr_rmin, 1.0)
                    r_hi = 2.0 * curr_rmax
                    s_bounds = {"R_hs": (r_lo, r_hi), "eta": (0.01, 0.48)}
                elif sf_mod == "Ornstein-Zernike":
                    r_lo = max(0.5 * curr_rmin, 1.0)
                    r_hi = 4.0 * curr_rmax
                    s_bounds = {"xi": (r_lo, r_hi)}
                elif sf_mod == "effective_2017":
                    r_lo = max(0.5 * curr_rmin, 1.0)
                    r_hi = 2.0 * curr_rmax
                    s_bounds = {"log_a": (-10.0, 10.0), "b": (-4.0, 0.0),
                                "R_hs": (r_lo, r_hi), "eta": (0.01, 0.48)}

                outer_mode = self.outer_mode.get()
                if outer_mode not in ["none", "grid_search"]:
                    outer_mode = "grid_search"

                s_steps = {}
                for k in s_bounds.keys():
                    if k not in sf_steps:
                        continue
                    try:
                        step_val = float(sf_steps[k])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(step_val) and step_val > 0:
                        s_steps[k] = step_val

                cfg = PSDInversionConfig(
                    r_min=curr_rmin,
                    r_max=curr_rmax,
                    n_r=nr,
                    sf_model=sf_mod,
                    sf_params0=sf_params,
                    include_background=self.inc_bg.get(),
                    include_powerlaw=self.inc_pl.get(),
                    powerlaw_exponent=float(self.pl_exp.get()),
                    maxent=maxent_cfg,
                    outer=OuterSearchConfig(
                        mode=outer_mode,
                        grid_points=grid_pts_val,
                        local_opt_top_k=top_k_val,
                        local_opt_xtol=refine_xtol_val,
                        local_opt_ftol=refine_ftol_val,
                        bounds=s_bounds,
                        step_scale=s_steps,
                        seed=0,
                        two_stage=self.two_stage.get(),
                        fine_grid_points=fine_pts,
                        fine_grid_factor=fine_fac,
                        fine_grid_warm_start=self.fine_warm_start.get(),
                    ))

                import time
                t0 = time.time()
                result = invert_psd(q, I, sigma, cfg)
                # Debug diagnostics
                print(f"[DEBUG] alpha={result.alpha:.3e}, C={result.C:.2f}, S={result.S:.4f}")
                print(f"[DEBUG] score={result.score:.3f}, D={result.D}")
                print(f"[DEBUG] psd sum={np.sum(result.psd_bins):.4e}")
                print(f"[DEBUG] I_fit range: {np.min(result.I_fit):.3e} ~ {np.max(result.I_fit):.3e}")
                print(f"[DEBUG] I_data range: {np.min(I):.3e} ~ {np.max(I):.3e}")
                t1 = time.time()

                # Porosity calculation
                sld_m = float(self.sld_matrix.get())
                sld_p = float(self.sld_pore.get())
                thick = float(self.thickness.get())
                area = float(self.beam_area.get())
                delta_rho = abs(sld_m - sld_p)

                sum_x = np.sum(result.psd_bins)
                porosity_frac = (sum_x * 1e4) / (delta_rho ** 2) if delta_rho > 0 else 0.0
                porosity_pct = porosity_frac * 100.0
                sample_vol_mm3 = area * thick
                pore_vol_mm3 = porosity_frac * sample_vol_mm3

                try:
                    c_rmin = float(self.calc_r_min.get())
                    c_rmax = float(self.calc_r_max.get())
                except ValueError:
                    c_rmin, c_rmax = 0.0, 1e9

                mask = (result.r_centers >= c_rmin) & (result.r_centers <= c_rmax)
                sum_x_partial = np.sum(result.psd_bins[mask])
                porosity_frac_partial = ((sum_x_partial * 1e4) / (delta_rho ** 2)
                                         if delta_rho > 0 else 0.0)
                porosity_pct_partial = porosity_frac_partial * 100.0
                pore_vol_mm3_partial = porosity_frac_partial * sample_vol_mm3

                if getattr(sys, 'frozen', False):
                    script_dir = os.path.dirname(sys.executable)
                    current_script_name = os.path.basename(sys.executable)
                else:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    core_file = os.environ.get("MAXENT_CORE_FILE", "")
                    current_script_name = (os.path.basename(core_file) if core_file
                                           else os.path.basename(os.path.abspath(__file__)))

                log_info = {
                    "script_name": current_script_name,
                    "fpath": fpath,
                    "data_mode": data_mode,
                    "used_scale_note": used_scale_note,
                    "sigma_fallback_used": sigma_fallback_used,
                    "delta_rho": delta_rho,
                    "porosity_pct": porosity_pct,
                    "sample_vol_mm3": sample_vol_mm3,
                    "pore_vol_mm3": pore_vol_mm3,
                    "porosity_pct_partial": porosity_pct_partial,
                    "pore_vol_mm3_partial": pore_vol_mm3_partial,
                    "c_rmin": c_rmin,
                    "c_rmax": c_rmax,
                    "n_eff": (int(n_eff) if maxent_cfg.likelihood == "poisson"
                              else int(len(q))),
                    "solve_time": t1 - t0,
                    "i_data_min": float(np.min(I)),
                    "i_data_max": float(np.max(I)),
                    "i_fit_min": float(np.min(result.I_fit)),
                    "i_fit_max": float(np.max(result.I_fit)),
                    "sigma_min": (float(np.min(sigma)) if sigma is not None else float("nan")),
                    "sigma_max": (float(np.max(sigma)) if sigma is not None else float("nan")),
                    "q_min": float(np.min(q)),
                    "q_max": float(np.max(q)),
                    "y_neg_count": int(np.sum(np.asarray(I, float) < 0)),
                    "y_zero_count": int(np.sum(np.asarray(I, float) == 0)),
                }

                out_root_ui = self.output_root.get().strip()
                results_root = out_root_ui if out_root_ui else os.path.join(script_dir, "PSD_results")
                results_root = os.path.abspath(results_root)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                data_basename = os.path.splitext(os.path.basename(fpath))[0]
                out_dir = os.path.join(results_root, f"{timestamp}_{data_basename}")
                os.makedirs(out_dir, exist_ok=True)

                import shutil, glob
                try:
                    code_backup_dir = os.path.join(out_dir, "code_backup")
                    os.makedirs(code_backup_dir, exist_ok=True)
                    for py_file in glob.glob(os.path.join(script_dir, "*.py")):
                        shutil.copy2(py_file, code_backup_dir)
                except Exception as e:
                    print(f"Backup warning: {e}")

                # Log output
                log_path = os.path.join(out_dir, "simulation_log.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write("=== MaxEnt PSD Inversion Log ===\n")
                    f.write(f"Script: {current_script_name}\n")
                    f.write(f"Date: {timestamp}\n")
                    f.write(f"Data: {fpath}\n\n")

                    f.write("--- Input Data Stats ---\n")
                    f.write(f"Mode: {log_info['data_mode']}\n")
                    f.write(f"Scale source: {log_info['used_scale_note']}\n")
                    f.write(f"Sigma fallback used: {log_info['sigma_fallback_used']}\n")
                    f.write(f"N points: {len(q)}\n")
                    f.write(f"Q range: {log_info['q_min']:.6g} ~ {log_info['q_max']:.6g}\n")
                    f.write(f"I range: {log_info['i_data_min']:.6g} ~ {log_info['i_data_max']:.6g}\n")
                    f.write(f"Sigma range: {log_info['sigma_min']:.6g} ~ {log_info['sigma_max']:.6g}\n")
                    f.write(f"y<0 count: {log_info['y_neg_count']}, y==0 count: {log_info['y_zero_count']}\n\n")

                    f.write("--- Configuration ---\n")
                    f.write(f"R_min: {cfg.r_min}, R_max: {cfg.r_max}, N_bins: {cfg.n_r}\n")
                    f.write(f"SF Model: {cfg.sf_model}\n")
                    f.write(f"SF Params Initial: {cfg.sf_params0}\n")
                    f.write(f"Background: {cfg.include_background}, "
                            f"Powerlaw: {cfg.include_powerlaw} (n={cfg.powerlaw_exponent})\n")
                    f.write(f"Likelihood: {cfg.maxent.likelihood}\n")
                    f.write(f"PSD smooth λ: {cfg.maxent.smooth_lambda}\n")
                    f.write(f"Outer: {cfg.outer.mode}, "
                            f"Coarse={cfg.outer.grid_points}/dim, "
                            f"Two-stage={cfg.outer.two_stage}, "
                            f"Fine={cfg.outer.fine_grid_points}/dim, "
                            f"Factor={cfg.outer.fine_grid_factor}\n")
                    f.write(f"Fine warm-start: {cfg.outer.fine_grid_warm_start}\n")
                    f.write(f"Alpha profiled out via bisection "
                            f"(bounds={cfg.maxent.alpha_bounds}, tol={cfg.maxent.alpha_tol})\n")
                    f.write(f"Bounds: {cfg.outer.bounds}\n\n")

                    f.write("--- Full MaxEntConfig ---\n")
                    for k, v in sorted(dataclasses.asdict(cfg.maxent).items()):
                        f.write(f"{k}: {v}\n")
                    f.write("\n")

                    f.write("--- Full OuterSearchConfig ---\n")
                    for k, v in sorted(dataclasses.asdict(cfg.outer).items()):
                        f.write(f"{k}: {v}\n")
                    f.write("\n")

                    f.write("--- Results ---\n")
                    f.write(f"Alpha: {result.alpha:.6e}\n")
                    f.write(f"S: {result.S:.6e}\n")
                    f.write(f"C: {result.C:.6e}\n")
                    f.write(f"Smooth cost: {getattr(result, 'smooth_cost', 0.0):.6e}\n")
                    ne = max(int(log_info.get("n_eff", len(q))), 1)
                    if cfg.maxent.likelihood == "poisson" and result.D is not None:
                        f.write(f"Deviance D: {result.D:.4f}\n")
                        f.write(f"D/n_eff: {result.D / ne:.4f}\n")
                        f.write(f"n_eff / M: {ne} / {len(q)}\n")
                    else:
                        chi2 = 2.0 * result.C
                        f.write(f"Chi2: {chi2:.4f}, Chi2/M: {chi2/len(q):.4f}\n")
                    f.write(f"Score: {result.score:.4f}\n")
                    f.write(f"SF Params: {result.sf_params}\n")
                    f.write(f"Extras: {result.extras}\n")
                    f.write(f"I_fit range: {log_info['i_fit_min']:.6g} ~ {log_info['i_fit_max']:.6g}\n")
                    f.write(f"Time: {t1-t0:.1f}s\n\n")

                    f.write("--- Porosity ---\n")
                    f.write(f"Δρ: {delta_rho}, Total: {porosity_pct:.4f}%\n")
                    f.write(f"Partial ({c_rmin:.0f}-{c_rmax:.0f} Å): "
                            f"{porosity_pct_partial:.4f}%\n")

                history_path = os.path.join(results_root, "run_history.txt")
                with open(history_path, "a", encoding="utf-8") as hf:
                    ne = max(int(log_info.get("n_eff", len(q))), 1)
                    d_by_ne = (result.D / ne) if (
                        cfg.maxent.likelihood == "poisson" and result.D is not None) else float("nan")
                    hf.write(
                        f"{timestamp}\t{os.path.basename(fpath)}\tcore={_loaded_core_version}\t"
                        f"likelihood={cfg.maxent.likelihood}\talpha={result.alpha:.3e}\t"
                        f"D_over_ne={d_by_ne:.6g}\tscore={result.score:.6g}\t"
                        f"smooth={cfg.maxent.smooth_lambda:.3g}\tout={out_dir}\n"
                    )

                csv_path = os.path.join(out_dir, "psd_result.csv")
                save_psd_csv(csv_path, result.edges, result.r_centers, result.psd_bins)

                try:
                    c_rmin_p = float(self.calc_r_min.get())
                    c_rmax_p = float(self.calc_r_max.get())
                    if c_rmin_p <= 1e-6:
                        c_rmin_p = cfg.r_min
                    if c_rmax_p <= c_rmin_p:
                        c_rmax_p = cfg.r_max
                except Exception:
                    c_rmin_p, c_rmax_p = cfg.r_min, cfg.r_max

                self.save_plots(out_dir, q, I, sigma, result, cfg,
                                log_info=log_info, psd_xlim=(c_rmin_p, c_rmax_p))
                processed_count += 1

            messagebox.showinfo(
                "Success",
                f"Done! {processed_count} files → {results_root}")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            import traceback
            traceback.print_exc()

    def save_plots(self, out_dir, q, y, sigma, res, cfg, log_info=None, psd_xlim=None):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib import font_manager

            def _pick_cjk_font():
                candidates = [
                    "Microsoft YaHei", "SimHei", "Microsoft JhengHei",
                    "PingFang SC", "Hiragino Sans GB",
                    "Noto Sans CJK SC", "Source Han Sans SC",
                    "WenQuanYi Micro Hei"]
                available = {f.name for f in font_manager.fontManager.ttflist}
                for name in candidates:
                    if name in available:
                        return name
                return "DejaVu Sans"

            cjk = _pick_cjk_font()
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'sans-serif',
                'font.sans-serif': [cjk, 'Arial', 'Helvetica', 'DejaVu Sans'],
                'axes.unicode_minus': False,
                'axes.linewidth': 1.5,
                'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
                'xtick.major.size': 6, 'ytick.major.size': 6,
                'xtick.direction': 'in', 'ytick.direction': 'in',
                'xtick.top': True, 'ytick.right': True,
                'axes.grid': True,
                'grid.color': '#EBEBEB', 'grid.linestyle': '--', 'grid.linewidth': 1.0,
                'legend.frameon': False,
            })

            color_psd = '#7BA4C7'
            color_fit = '#E64B35'
            color_data = '#6A7B8C'
            color_ideal = '#3B3B3B'

            fig = plt.figure(figsize=(17, 6.5))
            gs = gridspec.GridSpec(2, 3, height_ratios=[2.5, 1],
                                  width_ratios=[1.2, 1.1, 0.85])
            fig.subplots_adjust(hspace=0.08, wspace=0.28,
                                left=0.06, right=0.98, top=0.94, bottom=0.09)

            ax1 = fig.add_subplot(gs[0, 0])
            ax_metric = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax2 = fig.add_subplot(gs[:, 1])
            ax_info = fig.add_subplot(gs[:, 2])
            ax_info.set_axis_off()

            # Scattering fit
            ax1.loglog(q, y, 'o', label='Measured Data',
                       markerfacecolor=color_data, markeredgecolor='none',
                       markersize=5.5, alpha=0.9)
            ax1.loglog(q, res.I_fit, '-', label='MaxEnt Fit',
                       color=color_fit, lw=2.5)
            ax1.set_title("Scattering Data & Fit", fontsize=13, fontweight='bold', pad=8)
            ax1.set_ylabel(r"Intensity $I(Q)$ (cm$^{-1}$)", fontsize=12)
            ax1.legend(loc='upper right', fontsize=10)
            plt.setp(ax1.get_xticklabels(), visible=False)

            # Local metric
            if cfg.maxent.likelihood == "poisson":
                mu = np.maximum(res.I_fit, 1e-30)
                y_safe = np.maximum(y, 0.0)
                if cfg.maxent.poisson_scale is not None:
                    s_cur = np.asarray(cfg.maxent.poisson_scale, float)
                elif sigma is not None:
                    s_cur = np.zeros_like(y_safe)
                    ms = ((y_safe > 0) & np.isfinite(y_safe)
                          & np.isfinite(sigma) & (sigma > 0))
                    s_cur[ms] = y_safe[ms] / (sigma[ms] ** 2)
                    s_cur = np.clip(s_cur, *cfg.maxent.poisson_scale_clip)
                else:
                    s_cur = np.array(1.0)

                mft = (y_safe >= 0) & np.isfinite(y_safe) & np.isfinite(mu)
                if np.ndim(s_cur) > 0:
                    mft = mft & (s_cur > 0) & np.isfinite(s_cur)
                pt_metric = np.full_like(y_safe, np.nan, dtype=float)
                if np.ndim(s_cur) > 0:
                    mu_c = np.maximum(mu[mft] * s_cur[mft], 1e-30)
                    y_c = y_safe[mft] * s_cur[mft]
                else:
                    mu_c = np.maximum(mu[mft] * float(s_cur), 1e-30)
                    y_c = y_safe[mft] * float(s_cur)
                term = np.zeros_like(y_c)
                ml = y_c > 0
                term[ml] = y_c[ml] * np.log(y_c[ml] / mu_c[ml])
                pt_metric[mft] = 2.0 * (mu_c - y_c + term)
                metric_name = r"Local $D_i$"
            else:
                pt_metric = ((res.I_fit - y) / sigma) ** 2
                metric_name = r"Local $\chi^2_i$"

            win = max(3, len(q) // 20)
            mov_avg = np.convolve(pt_metric, np.ones(win) / win, mode='same')
            ax_metric.semilogx(q, pt_metric, '.', color=color_data, alpha=0.5,
                               markersize=4.5, label='Point-wise')
            ax_metric.semilogx(q, mov_avg, '-', color=color_fit, lw=2.5, label='Trend')
            ax_metric.axhline(1.0, color=color_ideal, linestyle='--', lw=1.5, label='Ideal = 1')
            ax_metric.set_xlabel(r"$Q$ ($\mathrm{\AA}^{-1}$)", fontsize=12)
            ax_metric.set_ylabel(metric_name, fontsize=12)
            ax_metric.legend(loc='upper right', fontsize=9, ncol=3)
            vm = pt_metric[np.isfinite(pt_metric)]
            if len(vm) > 0:
                ax_metric.set_ylim(0, max(5.0, np.percentile(vm, 95) * 1.5))

            # PSD
            centers = res.r_centers
            widths = np.diff(np.log(res.edges))
            density = res.psd_bins / widths
            ax2.semilogx(centers, density, '-o', color=color_psd, lw=3.0,
                         markerfacecolor='white', markeredgecolor=color_psd,
                         markeredgewidth=2.0, markersize=7, label='PSD Result')
            ax2.fill_between(centers, 0, density, color=color_psd, alpha=0.25, edgecolor='none')
            ax2.set_title("Pore Size Distribution", fontsize=13, fontweight='bold', pad=8)
            ax2.set_xlabel(r"Radius $r$ ($\mathrm{\AA}$)", fontsize=12)
            ax2.set_ylabel(r"$\mathrm{d}V/\mathrm{d}(\ln r)$", fontsize=12)
            ax2.legend(loc='upper left', fontsize=10)
            if psd_xlim:
                ax2.set_xlim(left=psd_xlim[0], right=psd_xlim[1])
                pmsk = (centers >= psd_xlim[0]) & (centers <= psd_xlim[1])
            else:
                pmsk = np.ones_like(centers, dtype=bool)
            if np.any(pmsk):
                vd = density[pmsk]
                md = np.max(vd)
                p95 = np.percentile(vd, 95)
                ax2.set_ylim(bottom=0,
                             top=(p95 * 1.5 if (md > 2.0 * p95 and p95 > 0) else md * 1.05))

            # Info panel
            def _fv(v):
                if not isinstance(v, float):
                    return str(v)
                return f"{v:.4g}" if (1e-3 < abs(v) < 1e5) else f"{v:.3e}"

            C_HEAD = '#D6E4F7'
            C_ALT = '#F4F9FF'
            C_NONE = '#FFFFFF'

            rows = []

            def H(t):
                rows.append((t, '', True))

            def R(k, v):
                rows.append((k, v, False))

            if log_info and 'script_name' in log_info:
                H("Script")
                s_name = log_info['script_name']
                R("Exec", (s_name[:14] + "…" + s_name[-12:]) if len(s_name) > 28 else s_name)

            H("Data File")
            if log_info and 'fpath' in log_info:
                fn = os.path.basename(log_info['fpath'])
                R("Name", (fn[:14] + "…" + fn[-12:]) if len(fn) > 28 else fn)

            H("Configuration")
            R("r range (Å)", f"{_fv(cfg.r_min)} – {_fv(cfg.r_max)}")
            R("N bins", str(cfg.n_r))
            R("SF model", cfg.sf_model)
            R("Background", "Yes" if cfg.include_background else "No")
            R("Power-law", f"Yes  (n={cfg.powerlaw_exponent})" if cfg.include_powerlaw else "No")
            R("Likelihood", cfg.maxent.likelihood.capitalize())
            R("Smooth λ", _fv(float(getattr(cfg.maxent, "smooth_lambda", 0.0))))
            olbl = cfg.outer.mode.capitalize()
            if cfg.outer.mode != "none":
                olbl += f"  {cfg.outer.grid_points}/Dim"
                if cfg.outer.two_stage:
                    olbl += f"+Fine {cfg.outer.fine_grid_points}/Dim"
                olbl += f", Top {cfg.outer.local_opt_top_k}"
            R("Outer search", olbl)
            R("Alpha mode", "Profiled (bisection)")
            R("Core version", _loaded_core_version or "?")

            H("Fit Metrics")
            nq = len(q)
            if cfg.maxent.likelihood == "poisson":
                ne = max(int(log_info.get("n_eff", nq)), 1) if log_info else nq
                if res.D is not None:
                    R("Deviance D", f"{res.D:.2f}")
                    R("D / n_eff", f"{res.D / ne:.3f}")
                    R("n_eff / M", f"{ne} / {nq}")
            else:
                chi2 = 2.0 * res.C
                R("χ²", f"{chi2:.2f}")
                R("χ² / M", f"{chi2 / nq:.3f}   (M = {nq})")
            R("α  (regulariz.)", f"{res.alpha:.3e}")
            R("Score", f"{res.score:.3f}")
            if log_info and 'solve_time' in log_info:
                R("Elapsed  (s)", f"{log_info['solve_time']:.1f}")

            sf_lbl = ("Optimized SF Params" if cfg.outer.mode != "none"
                      else "Fixed SF Params")
            H(sf_lbl)
            if res.sf_params:
                for pk, pv in res.sf_params.items():
                    R(f"  {pk}", _fv(pv))
            if res.extras:
                for ek, ev in res.extras.items():
                    R(f"  {ek}", _fv(ev))

            if log_info and 'delta_rho' in log_info:
                H("Porosity Analysis")
                R("Δρ (10^-6 A^-2)", f"{log_info['delta_rho']:.4g}")
                R("Sample vol  (mm³)", f"{log_info['sample_vol_mm3']:.3f}")
                R("Total porosity", f"{log_info['porosity_pct']:.4f} %")
                R("Total pore vol (mm³)", f"{log_info['pore_vol_mm3']:.3e}")
                cr0, cr1 = log_info['c_rmin'], log_info['c_rmax']
                R(f"Range {cr0:.0f}–{cr1:.0f} Å", "")
                R("  Partial porosity", f"{log_info['porosity_pct_partial']:.4f} %")
                R("  Part. pore vol", f"{log_info['pore_vol_mm3_partial']:.3e} mm³")

            nr_rows = len(rows)
            rh = 1.0 / max(nr_rows + 1, 1)
            PL, PR = 0.04, 0.98
            TL, TR = 0.06, 0.96

            for i, (key, val, is_hdr) in enumerate(rows):
                yt = 1.0 - i * rh
                yb = yt - rh
                ym = 0.5 * (yt + yb)
                fc = C_HEAD if is_hdr else (C_ALT if i % 2 == 0 else C_NONE)
                ax_info.add_patch(plt.Rectangle(
                    (PL, yb), PR - PL, rh * 0.96,
                    transform=ax_info.transAxes,
                    facecolor=fc, edgecolor='none', clip_on=False))
                if is_hdr:
                    ax_info.text(0.50, ym, key, transform=ax_info.transAxes,
                                fontsize=11.5, fontweight='bold', color='#1A3A5C',
                                va='center', ha='center')
                else:
                    ax_info.text(TL, ym, key, transform=ax_info.transAxes,
                                fontsize=10.5, color='#444444', va='center', ha='left')
                    ax_info.text(TR, ym, str(val), transform=ax_info.transAxes,
                                fontsize=10.5, color='#111111', va='center', ha='right',
                                fontweight='medium')

            ax_info.set_xlim(0, 1)
            ax_info.set_ylim(0, 1)

            save_path = os.path.join(out_dir, "fit_and_psd.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}", flush=True)
            plt.show()

        except Exception as e:
            print(f"Plotting error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            messagebox.showwarning("Plot Error", f"Could not create plots:\n{e}")


def run_gui_app():
    root = tk.Tk()
    root.attributes("-topmost", True)
    app = MaxEntApp(root)
    root.mainloop()


if __name__ == "__main__":
    import argparse

    if len(sys.argv) == 1:
        run_gui_app()
    else:
        p = argparse.ArgumentParser(
            description="Bayesian-MaxEnt PSD inversion for SAS data.")
        p.add_argument("data", help="Data file (q I sigma)")
        p.add_argument("--rmin", type=float, required=True)
        p.add_argument("--rmax", type=float, required=True)
        p.add_argument("--nr", type=int, default=60)
        p.add_argument("--sf", type=str, default="unity",
                       choices=list(STRUCTURE_FACTOR_MODELS.keys()))
        p.add_argument("--likelihood", default="gaussian",
                       choices=["gaussian", "poisson"])
        p.add_argument("--smooth-lambda", type=float, default=0.1)
        p.add_argument("--outer", default="none", choices=["none", "grid_search"])
        p.add_argument("--save", type=str, default="")
        p.add_argument("--plot", action="store_true")
        args = p.parse_args()

        q, I, sigma = load_sas_data(args.data)
        cfg = PSDInversionConfig(
            r_min=args.rmin, r_max=args.rmax, n_r=args.nr,
            sf_model=args.sf, sf_params0={},
            include_background=True, include_powerlaw=False,
            maxent=MaxEntConfig(likelihood=args.likelihood,
                                smooth_lambda=max(0.0, float(args.smooth_lambda))),
            outer=OuterSearchConfig(mode=args.outer))
        result = invert_psd(q, I, sigma, cfg)
        print(f"alpha={result.alpha}, S={result.S}, C={result.C}, score={result.score}")
        print(f"sf_params={result.sf_params}, extras={result.extras}")
        if args.save:
            save_psd_csv(args.save, result.edges, result.r_centers, result.psd_bins)
        if args.plot:
            try:
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.loglog(q, I, 'o', ms=4, label='Data')
                ax1.loglog(q, result.I_fit, '-', lw=2, label='Fit')
                ax1.set_xlabel('Q')
                ax1.set_ylabel('I(Q)')
                ax1.legend()

                widths = np.diff(np.log(np.maximum(result.edges, 1e-300)))
                density = result.psd_bins / np.maximum(widths, 1e-300)
                ax2.semilogx(result.r_centers, density, '-o', lw=2)
                ax2.set_xlabel('r (A)')
                ax2.set_ylabel('dV/d(ln r)')

                plt.tight_layout()
                plt.show()
            except ImportError:
                print("matplotlib not available, skipping plot")
