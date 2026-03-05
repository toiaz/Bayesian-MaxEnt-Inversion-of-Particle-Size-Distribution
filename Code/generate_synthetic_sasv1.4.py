#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic SAS Data Generator GUI component extracted from MaxEnt PSD v1.4.
"""

import os
import sys
import math
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

Array = np.ndarray

# ---------------------------------------------------------------------
# Synthetic Helpers
# ---------------------------------------------------------------------
def structure_factor_hardsphere_py(q: Array, params: dict) -> Array:
    R = float(params["R_hs"])
    eta = float(params["eta"])
    if not (0.0 < eta < 0.49):
        raise ValueError("eta must be between 0 and 0.49 for PY hard-sphere model.")
    q = np.asarray(q, dtype=float)
    x = q * 2.0 * R

    alpha = (1.0 + 2.0 * eta) ** 2 / (1.0 - eta) ** 4
    beta = -6.0 * eta * (1.0 + eta / 2.0) ** 2 / (1.0 - eta) ** 4
    gamma = 0.5 * eta * alpha

    term1 = np.empty_like(x)
    term2 = np.empty_like(x)
    term3 = np.empty_like(x)

    small = np.abs(x) < 1e-3
    xs = x[~small]

    if xs.size:
        sinx = np.sin(xs)
        cosx = np.cos(xs)
        term1[~small] = (sinx - xs * cosx) / xs ** 2
        term2[~small] = (2.0 * xs * sinx + (2.0 - xs ** 2) * cosx - 2.0) / xs ** 3
        term3num = (-xs ** 4) * cosx + 4.0 * (
            (3.0 * xs ** 2 - 6.0) * cosx + (xs ** 3 - 6.0 * xs) * sinx + 6.0
        )
        term3[~small] = term3num / xs ** 5

    if np.any(small):
        x0 = x[small]
        term1[small] = x0 / 3.0 - x0 ** 3 / 30.0 + x0 ** 5 / 840.0
        term2[small] = x0 / 4.0 - x0 ** 3 / 36.0 + x0 ** 5 / 960.0
        term3[small] = x0 / 6.0 - x0 ** 3 / 48.0 + x0 ** 5 / 1200.0

    G = alpha * term1 + beta * term2 + gamma * term3

    S = np.empty_like(x)
    zero = np.abs(x) < 1e-12
    nonzero = ~zero
    S[zero] = 1.0 / alpha
    S[nonzero] = 1.0 / (1.0 + 24.0 * eta * G[nonzero] / x[nonzero])
    return S

def form_factor_sphere_sq(q: Array, r: Array) -> Array:
    qr = np.outer(q, r)
    phi = np.ones_like(qr)
    small = np.abs(qr) < 1e-3
    normal = ~small
    if np.any(normal):
        phi[normal] = 3.0 * (np.sin(qr[normal]) - qr[normal] * np.cos(qr[normal])) / (qr[normal] ** 3)
    if np.any(small):
        phi[small] = 1.0 - (qr[small] ** 2) / 10.0 + (qr[small] ** 4) / 280.0
    return phi ** 2

def gaussian_in_logr(r: Array, r0: float, sigma: float) -> Array:
    z = (np.log(r) - np.log(r0)) / sigma
    return np.exp(-0.5 * z * z)

def _smooth_gaussian_1d(w: Array, sigma_pts: float) -> Array:
    sigma_pts = float(max(sigma_pts, 1e-6))
    n = int(len(w))
    half = int(min(np.ceil(4.0 * sigma_pts), max(1, (n - 1) // 2)))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma_pts) ** 2)
    k /= np.sum(k)
    return np.convolve(w, k, mode="same")

def correlated_ripple_on_logq(q: Array, rng, rel_amp: float = 0.01, corr_decades: float = 0.6) -> Array:
    x = np.log10(q)
    dx = np.median(np.diff(x))
    sigma_pts = corr_decades / max(dx, 1e-12)
    w = rng.normal(0.0, 1.0, size=q.size)
    s = _smooth_gaussian_1d(w, sigma_pts=sigma_pts)
    std = float(np.std(s)) if np.std(s) > 0 else 1.0
    s = s / std
    g = rel_amp * s
    g -= np.mean(g)
    return g

# ---------------------------------------------------------------------
# GUI Interface
# ---------------------------------------------------------------------
class SyntheticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synthetic SAS Data Generator v1.4")
        self.root.resizable(True, True)
        self.style_setup()

        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill="both", expand=True)
        self.create_synthetic_widgets(main_frame)

    def style_setup(self):
        self.font_norm = ("Segoe UI", 10)
        self.font_bold = ("Segoe UI", 10, "bold")
        self.pad_x = 8
        self.pad_y = 5

    def add_label(self, parent, text, row, col=0, font=None):
        if font is None: font = self.font_norm
        lbl = ttk.Label(parent, text=text, font=font)
        lbl.grid(row=row, column=col, sticky="e", padx=self.pad_x, pady=self.pad_y)
        return lbl

    def add_entry(self, parent, default, row, col=1, width=20):
        var = tk.StringVar(value=str(default))
        ent = ttk.Entry(parent, textvariable=var, width=width, font=self.font_norm)
        ent.grid(row=row, column=col, sticky="w", padx=self.pad_x, pady=self.pad_y)
        return var

    def create_synthetic_widgets(self, parent):
        row = 0
        ttk.Label(parent, text="Synthetic Generation Parameters", font=("Segoe UI", 12, "bold")).grid(row=row, column=0, columnspan=4, pady=(0, 15), sticky="w")
        row += 1

        # Q Grid
        ttk.Label(parent, text="1. Q Grid Parameters", font=self.font_bold).grid(row=row, column=0, columnspan=4, sticky="w", pady=5)
        row += 1
        self.add_label(parent, "Q min:", row, 0)
        self.syn_qmin = self.add_entry(parent, "2e-4", row, 1, 10)
        self.add_label(parent, "Q max:", row, 2)
        self.syn_qmax = self.add_entry(parent, "0.3", row, 3, 10)
        row += 1
        self.add_label(parent, "Q points (N_q):", row, 0)
        self.syn_nq = self.add_entry(parent, "2000", row, 1, 10)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=5)
        row += 1

        # True PSD Grid
        ttk.Label(parent, text="2. True PSD Settings (Multimodal)", font=self.font_bold).grid(row=row, column=0, columnspan=4, sticky="w", pady=5)
        row += 1
        self.add_label(parent, "R min (Å):", row, 0)
        self.syn_rmin = self.add_entry(parent, "5.0", row, 1, 10)
        self.add_label(parent, "R max (Å):", row, 2)
        self.syn_rmax = self.add_entry(parent, "15000.0", row, 3, 10)
        row += 1
        self.add_label(parent, "N bins:", row, 0)
        self.syn_nbins = self.add_entry(parent, "90", row, 1, 10)
        self.add_label(parent, "Volume Frac:", row, 2)
        self.syn_phi = self.add_entry(parent, "0.10", row, 3, 10)
        row += 1

        self.add_label(parent, "Peak 1 (w, r0, s):", row, 0)
        self.syn_p1_w = self.add_entry(parent, "2.0", row, 1, 6)
        self.syn_p1_r = self.add_entry(parent, "150.0", row, 2, 8)
        self.syn_p1_s = self.add_entry(parent, "0.60", row, 3, 8)
        row += 1

        self.add_label(parent, "Peak 2 (w, r0, s):", row, 0)
        self.syn_p2_w = self.add_entry(parent, "3.0", row, 1, 6)
        self.syn_p2_r = self.add_entry(parent, "2500.0", row, 2, 8)
        self.syn_p2_s = self.add_entry(parent, "0.55", row, 3, 8)
        row += 1

        self.add_label(parent, "Peak 3 (w, r0, s):", row, 0)
        self.syn_p3_w = self.add_entry(parent, "0.0", row, 1, 6)
        self.syn_p3_r = self.add_entry(parent, "0.0", row, 2, 8)
        self.syn_p3_s = self.add_entry(parent, "0.0", row, 3, 8)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=5)
        row += 1

        # Physical & Noise
        ttk.Label(parent, text="3. Systematics & Noise", font=self.font_bold).grid(row=row, column=0, columnspan=4, sticky="w", pady=5)
        row += 1

        self.add_label(parent, "SLD Contrast:", row, 0)
        self.syn_sld = self.add_entry(parent, "3.98", row, 1, 10)
        self.add_label(parent, "Poiss Time:", row, 2)
        self.syn_time = self.add_entry(parent, "50000.0", row, 3, 10)
        row += 1

        self.add_label(parent, "BG cm-1:", row, 0)
        self.syn_bg = self.add_entry(parent, "0.015", row, 1, 10)
        self.add_label(parent, "BG rel err:", row, 2)
        self.syn_bgerr = self.add_entry(parent, "0.02", row, 3, 10)
        row += 1

        self.add_label(parent, "Sys err lowQ:", row, 0)
        self.syn_syslow = self.add_entry(parent, "0.001", row, 1, 10)
        self.add_label(parent, "Sys err highQ:", row, 2)
        self.syn_syshi = self.add_entry(parent, "8e-4", row, 3, 10)
        row += 1

        self.add_label(parent, "Sys q0:", row, 0)
        self.syn_q0sys = self.add_entry(parent, "2.0e-3", row, 1, 10)
        self.add_label(parent, "Sys p:", row, 2)
        self.syn_psys = self.add_entry(parent, "1.5", row, 3, 10)
        row += 1

        self.add_label(parent, "Sigma floor abs:", row, 0)
        self.syn_sigmafloor = self.add_entry(parent, "2e-5", row, 1, 10)
        self.syn_global_s0 = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Enable Pure Poisson Counts (Global s0)", variable=self.syn_global_s0).grid(row=row, column=2, columnspan=2, sticky="w", padx=self.pad_x)
        row += 1

        # Counts cap control
        self.add_label(parent, "Counts Cap:", row, 0)
        self.syn_counts_cap = self.add_entry(parent, "2e5", row, 1, 10)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=5)
        row += 1

        ttk.Label(parent, text="4. Structure Factor", font=self.font_bold).grid(row=row, column=0, columnspan=4, sticky="w", pady=5)
        row += 1

        self.syn_sf = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Enable Hard Sphere SF", variable=self.syn_sf).grid(row=row, column=0, columnspan=2, sticky="w", padx=self.pad_x)
        self.add_label(parent, "R_hs (Å):", row, 2)
        self.syn_sf_r = self.add_entry(parent, "200.0", row, 3, 10)
        row += 1

        self.add_label(parent, "eta (vol frac):", row, 2)
        self.syn_sf_eta = self.add_entry(parent, "0.25", row, 3, 10)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=5)
        row += 1

        self.syn_ripple = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Enable Ripple", variable=self.syn_ripple).grid(row=row, column=0, columnspan=2, sticky="w", padx=self.pad_x)
        self.add_label(parent, "Ripple Rel Amp:", row, 2)
        self.syn_rip_amp = self.add_entry(parent, "0.01", row, 3, 10)
        row += 1

        self.add_label(parent, "Ripple Corr Dec:", row, 2)
        self.syn_rip_corr = self.add_entry(parent, "0.6", row, 3, 10)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=5)
        row += 1

        self.syn_stitch = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Stitch Mismatch", variable=self.syn_stitch).grid(row=row, column=0, columnspan=2, sticky="w", padx=self.pad_x)
        self.add_label(parent, "Stitch Rel Sigma:", row, 2)
        self.syn_stitch_sig = self.add_entry(parent, "0.02", row, 3, 10)
        row += 1

        self.add_label(parent, "Stitch Edges (Q):", row, 2)
        self.syn_stitch_edges = self.add_entry(parent, "8e-4, 2e-2", row, 3, 10)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=5)
        row += 1

        ttk.Button(parent, text="Generate Data & Plot", command=self.on_generate_synthetic).grid(row=row, column=0, columnspan=4, pady=20)


    def on_generate_synthetic(self):
        try:
            q_min = float(self.syn_qmin.get())
            q_max = float(self.syn_qmax.get())
            N_q = int(self.syn_nq.get())
            q = np.logspace(np.log10(q_min), np.log10(q_max), N_q)

            N_bins = int(self.syn_nbins.get())
            R_min = float(self.syn_rmin.get())
            R_max = float(self.syn_rmax.get())

            r_edges = np.logspace(np.log10(R_min), np.log10(R_max), N_bins + 1)
            r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])
            dlnr = np.diff(np.log(r_edges))

            w1, r01, s1 = float(self.syn_p1_w.get()), float(self.syn_p1_r.get()), float(self.syn_p1_s.get())
            w2, r02, s2 = float(self.syn_p2_w.get()), float(self.syn_p2_r.get()), float(self.syn_p2_s.get())
            w3, r03, s3 = float(self.syn_p3_w.get()), float(self.syn_p3_r.get()), float(self.syn_p3_s.get())

            dphi_dlnr = np.zeros_like(r_centers)
            if w1 > 0 and r01 > 0 and s1 > 0:
                dphi_dlnr += w1 * gaussian_in_logr(r_centers, r01, s1)
            if w2 > 0 and r02 > 0 and s2 > 0:
                dphi_dlnr += w2 * gaussian_in_logr(r_centers, r02, s2)
            if w3 > 0 and r03 > 0 and s3 > 0:
                dphi_dlnr += w3 * gaussian_in_logr(r_centers, r03, s3)

            if np.sum(dphi_dlnr) == 0:
                dphi_dlnr = np.ones_like(r_centers) # Fallback to prevent crash

            phi_total = float(self.syn_phi.get())
            phi_bin = dphi_dlnr * dlnr
            if np.sum(phi_bin) > 0:
                phi_bin *= (phi_total / np.sum(phi_bin))
                dphi_dlnr = phi_bin / dlnr

            P_qr = form_factor_sphere_sq(q, r_centers)
            V = (4.0 / 3.0) * np.pi * (r_centers ** 3)

            # Apply Structure Factor (Hard Sphere)
            if self.syn_sf.get():
                R_hs = float(self.syn_sf_r.get())
                eta_hs = float(self.syn_sf_eta.get())
                S_q_true = structure_factor_hardsphere_py(q, {"R_hs": R_hs, "eta": eta_hs})
            else:
                S_q_true = np.ones_like(q)

            I_base = S_q_true * ((P_qr * V) @ phi_bin)

            delta_rho = float(self.syn_sld.get())
            bg_cm = float(self.syn_bg.get())

            I_scatt_ideal = (delta_rho ** 2) * 1e-4 * I_base
            I_ideal = I_scatt_ideal + bg_cm

            # Use an explicit per-run seed so the full realization is reproducible from log.
            rng_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
            rng = np.random.default_rng(rng_seed)
            time_factor = float(self.syn_time.get())
            counts_cap_val = float(self.syn_counts_cap.get())

            rel_sys_lowq = float(self.syn_syslow.get())
            rel_sys_highq = float(self.syn_syshi.get())
            q0_sys = float(self.syn_q0sys.get())
            p_sys = float(self.syn_psys.get())
            rel_sys = rel_sys_highq + rel_sys_lowq / (1.0 + (q / q0_sys) ** p_sys)
            bg_sigma_rel = float(self.syn_bgerr.get())
            bg_sigma = bg_sigma_rel * bg_cm

            ENABLE_RIPPLE = self.syn_ripple.get()
            ripple_rel = float(self.syn_rip_amp.get())
            ripple_corr_decades = float(self.syn_rip_corr.get())
            ripple = np.zeros_like(q)
            if ENABLE_RIPPLE:
                ripple = correlated_ripple_on_logq(q, rng, rel_amp=ripple_rel, corr_decades=ripple_corr_decades)

            ENABLE_STITCH_MISMATCH = self.syn_stitch.get()
            stitch_rel_sigma = float(self.syn_stitch_sig.get())
            stitch_edges_str = self.syn_stitch_edges.get()
            stitch_edges_pts = [float(x.strip()) for x in stitch_edges_str.split(",")[:2] if x.strip()]
            if len(stitch_edges_pts) >= 2:
                stitch_edges_q = (stitch_edges_pts[0], stitch_edges_pts[1])
            else:
                stitch_edges_q = (8e-4, 2e-2)
            seg_scale = np.ones_like(q)
            if ENABLE_STITCH_MISMATCH:
                q_edges = (q.min() * 0.999, stitch_edges_q[0], stitch_edges_q[1], q.max() * 1.001)
                scales = 1.0 + rng.normal(0.0, stitch_rel_sigma, size=3)
                scales = np.clip(scales, 0.90, 1.10)
                for k in range(3):
                    m = (q >= q_edges[k]) & (q < q_edges[k + 1])
                    seg_scale[m] = scales[k]
            else:
                scales = np.ones(3, dtype=float)

            I_scatt_sys = I_scatt_ideal * seg_scale * (1.0 + ripple)
            bg_offset = rng.normal(0.0, bg_sigma)
            I_sys = np.clip(I_scatt_sys + bg_cm + bg_offset, 0.0, None)

            # Define sigma_sys first.
            rel_extra_const2 = 0.0
            if ENABLE_RIPPLE:
                rel_extra_const2 += ripple_rel ** 2
            if ENABLE_STITCH_MISMATCH:
                rel_extra_const2 += stitch_rel_sigma ** 2

            rel_total = np.sqrt(rel_sys ** 2 + rel_extra_const2)
            sigma_sys = rel_total * np.maximum(I_scatt_sys, 0.0)

            # Back-calculate effective exposure scale_i(q) from sigma_target.
            sigma_floor_abs = float(self.syn_sigmafloor.get())

            sigma_target = np.sqrt(sigma_sys ** 2 + bg_sigma ** 2 + sigma_floor_abs ** 2)

            # Two Poisson generation modes.
            USE_GLOBAL_S0 = self.syn_global_s0.get()

            if USE_GLOBAL_S0:
                # Standard Poisson mode with global s0.
                s0 = time_factor
                scale_i = np.full_like(q, s0, dtype=float)

                expected_counts = I_sys * scale_i
                COUNTS_CAP = counts_cap_val
                COUNTS_MIN = 0.0
                expected_counts = np.clip(expected_counts, COUNTS_MIN, COUNTS_CAP)

                y_counts = rng.poisson(expected_counts).astype(np.int64)
                I_noisy = y_counts / scale_i
                sigma_stat = np.sqrt(np.maximum(y_counts, 1.0)) / scale_i

                # Gaussian file gets total noise
                sigma_total = np.sqrt(sigma_stat**2 + sigma_target**2)
            else:
                # Effective point-by-point scale_i(q) mode.
                scale_i = I_sys / np.maximum(sigma_target, 1e-30) ** 2
                scale_i *= (time_factor / 50000.0)

                SCALE_I_MIN = 1e-8
                SCALE_I_MAX = 1e8
                scale_i = np.clip(scale_i, SCALE_I_MIN, SCALE_I_MAX)

                expected_counts = np.clip(I_sys * scale_i, 0.0, None)

                COUNTS_CAP = counts_cap_val
                COUNTS_MIN = 5.0
                expected_counts = np.clip(expected_counts, COUNTS_MIN, COUNTS_CAP)

                scale_i = expected_counts / np.maximum(I_sys, 1e-30)

                y_counts = rng.poisson(expected_counts).astype(np.int64)

                I_noisy = y_counts / scale_i
                sigma_stat = np.sqrt(np.maximum(y_counts, 1.0)) / scale_i

                sigma_total = np.maximum(sigma_target, sigma_stat)

            # Keep clip limits explicit for logging even in global-s0 mode.
            if USE_GLOBAL_S0:
                SCALE_I_MIN = np.nan
                SCALE_I_MAX = np.nan

            if getattr(sys, 'frozen', False):
                script_dir = os.path.dirname(sys.executable)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))

            synthetic_root = os.path.join(script_dir, "Synthetic_data")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(synthetic_root, f"GUI_v1.4_{timestamp}")
            os.makedirs(out_dir, exist_ok=True)

            if USE_GLOBAL_S0:
                poisson_mode_str = "global_s0"
                s0_str = f"# s0 = time_factor = {time_factor}"
            else:
                poisson_mode_str = "effective_scale_i"
                s0_str = "# scale_i saved in SynData_Counts.dat third column"

            # Write systematic terms into the Poisson header.
            f_poiss = os.path.join(out_dir, "SynData_Poisson.dat")
            header_poiss = (f"Q I_noisy Sigma_stat\n"
                            f"# time_factor={time_factor}, counts_cap={counts_cap_val}\n"
                            f"# poisson_mode={poisson_mode_str}\n"
                            f"{s0_str}\n"
                            f"# N_q={N_q} R_min={R_min} R_max={R_max} N_bins={N_bins}\n"
                            f"# rel_sys: lowq={rel_sys_lowq}, highq={rel_sys_highq}, q0={q0_sys}, p={p_sys}\n"
                            f"# ripple: {ENABLE_RIPPLE}, ripple_rel={ripple_rel}, corr_decades={ripple_corr_decades}\n"
                            f"# stitch: {ENABLE_STITCH_MISMATCH}, stitch_rel_sigma={stitch_rel_sigma}, edges={stitch_edges_q}")
            np.savetxt(f_poiss, np.column_stack((q, I_noisy, sigma_stat)), header=header_poiss, comments="# ")
            f_gauss = os.path.join(out_dir, "SynData_Gaussian.dat")
            header_gauss=(f"Q I_noisy Sigma_total\n# time_factor={time_factor}, counts_cap={counts_cap_val}\n# poisson_mode={poisson_mode_str}\n{s0_str}\n# N_q={N_q} R_min={R_min} R_max={R_max} N_bins={N_bins}\n"
                          f"# sigma_total^2 = sigma_stat^2 + sigma_sys^2 + bg_sigma^2 + sigma_floor^2\n"
                          f"# rel_sys: lowq={rel_sys_lowq}, highq={rel_sys_highq}, q0={q0_sys}, p={p_sys}\n"
                          f"# ripple: {ENABLE_RIPPLE}, ripple_rel={ripple_rel}, corr_decades={ripple_corr_decades}\n"
                          f"# stitch: {ENABLE_STITCH_MISMATCH}, stitch_rel_sigma={stitch_rel_sigma}, edges={stitch_edges_q}")
            np.savetxt(f_gauss, np.column_stack((q, I_noisy, sigma_total)), header=header_gauss, comments="# ")

            file_both = os.path.join(out_dir, "SynData_BothSigma.dat")
            np.savetxt(file_both, np.column_stack((q, I_noisy, sigma_stat, sigma_total)), header="Q I_noisy Sigma_stat Sigma_total", comments="# ")

            file_counts = os.path.join(out_dir, "SynData_Counts.dat")
            np.savetxt(file_counts, np.column_stack((q, y_counts, scale_i)), header="Q counts scale_i\n# counts ~ Poisson(I_sys*scale_i)", comments="# ")

            scale_from_stat = np.maximum(I_noisy, 0.0) / np.maximum(sigma_stat, 1e-30) ** 2
            scale_from_total = np.maximum(I_noisy, 0.0) / np.maximum(sigma_total, 1e-30) ** 2
            file_scale = os.path.join(out_dir, "SynData_EffectiveScale_i.dat")
            np.savetxt(file_scale, np.column_stack((q, scale_from_stat, scale_from_total)), header="Q scale_from_stat scale_from_total\n# scale_from_stat = I/sigma_stat^2\n# scale_from_total = I/sigma_total^2", comments="# ")

            f_truedata = os.path.join(out_dir, "SynData_TruePSD.csv")
            np.savetxt(f_truedata, np.column_stack((r_centers, dphi_dlnr, phi_bin)), header="r_center, dphi_dlnr, phi_bin", delimiter=",", comments="")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            ax1.loglog(q, I_noisy, ".", color="gray", label="Noisy Data", alpha=0.6, markersize=2)
            ax1.loglog(q, I_ideal, "-", color="red", lw=2, label="Ideal I(q)")
            ax1.loglog(q, I_sys, "--", color="black", lw=1.5, label="I_sys", alpha=0.85)
            ax1.set_xlabel(r"Q ($\mathrm{\AA}^{-1}$)")
            ax1.set_ylabel("Intensity (cm$^{-1}$)")
            ax1.set_title("Simulated Scattering Profile")
            ax1.legend()
            ax1.grid(True, alpha=0.25)

            ax2.semilogx(r_centers, dphi_dlnr, "-", color="blue", lw=2, label=r"True PSD")
            ax2.set_xlabel(r"Radius r ($\mathrm{\AA}$)")
            ax2.set_ylabel(r"d$\phi$/dln$r$")
            ax2.set_title("True Pore Size Distribution")
            ax2.set_xlim(R_min, min(R_max, 10000))
            ax2.legend()
            ax2.grid(True, alpha=0.25)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "SynData_Plot.png"), dpi=300)

            scale_from_sigma_total = np.maximum(I_noisy, 0.0) / np.maximum(sigma_total, 1e-30)**2
            scale_from_sigma_stat  = np.maximum(I_noisy, 0.0) / np.maximum(sigma_stat, 1e-30)**2

            def pct(a):
                a = a[np.isfinite(a)]
                if len(a) == 0:
                     return [0.0, 0.0, 0.0]
                return np.percentile(a, [1, 50, 99])

            print("\n--- Noise/Weight Diagnostics (GUI Gen) ---")
            print("scale_i true         p[1,50,99] =", pct(scale_i))
            print("scale(I/sigTot^2)    p[1,50,99] =", pct(scale_from_sigma_total))
            print("scale(I/sigStat^2)   p[1,50,99] =", pct(scale_from_sigma_stat))
            print("expected_counts      p[1,50,99] =", pct(expected_counts), "max=", np.max(expected_counts))
            print("sigma_total rel      p[1,50,99] =", pct(sigma_total/np.maximum(I_noisy,1e-30)))
            print("------------------------------------------")

            log_path = os.path.join(out_dir, "synthetic_parameters.log")
            with open(log_path, "w", encoding="utf-8") as ff:
                ff.write("=== Synthetic Data Generation Parameters ===\n")
                ff.write(f"Q Grid: min={q_min}, max={q_max}, N_q={N_q}\n")
                ff.write(f"PSD True Grid: R_min={R_min}, R_max={R_max}, N_bins={N_bins}, phi_total={phi_total}\n")
                ff.write(f"Peak 1: w={w1}, r0={r01}, s={s1}\n")
                ff.write(f"Peak 2: w={w2}, r0={r02}, s={s2}\n")
                ff.write(f"Peak 3: w={w3}, r0={r03}, s={s3}\n")
                ff.write(f"Systematics & Noise:\n")
                ff.write(f"  SLD Contrast: {delta_rho}\n")
                ff.write(f"  Poiss Time (time_factor): {time_factor}\n")
                ff.write(f"  Poisson Generation Mode: {'Global s0' if USE_GLOBAL_S0 else 'Effective scale_i(q)'}\n")
                ff.write(f"  BG cm-1: {bg_cm}\n")
                ff.write(f"  BG rel err: {bg_sigma_rel}\n")
                ff.write(f"  Sys err lowQ: {rel_sys_lowq}\n")
                ff.write(f"  Sys err highQ: {rel_sys_highq}\n")
                ff.write(f"  Sys q0: {q0_sys}\n")
                ff.write(f"  Sys p: {p_sys}\n")
                ff.write(f"  Sigma floor abs: {sigma_floor_abs}\n")
                ff.write(f"  Counts Cap: {counts_cap_val}\n")
                ff.write(f"Ripple Effect: Enabled={ENABLE_RIPPLE}, Rel Amp={ripple_rel}, Corr Decades={ripple_corr_decades}\n")
                ff.write(f"Stitch Mismatch: Enabled={ENABLE_STITCH_MISMATCH}, Rel Sigma={stitch_rel_sigma}, Edges={stitch_edges_q}\n")
                if self.syn_sf.get():
                    ff.write(f"Structure Factor: Hard Sphere (R_hs={self.syn_sf_r.get()}, eta={self.syn_sf_eta.get()})\n")
                else:
                    ff.write("Structure Factor: None (Unity)\n")
                ff.write("\n=== Full Effective Parameter Snapshot ===\n")
                ff.write(f"Script Version: GUI_v1.4\n")
                ff.write(f"Output Directory: {out_dir}\n")
                ff.write(f"RNG Seed: {rng_seed}\n")
                ff.write(f"Form Factor Model: sphere_sq\n")
                ff.write(f"Kernel Weighting: sphere volume V(r)=4*pi*r^3/3\n")
                ff.write(f"Density Scale Factor in I_scatt_ideal: 1e-4\n")
                ff.write(f"Poisson Mode Flag USE_GLOBAL_S0: {USE_GLOBAL_S0}\n")
                ff.write(f"Poisson Mode Name: {poisson_mode_str}\n")
                ff.write(f"s0 (global mode only): {time_factor if USE_GLOBAL_S0 else np.nan}\n")
                ff.write(f"Counts Clip Min/Max: min={COUNTS_MIN}, max={COUNTS_CAP}\n")
                ff.write(f"scale_i Clip Min/Max (effective mode): min={SCALE_I_MIN}, max={SCALE_I_MAX}\n")
                ff.write(f"Scale factor from time_factor in effective mode: {time_factor/50000.0}\n")
                ff.write(f"Background sigma absolute (bg_sigma_rel*bg_cm): {bg_sigma}\n")
                ff.write(f"Sampled background offset: {bg_offset}\n")
                ff.write(f"rel_extra_const2 (ripple^2 + stitch^2): {rel_extra_const2}\n")
                ff.write(f"Stitch edges raw input: {stitch_edges_str}\n")
                ff.write(f"Stitch edges parsed: {stitch_edges_q}\n")
                ff.write(f"Stitch segment scales sampled (3 bands): {scales.tolist()}\n")
                ff.write(
                    "Ripple realization stats "
                    f"(mean/std/min/max): {float(np.mean(ripple))}, {float(np.std(ripple))}, "
                    f"{float(np.min(ripple))}, {float(np.max(ripple))}\n"
                )
                ff.write(
                    "Derived arrays min/max: "
                    f"I_base=({float(np.min(I_base))}, {float(np.max(I_base))}), "
                    f"I_scatt_ideal=({float(np.min(I_scatt_ideal))}, {float(np.max(I_scatt_ideal))}), "
                    f"I_sys=({float(np.min(I_sys))}, {float(np.max(I_sys))}), "
                    f"sigma_sys=({float(np.min(sigma_sys))}, {float(np.max(sigma_sys))}), "
                    f"sigma_target=({float(np.min(sigma_target))}, {float(np.max(sigma_target))}), "
                    f"sigma_stat=({float(np.min(sigma_stat))}, {float(np.max(sigma_stat))}), "
                    f"sigma_total=({float(np.min(sigma_total))}, {float(np.max(sigma_total))})\n"
                )
                ff.write(
                    "PSD normalization check: "
                    f"sum(phi_bin)={float(np.sum(phi_bin))}, "
                    f"sum(dphi_dlnr*dlnr)={float(np.sum(dphi_dlnr * dlnr))}\n"
                )
                ff.write("\n=== Diagnostics ===\n")
                ff.write("scale_i pct[1,50,99]: " + str(pct(scale_i)) + "\n")
                ff.write("expected_counts pct[1,50,99]: " + str(pct(expected_counts)) + f" max={np.max(expected_counts)}\n")
                ff.write("sigma_total/I pct[1,50,99]: " + str(pct(sigma_total/np.maximum(I_noisy,1e-30))) + "\n")

            messagebox.showinfo("Success", f"Synthetic data generated!\nSaved to:\n{out_dir}")
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-topmost", True)
    app = SyntheticApp(root)
    root.mainloop()
