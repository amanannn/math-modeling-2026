"""
Purpose: Comprehensive demo of interpolation and fitting methods for mathematical modeling.
         Covers polynomial fitting (overfitting demo), cubic spline interpolation,
         nonlinear curve_fit, and a real-case experimental data fitting example.

Input:  No external input; synthetic/example data is generated inside the script.
Output: Console output with key numerical results and a 2x2 figure saved to PNG.
        Figure path: D:/虚拟C盘/数学建模培训/output/interpolation_result.png

Usage:  python interpolation_demo.py
        Requires: numpy, scipy, matplotlib (Agg backend)

Author: Programming Lead / Math Modeling Team
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========== Style settings ==========
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

OUTPUT_PATH = "D:/虚拟C盘/数学建模培训/output/interpolation_result.png"
np.random.seed(42)


# ====================================================================
# Section 1: Polynomial fitting and the overfitting problem
# ====================================================================
def section_1_polynomial_overfitting():
    """
    Demonstrate polynomial fitting with the Runge function
    f(x) = 1 / (1 + 25*x^2) on [-1, 1].
    Low-degree (5) fits reasonably; high-degree (15) overfits badly.
    """
    print("=" * 65)
    print("SECTION 1: Polynomial Fitting & Overfitting (Runge Phenomenon)")
    print("=" * 65)

    def runge(x):
        return 1.0 / (1.0 + 25.0 * x**2)

    x_fine = np.linspace(-1, 1, 500)
    y_true = runge(x_fine)

    # --- Low-degree fit: 5 data points, degree 4 ---
    x_pts_low = np.linspace(-1, 1, 5)
    y_pts_low = runge(x_pts_low)
    coeff_low = np.polyfit(x_pts_low, y_pts_low, deg=4)
    y_fit_low = np.polyval(coeff_low, x_fine)
    err_low = np.max(np.abs(y_fit_low - y_true))
    print(f"  Degree-4 fit (5 points):  max error = {err_low:.4f}")

    # --- Medium-degree fit: 10 data points, degree 9 ---
    x_pts_mid = np.linspace(-1, 1, 10)
    y_pts_mid = runge(x_pts_mid)
    coeff_mid = np.polyfit(x_pts_mid, y_pts_mid, deg=9)
    y_fit_mid = np.polyval(coeff_mid, x_fine)
    err_mid = np.max(np.abs(y_fit_mid - y_true))
    print(f"  Degree-9 fit (10 points): max error = {err_mid:.4f}")

    # --- Severe overfit: 16 data points, degree 15 ---
    x_pts_high = np.linspace(-1, 1, 16)
    y_pts_high = runge(x_pts_high)
    coeff_high = np.polyfit(x_pts_high, y_pts_high, deg=15)
    y_fit_high = np.polyval(coeff_high, x_fine)
    err_high = np.max(np.abs(y_fit_high - y_true))
    print(f"  Degree-15 fit (16 points): max error = {err_high:.4f}  <-- severe oscillation at edges")
    print()

    return x_fine, y_true, (x_pts_low, y_pts_low, y_fit_low,
                            x_pts_mid, y_pts_mid, y_fit_mid,
                            x_pts_high, y_pts_high, y_fit_high)


# ====================================================================
# Section 2: Cubic spline interpolation
# ====================================================================
def section_2_cubic_spline():
    """
    Cubic spline interpolation on temperature-vs-time data.
    Compare 'natural' and 'not-a-knot' boundary conditions.
    """
    print("=" * 65)
    print("SECTION 2: Cubic Spline Interpolation")
    print("=" * 65)

    # Simulated temperature data (every 2 hours)
    hours = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
    temp = np.array([15.2, 14.8, 14.5, 16.1, 18.5, 21.3, 23.8,
                     24.5, 23.9, 21.2, 18.6, 16.5, 15.0])

    # Create splines with different boundary conditions
    cs_natural = CubicSpline(hours, temp, bc_type="natural")
    cs_notaknot = CubicSpline(hours, temp, bc_type="not-a-knot")

    # Fine grid for plotting (every 0.1 hour)
    hours_fine = np.linspace(0, 24, 241)
    temp_natural = cs_natural(hours_fine)
    temp_notaknot = cs_notaknot(hours_fine)

    # First derivative (temperature rate of change)
    temp_rate = cs_natural(hours_fine, nu=1)

    # Find the time of maximum temperature
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda t: -cs_natural(t), bounds=(0, 24), method="bounded")
    print(f"  Max temperature:  T = {-res.fun:.2f} C  at  t = {res.x:.2f} h")
    print(f"  Data points: {len(hours)},  interpolation points: {len(hours_fine)}")
    print()

    return (hours, temp, hours_fine, temp_natural, temp_notaknot, temp_rate, cs_natural)


# ====================================================================
# Section 3: Nonlinear curve fitting (curve_fit)
# ====================================================================
def section_3_nonlinear_curve_fit():
    """
    Nonlinear least-squares fitting with curve_fit.
    Example: Drug concentration decay follows an exponential model.
    """
    print("=" * 65)
    print("SECTION 3: Nonlinear Curve Fitting (curve_fit)")
    print("=" * 65)

    # Simulated drug concentration data
    t = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12])
    conc = np.array([100, 62, 40, 26, 18, 13, 10, 6.5, 4.5, 3.5])

    # Model 1: Exponential decay  C(t) = C0 * exp(-k * t)
    def exp_decay(t, C0, k):
        return C0 * np.exp(-k * t)

    popt1, pcov1 = curve_fit(exp_decay, t, conc, p0=[100, 0.2])
    C0_opt, k_opt = popt1
    C0_err, k_err = np.sqrt(np.diag(pcov1))
    half_life = np.log(2) / k_opt

    y_pred1 = exp_decay(t, *popt1)
    ss_res1 = np.sum((conc - y_pred1)**2)
    ss_tot1 = np.sum((conc - np.mean(conc))**2)
    r2_1 = 1 - ss_res1 / ss_tot1
    rmse_1 = np.sqrt(ss_res1 / len(t))

    print(f"  Model: C(t) = C0 * exp(-k * t)")
    print(f"    C0 = {C0_opt:.2f} +/- {C0_err:.2f}")
    print(f"    k  = {k_opt:.4f} +/- {k_err:.4f}")
    print(f"    Half-life = {half_life:.2f} h")
    print(f"    R^2 = {r2_1:.4f},  RMSE = {rmse_1:.4f}")

    # Model 2: Exponential decay with offset  C(t) = C0 * exp(-k * t) + b
    def exp_decay_offset(t, C0, k, b):
        return C0 * np.exp(-k * t) + b

    popt2, pcov2 = curve_fit(exp_decay_offset, t, conc, p0=[100, 0.2, 0])
    y_pred2 = exp_decay_offset(t, *popt2)
    ss_res2 = np.sum((conc - y_pred2)**2)
    r2_2 = 1 - ss_res2 / ss_tot1
    rmse_2 = np.sqrt(ss_res2 / len(t))
    print(f"  Model: C(t) = C0 * exp(-k * t) + b")
    print(f"    C0 = {popt2[0]:.2f},  k = {popt2[1]:.4f},  b = {popt2[2]:.4f}")
    print(f"    R^2 = {r2_2:.4f},  RMSE = {rmse_2:.4f}")

    # Prediction on fine grid for plotting
    t_fine = np.linspace(0, 12, 100)
    c_fine1 = exp_decay(t_fine, *popt1)
    c_fine2 = exp_decay_offset(t_fine, *popt2)

    print()
    return t, conc, t_fine, c_fine1, c_fine2


# ====================================================================
# Section 4: Real-case example — fitting experimental stress-strain data
# ====================================================================
def section_4_real_case():
    """
    Real-case example: material stress-strain curve fitting.
    Compare quadratic, power-law, and exponential-offset models.
    """
    print("=" * 65)
    print("SECTION 4: Real-Case Example - Stress-Strain Curve Fitting")
    print("=" * 65)

    # Experimental data: stress (MPa) vs. strain (%)
    stress = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    strain = np.array([0.00, 0.03, 0.08, 0.15, 0.24, 0.36,
                       0.50, 0.68, 0.88, 1.12, 1.40])

    # --- Model 1: Quadratic polynomial  y = a*x^2 + b*x + c ---
    coeff_quad = np.polyfit(stress, strain, deg=2)
    strain_quad = np.polyval(coeff_quad, stress)
    ss_res_q = np.sum((strain - strain_quad)**2)
    ss_tot = np.sum((strain - np.mean(strain))**2)
    r2_q = 1 - ss_res_q / ss_tot
    rmse_q = np.sqrt(ss_res_q / len(stress))
    print(f"  Model 1 - Quadratic:  y = {coeff_quad[0]:.6f} x^2 + {coeff_quad[1]:.6f} x + {coeff_quad[2]:.6f}")
    print(f"    R^2 = {r2_q:.4f},  RMSE = {rmse_q:.4f}")

    # --- Model 2: Power law  y = a * x^b ---
    def power_law(x, a, b):
        return a * x**b

    popt_pl, _ = curve_fit(power_law, stress[1:], strain[1:], p0=[0.001, 1.5])
    # exclude stress=0 point for power law (log issue)
    strain_pl = np.zeros_like(stress)
    strain_pl[1:] = power_law(stress[1:], *popt_pl)
    ss_res_pl = np.sum((strain - strain_pl)**2)
    r2_pl = 1 - ss_res_pl / ss_tot
    rmse_pl = np.sqrt(ss_res_pl / len(stress))
    print(f"  Model 2 - Power law:  y = {popt_pl[0]:.6f} * x^{popt_pl[1]:.4f}")
    print(f"    R^2 = {r2_pl:.4f},  RMSE = {rmse_pl:.4f}")

    # --- Model 3: Exponential offset  y = a * (exp(b*x) - 1) ---
    def exp_off(x, a, b):
        return a * (np.exp(b * x) - 1)

    popt_eo, _ = curve_fit(exp_off, stress, strain, p0=[0.1, 0.02])
    strain_eo = exp_off(stress, *popt_eo)
    ss_res_eo = np.sum((strain - strain_eo)**2)
    r2_eo = 1 - ss_res_eo / ss_tot
    rmse_eo = np.sqrt(ss_res_eo / len(stress))
    print(f"  Model 3 - Exponential off.:  y = {popt_eo[0]:.4f} * (exp({popt_eo[1]:.4f} * x) - 1)")
    print(f"    R^2 = {r2_eo:.4f},  RMSE = {rmse_eo:.4f}")

    # Fine grid for smooth curves
    stress_fine = np.linspace(0, 100, 200)
    strain_fine_q = np.polyval(coeff_quad, stress_fine)
    strain_fine_pl = np.zeros_like(stress_fine)
    strain_fine_pl[1:] = power_law(stress_fine[1:], *popt_pl)
    strain_fine_eo = exp_off(stress_fine, *popt_eo)

    # Identify the best model
    models_r2 = {"Quadratic": r2_q, "Power law": r2_pl, "Exp. offset": r2_eo}
    best = max(models_r2, key=models_r2.get)
    print(f"\n  >>> Best model (by R^2): {best}  (R^2 = {models_r2[best]:.4f})")
    print()

    return (stress, strain, stress_fine,
            strain_fine_q, strain_fine_pl, strain_fine_eo,
            coeff_quad, popt_pl, popt_eo,
            strain_quad, strain_pl, strain_eo)


# ====================================================================
# Main: run all sections and create the composite figure
# ====================================================================
def main():
    print("=" * 65)
    print("  INTERPOLATION & FITTING DEMO")
    print("  Mathematical Modeling - National Competition Training")
    print("=" * 65)
    print()

    # Run all four sections
    res1 = section_1_polynomial_overfitting()
    res2 = section_2_cubic_spline()
    res3 = section_3_nonlinear_curve_fit()
    res4 = section_4_real_case()

    # Unpack results
    (x_fine, y_true,
     (x_low, y_low, y_fit_low,
      x_mid, y_mid, y_fit_mid,
      x_high, y_high, y_fit_high)) = res1

    (hours, temp, hours_fine, temp_natural, temp_notaknot, temp_rate, cs_natural) = res2
    (t_drug, conc_drug, t_fine_drug, c_fine1, c_fine2) = res3
    (stress, strain, stress_fine,
     strain_q, strain_pl, strain_eo,
     coeff_q, popt_pl, popt_eo,
     strain_quad_pts, strain_pl_pts, strain_eo_pts) = res4

    # ========== Build 2x2 figure ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Panel A: Polynomial overfitting ----
    ax = axes[0, 0]
    ax.plot(x_fine, y_true, "k-", linewidth=2, label="True f(x)")
    ax.plot(x_fine, y_fit_low, "b--", linewidth=1.5, label="Degree 4 (5 pts)")
    ax.plot(x_fine, y_fit_mid, "g-.", linewidth=1.5, label="Degree 9 (10 pts)")
    ax.plot(x_fine, y_fit_high, "r:", linewidth=1.5, label="Degree 15 (16 pts)")
    ax.scatter(x_low, y_low, c="b", s=30, zorder=5, marker="o")
    ax.scatter(x_high, y_high, c="r", s=30, zorder=5, marker="s", alpha=0.5)
    ax.set_ylim(-1.5, 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Polynomial Fitting — Overfitting Demo\n(Runge: $f(x)=1/(1+25x^2)$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- Panel B: Cubic spline interpolation ----
    ax = axes[0, 1]
    ax.plot(hours, temp, "ro", markersize=6, label="Measured data")
    ax.plot(hours_fine, temp_natural, "b-", linewidth=2, label="Natural spline")
    ax.plot(hours_fine, temp_notaknot, "g--", linewidth=1.5, label="Not-a-knot spline")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Cubic Spline Interpolation\n(Temperature vs. Time)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Inset: first derivative
    ax_twin = ax.twinx()
    ax_twin.plot(hours_fine, temp_rate, "k:", linewidth=1, alpha=0.6)
    ax_twin.set_ylabel("dT/dt (C/h)", color="gray", fontsize=8)
    ax_twin.tick_params(colors="gray", labelsize=7)

    # ---- Panel C: Nonlinear curve_fit ----
    ax = axes[1, 0]
    ax.scatter(t_drug, conc_drug, c="red", s=40, label="Measured conc.")
    ax.plot(t_fine_drug, c_fine1, "b-", linewidth=2, label="Exp decay")
    ax.plot(t_fine_drug, c_fine2, "g--", linewidth=1.5, label="Exp decay + offset")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Concentration (mg/L)")
    ax.set_title("Nonlinear Curve Fit\n(Drug Concentration Decay)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- Panel D: Real-case stress-strain fitting ----
    ax = axes[1, 1]

    # Pre-compute R^2 values for clean labels
    ss_tot_d = np.sum((strain - np.mean(strain))**2)
    r2_q_d = 1 - np.sum((strain - strain_quad_pts)**2) / ss_tot_d
    r2_pl_d = 1 - np.sum((strain - strain_pl_pts)**2) / ss_tot_d
    r2_eo_d = 1 - np.sum((strain - strain_eo_pts)**2) / ss_tot_d

    ax.scatter(stress, strain, c="red", s=40, label="Experimental data")
    ax.plot(stress_fine, strain_q, "b-", linewidth=2,
            label=f"Quadratic  (R^2={r2_q_d:.3f})")
    ax.plot(stress_fine, strain_pl, "g-.", linewidth=1.5,
            label=f"Power law  (R^2={r2_pl_d:.3f})")
    ax.plot(stress_fine, strain_eo, "m:", linewidth=1.5,
            label=f"Exp. offset  (R^2={r2_eo_d:.3f})")
    ax.set_xlabel("Stress (MPa)")
    ax.set_ylabel("Strain (%)")
    ax.set_title("Real-Case: Stress-Strain Curve Fitting\n(Experimental Data)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout(pad=3.0)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {OUTPUT_PATH}")
    print()

    print("=" * 65)
    print("  DEMO COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
