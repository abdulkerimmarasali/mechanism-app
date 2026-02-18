# core.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# =========================
# Data models
# =========================
@dataclass
class Params:
    # Geometry (mm)
    L1: float          # A-B
    L2: float          # O-B
    L_tip: float       # blade length (O reference note handled in UI)

    # Offsets (mm)
    D1: float          # A x-offset
    D2: float          # O x-offset
    H1: float
    H2: float          # used for delta_tip = atan(H2/D2)


@dataclass
class StrokeCfg:
    t_end: float
    Nt: int
    stroke_mm: float   # 0 -> stroke_mm


# =========================
# Public API
# =========================
def simulate(p: Params, cfg: StrokeCfg) -> dict:
    """
    Runs kinematics simulation with:
    - constant acceleration stroke profile: S(t) from 0 -> stroke_mm
    - per-step Newton solve for theta1, theta2 (2 eq, 2 unknown)
    - velocity & acceleration from linear solves using Jacobian

    Returns dict with arrays used by UI:
      t, S34, dS34, ddS34,
      theta1, theta2, dtheta1, dtheta2, ddtheta1, ddtheta2,
      theta_tip, dtheta_tip, ddtheta_tip,
      v_tan, v_norm, a_tan, a_norm,
      ok (bool array)
    """
    _validate_inputs(p, cfg)

    t, S34, dS34, ddS34 = stroke_profile(cfg)

    Nt = t.size

    theta1 = np.zeros(Nt, dtype=float)
    theta2 = np.zeros(Nt, dtype=float)
    dtheta1 = np.zeros(Nt, dtype=float)
    dtheta2 = np.zeros(Nt, dtype=float)
    ddtheta1 = np.zeros(Nt, dtype=float)
    ddtheta2 = np.zeros(Nt, dtype=float)
    ok = np.ones(Nt, dtype=bool)

    # Initial guess (radians) - consistent with original MATLAB idea
    th1_0 = np.arctan2(p.H1, p.D1)           # ~ atan(H1/D1)
    th2_0 = np.arctan2(p.H2, -p.D2)          # ~ atan(H2/(-D2))
    x = np.array([th1_0, th2_0], dtype=float)

    # Solve theta at each time step
    for k in range(Nt):
        S = float(S34[k])

        x, ok_k = solve_theta_newton(x, S, p)
        ok[k] = ok_k

        if k == 0:
            theta1[k], theta2[k] = x
        else:
            theta1[k] = unwrap_angle(x[0], theta1[k - 1])
            theta2[k] = unwrap_angle(x[1], theta2[k - 1])
            x = np.array([theta1[k], theta2[k]], dtype=float)

        # Vel/accel
        dth1, dth2, ddth1, ddth2 = velocity_accel(
            theta1[k], theta2[k],
            float(dS34[k]), float(ddS34[k]),
            p
        )
        dtheta1[k], dtheta2[k] = dth1, dth2
        ddtheta1[k], ddtheta2[k] = ddth1, ddth2

    # delta_tip is automatic: atan(H2/D2)
    delta_tip = np.arctan2(p.H2, p.D2)

    theta_tip = theta2 + delta_tip
    dtheta_tip = dtheta2.copy()
    ddtheta_tip = ddtheta2.copy()

    # tip kinematics (t-n)
    v_tan = p.L_tip * dtheta_tip
    v_norm = np.zeros_like(v_tan)
    a_tan = p.L_tip * ddtheta_tip
    a_norm = p.L_tip * (dtheta_tip ** 2)

    out = {
        "t": t,
        "S34": S34,
        "dS34": dS34,
        "ddS34": ddS34,

        "theta1": theta1,
        "theta2": theta2,
        "dtheta1": dtheta1,
        "dtheta2": dtheta2,
        "ddtheta1": ddtheta1,
        "ddtheta2": ddtheta2,

        "theta_tip": theta_tip,
        "dtheta_tip": dtheta_tip,
        "ddtheta_tip": ddtheta_tip,

        "v_tan": v_tan,
        "v_norm": v_norm,
        "a_tan": a_tan,
        "a_norm": a_norm,

        "ok": ok,
    }
    return out


# =========================
# Stroke profile (constant acceleration)
# =========================
def stroke_profile(cfg: StrokeCfg):
    t = np.linspace(0.0, cfg.t_end, int(cfg.Nt), dtype=float)

    S0 = 0.0
    Sf = float(cfg.stroke_mm)
    a = 2.0 * (Sf - S0) / (cfg.t_end ** 2)

    S = S0 + 0.5 * a * (t ** 2)
    dS = a * t
    ddS = a * np.ones_like(t)

    return t, S, dS, ddS


# =========================
# Newton solve for theta
# =========================
def solve_theta_newton(x0: np.ndarray, S34: float, p: Params,
                      max_iter: int = 50, tol_f: float = 1e-12, tol_x: float = 1e-12):
    x = np.array(x0, dtype=float)

    for _ in range(max_iter):
        F = position_eq(x, S34, p)
        nF = float(np.linalg.norm(F, ord=2))
        if nF < tol_f:
            return x, True

        J = position_jacobian(x, p)

        # Solve J dx = -F
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            # singular Jacobian
            return x, False

        x = x + dx

        if float(np.linalg.norm(dx, ord=2)) < tol_x:
            # step tiny: accept
            return x, True

    # no converge
    return x, False


def position_eq(x: np.ndarray, S34: float, p: Params) -> np.ndarray:
    th1, th2 = float(x[0]), float(x[1])

    # original MATLAB: S34_phys = S34 + D1
    S_phys = S34 + p.D1
    Btot = p.H1 + p.H2

    F1 = p.D2 - S_phys + p.L1 * np.cos(th1) + p.L2 * np.cos(th2)
    F2 = -Btot + p.L1 * np.sin(th1) + p.L2 * np.sin(th2)
    return np.array([F1, F2], dtype=float)


def position_jacobian(x: np.ndarray, p: Params) -> np.ndarray:
    th1, th2 = float(x[0]), float(x[1])
    J11 = -p.L1 * np.sin(th1)
    J12 = -p.L2 * np.sin(th2)
    J21 =  p.L1 * np.cos(th1)
    J22 =  p.L2 * np.cos(th2)
    return np.array([[J11, J12], [J21, J22]], dtype=float)


# =========================
# Vel / Accel from Jacobian
# =========================
def velocity_accel(th1: float, th2: float, dS34: float, ddS34: float, p: Params):
    J = np.array([
        [-p.L1 * np.sin(th1), -p.L2 * np.sin(th2)],
        [ p.L1 * np.cos(th1),  p.L2 * np.cos(th2)],
    ], dtype=float)

    # velocity
    rhs_v = np.array([dS34, 0.0], dtype=float)
    w = np.linalg.solve(J, rhs_v)
    dth1, dth2 = float(w[0]), float(w[1])

    # acceleration
    term_x = p.L1 * (dth1 ** 2) * np.cos(th1) + p.L2 * (dth2 ** 2) * np.cos(th2)
    term_y = p.L1 * (dth1 ** 2) * np.sin(th1) + p.L2 * (dth2 ** 2) * np.sin(th2)

    rhs_a = np.array([ddS34 + term_x, term_y], dtype=float)
    a = np.linalg.solve(J, rhs_a)
    ddth1, ddth2 = float(a[0]), float(a[1])

    return dth1, dth2, ddth1, ddth2


# =========================
# Angle unwrap
# =========================
def unwrap_angle(th_new: float, th_old: float) -> float:
    th = float(th_new)
    while th - th_old > np.pi:
        th -= 2.0 * np.pi
    while th - th_old < -np.pi:
        th += 2.0 * np.pi
    return th


# =========================
# Validation
# =========================
def _validate_inputs(p: Params, cfg: StrokeCfg):
    for name in ("L1", "L2", "L_tip", "D1", "D2", "H1", "H2"):
        v = getattr(p, name)
        if not np.isfinite(v):
            raise ValueError(f"{name} geçersiz (NaN/inf).")

    if p.L1 <= 0 or p.L2 <= 0 or p.L_tip <= 0:
        raise ValueError("L1, L2, L_tip > 0 olmalı.")
    if cfg.t_end <= 0:
        raise ValueError("Hareket Süresi (t_end) > 0 olmalı.")
    if cfg.Nt < 2:
        raise ValueError("Zaman Adım Sayısı (Nt) >= 2 olmalı.")
    if cfg.stroke_mm <= 0:
        raise ValueError("Stroke > 0 olmalı.")
