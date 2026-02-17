# src/core.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root


@dataclass
class Params:
    # lengths (mm)
    L1: float = 28.0
    L2: float = 30.017
    L_tip: float = 565.0

    # offsets (mm)
    D1: float = 9.042
    D2: float = 30.0
    H1: float = 26.5
    H2: float = 1.0

    # tip offset (rad)
    delta_tip: float = np.deg2rad(1.909)

    # initial guesses (rad)
    theta1_0: float = 0.0
    theta2_0: float = 0.0

    def finalize(self) -> None:
        """Compute derived quantities and initial guesses."""
        self.Btot = self.H1 + self.H2
        # Use radians (atan2 returns rad). Variable names are RAD here.
        self.theta1_0 = float(np.arctan2(self.H1, self.D1))
        self.theta2_0 = float(np.arctan2(self.H2, -self.D2))


@dataclass
class StrokeCfg:
    t_end: float = 0.160
    Nt: int = 3000
    S34_0: float = 0.0
    S34_f: float = 50.0


def stroke_constant_accel(cfg: StrokeCfg):
    """Constant-acceleration stroke profile (same as MATLAB)."""
    t = np.linspace(0.0, cfg.t_end, int(cfg.Nt))
    a = 2.0 * (cfg.S34_f - cfg.S34_0) / (cfg.t_end ** 2)
    S34 = cfg.S34_0 + 0.5 * a * t ** 2
    dS34 = a * t
    ddS34 = np.full_like(t, a)
    return t, S34, dS34, ddS34


def position_eq(x: NDArray[np.float64], S34: float, p: Params) -> NDArray[np.float64]:
    """Position equations (schematic)."""
    th1 = float(x[0])
    th2 = float(x[1])

    S34_phys = S34 + p.D1  # same as MATLAB

    F1 = p.D2 - S34_phys + p.L1 * np.cos(th1) + p.L2 * np.cos(th2)
    F2 = -p.Btot + p.L1 * np.sin(th1) + p.L2 * np.sin(th2)
    return np.array([F1, F2], dtype=np.float64)


def unwrap_angle(th_new: float, th_old: float) -> float:
    th = th_new
    while th - th_old > np.pi:
        th -= 2.0 * np.pi
    while th - th_old < -np.pi:
        th += 2.0 * np.pi
    return th


def solve_theta(x0: NDArray[np.float64], S34k: float, p: Params, k: int,
                th1_prev: float | None, th2_prev: float | None):
    """Solve for theta1, theta2 using scipy.root (fsolve-like)."""
    fun = lambda x: position_eq(np.asarray(x, dtype=np.float64), S34k, p)

    sol = root(fun, x0, method="hybr", tol=1e-12)

    th1 = float(sol.x[0])
    th2 = float(sol.x[1])

    ok = bool(sol.success)

    if (k > 0) and (th1_prev is not None) and (th2_prev is not None):
        th1 = unwrap_angle(th1, th1_prev)
        th2 = unwrap_angle(th2, th2_prev)

    x0_next = np.array([th1, th2], dtype=np.float64)
    return th1, th2, x0_next, ok


def velocity_accel(th1: float, th2: float, dS34: float, ddS34: float, p: Params):
    """Compute angular velocity and acceleration (same equations as MATLAB)."""
    L1, L2 = p.L1, p.L2

    J = np.array(
        [
            [-L1 * np.sin(th1), -L2 * np.sin(th2)],
            [ L1 * np.cos(th1),  L2 * np.cos(th2)],
        ],
        dtype=np.float64,
    )

    rhs_v = np.array([dS34, 0.0], dtype=np.float64)
    w = np.linalg.solve(J, rhs_v)
    dth1, dth2 = float(w[0]), float(w[1])

    term_x = L1 * (dth1 ** 2) * np.cos(th1) + L2 * (dth2 ** 2) * np.cos(th2)
    term_y = L1 * (dth1 ** 2) * np.sin(th1) + L2 * (dth2 ** 2) * np.sin(th2)

    rhs_a = np.array([ddS34 + term_x, term_y], dtype=np.float64)
    a = np.linalg.solve(J, rhs_a)
    ddth1, ddth2 = float(a[0]), float(a[1])

    return dth1, dth2, ddth1, ddth2


def simulate(p: Params, cfg: StrokeCfg):
    """Full simulation producing outputs analogous to MATLAB out struct."""
    p.finalize()
    t, S34, dS34, ddS34 = stroke_constant_accel(cfg)
    Nt = t.size

    theta1 = np.zeros(Nt, dtype=np.float64)
    theta2 = np.zeros(Nt, dtype=np.float64)
    dtheta1 = np.zeros(Nt, dtype=np.float64)
    dtheta2 = np.zeros(Nt, dtype=np.float64)
    ddtheta1 = np.zeros(Nt, dtype=np.float64)
    ddtheta2 = np.zeros(Nt, dtype=np.float64)
    ok_flags = np.ones(Nt, dtype=bool)

    x0 = np.array([p.theta1_0, p.theta2_0], dtype=np.float64)

    for k in range(Nt):
        th1_prev = theta1[k - 1] if k > 0 else None
        th2_prev = theta2[k - 1] if k > 0 else None

        th1, th2, x0, ok = solve_theta(x0, float(S34[k]), p, k, th1_prev, th2_prev)
        ok_flags[k] = ok

        theta1[k] = th1
        theta2[k] = th2

        dth1, dth2, ddth1, ddth2 = velocity_accel(th1, th2, float(dS34[k]), float(ddS34[k]), p)
        dtheta1[k] = dth1
        dtheta2[k] = dth2
        ddtheta1[k] = ddth1
        ddtheta2[k] = ddth2

    theta_tip = theta2 + p.delta_tip
    dtheta_tip = dtheta2
    ddtheta_tip = ddtheta2

    v_tan = p.L_tip * dtheta_tip
    v_norm = np.zeros_like(v_tan)
    a_tan = p.L_tip * ddtheta_tip
    a_norm = p.L_tip * (dtheta_tip ** 2)

    out = {
        "t": t,
        "S34": S34,
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
        "ok": ok_flags,
    }
    return out

