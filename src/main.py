# src/main.py
from __future__ import annotations

import sys
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTabWidget, QMessageBox, QGroupBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core import Params, StrokeCfg, simulate


def _float(le: QLineEdit, name: str) -> float:
    try:
        return float(le.text().strip())
    except Exception:
        raise ValueError(f"{name} sayısal olmalı.")


def _int(le: QLineEdit, name: str) -> int:
    try:
        v = int(float(le.text().strip()))
        return v
    except Exception:
        raise ValueError(f"{name} tam sayı olmalı.")


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mechanism Analyzer (S34 - Constant Accel)")
        self.resize(1100, 750)

        root = QVBoxLayout(self)

        # -------- Inputs --------
        box = QGroupBox("Girdiler (mm, s, deg)")
        grid = QGridLayout(box)

        # Defaults from your MATLAB
        self.le_L1 = QLineEdit("28")
        self.le_L2 = QLineEdit("30.017")
        self.le_Ltip = QLineEdit("565")

        self.le_D1 = QLineEdit("9.042")
        self.le_D2 = QLineEdit("30")
        self.le_H1 = QLineEdit("26.5")
        self.le_H2 = QLineEdit("1")

        self.le_delta_tip_deg = QLineEdit("1.909")

        self.le_t_end = QLineEdit("0.160")
        self.le_Nt = QLineEdit("3000")
        self.le_S0 = QLineEdit("0")
        self.le_Sf = QLineEdit("50")

        row = 0
        def add(name, widget):
            nonlocal row
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(widget, row, 1)
            row += 1

        add("L1 (A-B)", self.le_L1)
        add("L2 (O-B)", self.le_L2)
        add("L_tip", self.le_Ltip)
        add("D1", self.le_D1)
        add("D2", self.le_D2)
        add("H1", self.le_H1)
        add("H2", self.le_H2)
        add("delta_tip (deg)", self.le_delta_tip_deg)
        add("t_end (s)", self.le_t_end)
        add("Nt", self.le_Nt)
        add("S34_0 (mm)", self.le_S0)
        add("S34_f (mm)", self.le_Sf)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Hesapla")
        self.btn_run.clicked.connect(self.on_run)
        btn_row.addWidget(self.btn_run)

        self.lbl_summary = QLabel("Hazır.")
        self.lbl_summary.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        btn_row.addWidget(self.lbl_summary, 1)

        root.addWidget(box)
        root.addLayout(btn_row)

        # -------- Tabs with plots --------
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # Tab 1: theta/omega/alpha vs S34 (3 subplots)
        self.canvas1 = MplCanvas()
        self.ax1 = self.canvas1.fig.add_subplot(311)
        self.ax2 = self.canvas1.fig.add_subplot(312)
        self.ax3 = self.canvas1.fig.add_subplot(313)
        self.canvas1.fig.tight_layout()

        w1 = QWidget()
        l1 = QVBoxLayout(w1)
        l1.addWidget(self.canvas1)
        self.tabs.addTab(w1, "Açı / Hız / İvme")

        # Tab 2: v and a components (2 subplots)
        self.canvas2 = MplCanvas()
        self.ax4 = self.canvas2.fig.add_subplot(211)
        self.ax5 = self.canvas2.fig.add_subplot(212)
        self.canvas2.fig.tight_layout()

        w2 = QWidget()
        l2 = QVBoxLayout(w2)
        l2.addWidget(self.canvas2)
        self.tabs.addTab(w2, "Tip t-n (v,a)")

    def on_run(self):
        try:
            p = Params(
                L1=_float(self.le_L1, "L1"),
                L2=_float(self.le_L2, "L2"),
                L_tip=_float(self.le_Ltip, "L_tip"),
                D1=_float(self.le_D1, "D1"),
                D2=_float(self.le_D2, "D2"),
                H1=_float(self.le_H1, "H1"),
                H2=_float(self.le_H2, "H2"),
                delta_tip=np.deg2rad(_float(self.le_delta_tip_deg, "delta_tip")),
            )
            cfg = StrokeCfg(
                t_end=_float(self.le_t_end, "t_end"),
                Nt=_int(self.le_Nt, "Nt"),
                S34_0=_float(self.le_S0, "S34_0"),
                S34_f=_float(self.le_Sf, "S34_f"),
            )

            if cfg.Nt < 10:
                raise ValueError("Nt en az 10 olmalı.")
            if cfg.t_end <= 0:
                raise ValueError("t_end > 0 olmalı.")
            if p.L1 <= 0 or p.L2 <= 0 or p.L_tip <= 0:
                raise ValueError("Uzunluklar > 0 olmalı.")

            out = simulate(p, cfg)

            # Summary
            theta_tip_end = np.rad2deg(out["theta_tip"][-1])
            s_end = out["S34"][-1]
            ok_ratio = float(np.mean(out["ok"])) * 100.0
            self.lbl_summary.setText(
                f"Son: S34={s_end:.3f} mm | theta_tip={theta_tip_end:.3f} deg | Çözüm başarı oranı={ok_ratio:.1f}%"
            )

            self.update_plots(out)

            if ok_ratio < 99.0:
                QMessageBox.warning(
                    self,
                    "Uyarı",
                    "Bazı adımlarda kök bulma başarısız olmuş görünüyor. Parametreleri kontrol edin veya Nt/t_end ayarlarını değiştirin."
                )

        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))

    def update_plots(self, out: dict):
        S = out["S34"]

        theta_tip_deg = np.rad2deg(out["theta_tip"])
        dtheta_tip_deg = np.rad2deg(out["dtheta_tip"])
        ddtheta_tip_deg = np.rad2deg(out["ddtheta_tip"])

        # Tab 1
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax1.plot(S, theta_tip_deg, linewidth=2)
        self.ax1.set_ylabel(r"$\theta_{tip}$ (deg)")
        self.ax1.grid(True)
        self.ax1.set_title(rf"$\theta_{{tip}}$ vs $S_{{34}}$ | Son: {theta_tip_deg[-1]:.2f} deg")

        self.ax2.plot(S, dtheta_tip_deg, linewidth=2)
        self.ax2.set_ylabel(r"$\omega_{tip}$ (deg/s)")
        self.ax2.grid(True)

        self.ax3.plot(S, ddtheta_tip_deg, linewidth=2)
        self.ax3.set_xlabel(r"$S_{34}$ (mm)")
        self.ax3.set_ylabel(r"$\alpha_{tip}$ (deg/s$^2$)")
        self.ax3.grid(True)

        self.canvas1.fig.tight_layout()
        self.canvas1.draw()

        # Tab 2
        self.ax4.clear()
        self.ax5.clear()

        self.ax4.plot(S, out["v_tan"], linewidth=2, label="v_t (tanj.)")
        self.ax4.plot(S, out["v_norm"], linewidth=1.5, linestyle="--", label="v_n (0)")
        self.ax4.set_xlabel(r"$S_{34}$ (mm)")
        self.ax4.set_ylabel("v (mm/s)")
        self.ax4.grid(True)
        self.ax4.set_title("Tip Hız Bileşenleri (t-n)")
        self.ax4.legend(loc="best")

        self.ax5.plot(S, out["a_tan"], linewidth=2, label="a_t (tanj.)")
        self.ax5.plot(S, out["a_norm"], linewidth=1.5, linestyle="--", label="a_n (merkezcil)")
        self.ax5.set_xlabel(r"$S_{34}$ (mm)")
        self.ax5.set_ylabel(r"a (mm/s$^2$)")
        self.ax5.grid(True)
        self.ax5.set_title("Tip İvme Bileşenleri (t-n)")
        self.ax5.legend(loc="best")

        self.canvas2.fig.tight_layout()
        self.canvas2.draw()


def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

