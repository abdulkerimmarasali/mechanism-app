# src/main.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QFormLayout,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core import Params, StrokeCfg, simulate


# ----------------------------
# Helpers
# ----------------------------
def to_float(le: QLineEdit, name: str) -> float:
    try:
        return float(le.text().strip())
    except Exception:
        raise ValueError(f"{name} sayısal olmalı.")


def to_int(le: QLineEdit, name: str) -> int:
    try:
        return int(float(le.text().strip()))
    except Exception:
        raise ValueError(f"{name} tam sayı olmalı.")


def resource_path(relative: str) -> Path:
    """
    Resolves a data file path for:
    - dev run (repo layout)
    - PyInstaller one-dir bundle (exe directory)
    - PyInstaller one-file bundle (_MEIPASS)
    """
    # one-file
    if hasattr(sys, "_MEIPASS"):
        base = Path(getattr(sys, "_MEIPASS"))
        return base / relative

    # one-dir: exe dir, or when running python: project root assumptions
    exe_dir = Path(sys.executable).resolve().parent
    p1 = exe_dir / relative
    if p1.exists():
        return p1

    # dev: src/main.py -> project root = parents[1]
    proj = Path(__file__).resolve().parents[1]
    return proj / relative


class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure()
        super().__init__(fig)
        self.fig = fig


# ----------------------------
# Main UI
# ----------------------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mechanism Analyzer")
        self.resize(1200, 780)

        self.out: dict | None = None

        root = QVBoxLayout(self)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        # Tabs
        self.tab_input = QWidget()
        self.tab_summary = QWidget()
        self.tab_theta = QWidget()
        self.tab_omega = QWidget()
        self.tab_alpha = QWidget()
        self.tab_v = QWidget()
        self.tab_a = QWidget()
        self.tab_data = QWidget()

        self.tabs.addTab(self.tab_input, "Giriş")
        self.tabs.addTab(self.tab_summary, "Özet")
        self.tabs.addTab(self.tab_theta, "θ_tip")
        self.tabs.addTab(self.tab_omega, "ω_tip")
        self.tabs.addTab(self.tab_alpha, "α_tip")
        self.tabs.addTab(self.tab_v, "Tip hız")
        self.tabs.addTab(self.tab_a, "Tip ivme")
        self.tabs.addTab(self.tab_data, "Veriler")

        # Disable result tabs until first run
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, False)

        # Build tabs
        self._build_tab_input()
        self._build_tab_summary()
        self._build_tab_plots()
        self._build_tab_data()

    # ----------------------------
    # Tab 1: Input + Image
    # ----------------------------
    def _build_tab_input(self):
        layout = QHBoxLayout(self.tab_input)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Left: compact inputs
        left = QVBoxLayout()
        left.setSpacing(8)

        # Geometry
        box_geo = QGroupBox("Geometri (mm)")
        form_geo = QFormLayout(box_geo)
        form_geo.setLabelAlignment(Qt.AlignLeft)
        self.le_L1 = QLineEdit("28")
        self.le_L2 = QLineEdit("30.017")
        self.le_Ltip = QLineEdit("565")
        form_geo.addRow("L1 (A-B)", self.le_L1)
        form_geo.addRow("L2 (O-B)", self.le_L2)
        form_geo.addRow("L_tip", self.le_Ltip)

        # Offsets
        box_off = QGroupBox("Ofsetler (mm)")
        form_off = QFormLayout(box_off)
        self.le_D1 = QLineEdit("9.042")
        self.le_D2 = QLineEdit("30")
        self.le_H1 = QLineEdit("26.5")
        self.le_H2 = QLineEdit("1")
        self.lbl_Btot = QLabel("Btot: 27.5")
        form_off.addRow("D1", self.le_D1)
        form_off.addRow("D2", self.le_D2)
        form_off.addRow("H1", self.le_H1)
        form_off.addRow("H2", self.le_H2)
        form_off.addRow(self.lbl_Btot)

        # Tip
        box_tip = QGroupBox("Tip")
        form_tip = QFormLayout(box_tip)
        self.le_delta_tip_deg = QLineEdit("1.909")
        form_tip.addRow("delta_tip (deg)", self.le_delta_tip_deg)

        # Stroke
        box_stroke = QGroupBox("Stroke + Zaman")
        form_st = QFormLayout(box_stroke)
        self.le_t_end = QLineEdit("0.160")
        self.le_Nt = QLineEdit("3000")
        self.le_S0 = QLineEdit("0")
        self.le_Sf = QLineEdit("50")
        form_st.addRow("t_end (s)", self.le_t_end)
        form_st.addRow("Nt", self.le_Nt)
        form_st.addRow("S34_0 (mm)", self.le_S0)
        form_st.addRow("S34_f (mm)", self.le_Sf)

        # Buttons + status
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Hesapla")
        self.btn_reset = QPushButton("Reset")
        self.btn_run.clicked.connect(self.on_run)
        self.btn_reset.clicked.connect(self.on_reset)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_reset)

        self.lbl_status = QLabel("Hazır.")
        self.lbl_status.setWordWrap(True)

        # compact left width
        left_container = QWidget()
        left_container.setLayout(left)
        left_container.setMaximumWidth(380)

        left.addWidget(box_geo)
        left.addWidget(box_off)
        left.addWidget(box_tip)
        left.addWidget(box_stroke)
        left.addLayout(btn_row)
        left.addWidget(self.lbl_status)
        left.addStretch(1)

        # Right: image
        right = QVBoxLayout()
        right.setSpacing(6)
        title = QLabel("Mekanizma")
        title.setStyleSheet("font-weight: 600;")
        right.addWidget(title, 0, Qt.AlignLeft)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.img_label)
        right.addWidget(scroll, 1)

        layout.addWidget(left_container, 0)
        layout.addLayout(right, 1)

        # wire up Btot update
        self.le_H1.textChanged.connect(self._update_btot)
        self.le_H2.textChanged.connect(self._update_btot)
        self._update_btot()

        # load image
        self._load_image()

    def _update_btot(self):
        try:
            h1 = float(self.le_H1.text().strip())
            h2 = float(self.le_H2.text().strip())
            self.lbl_Btot.setText(f"Btot: {h1 + h2:.3f}")
        except Exception:
            self.lbl_Btot.setText("Btot: ?")

    def _load_image(self):
        img_path = resource_path("assets/mechanism.png")
        if img_path.exists():
            pix = QPixmap(str(img_path))
            # initial fit to width
            self.img_label.setPixmap(pix.scaledToWidth(780, Qt.SmoothTransformation))
        else:
            self.img_label.setText("assets/mechanism.png bulunamadı.")

    def on_reset(self):
        self.le_L1.setText("28")
        self.le_L2.setText("30.017")
        self.le_Ltip.setText("565")
        self.le_D1.setText("9.042")
        self.le_D2.setText("30")
        self.le_H1.setText("26.5")
        self.le_H2.setText("1")
        self.le_delta_tip_deg.setText("1.909")
        self.le_t_end.setText("0.160")
        self.le_Nt.setText("3000")
        self.le_S0.setText("0")
        self.le_Sf.setText("50")
        self.lbl_status.setText("Hazır.")
        self.tabs.setCurrentIndex(0)

    # ----------------------------
    # Tab 2: Summary
    # ----------------------------
    def _build_tab_summary(self):
        lay = QVBoxLayout(self.tab_summary)
        self.lbl_summary = QLabel("Henüz sonuç yok. Giriş sekmesinden Hesapla.")
        self.lbl_summary.setWordWrap(True)
        lay.addWidget(self.lbl_summary)
        lay.addStretch(1)

    # ----------------------------
    # Plot tabs (each single plot)
    # ----------------------------
    def _build_tab_plots(self):
        # theta
        self.canvas_theta = MplCanvas()
        self.ax_theta = self.canvas_theta.fig.add_subplot(111)
        lay = QVBoxLayout(self.tab_theta)
        lay.addWidget(self.canvas_theta)

        # omega
        self.canvas_omega = MplCanvas()
        self.ax_omega = self.canvas_omega.fig.add_subplot(111)
        lay = QVBoxLayout(self.tab_omega)
        lay.addWidget(self.canvas_omega)

        # alpha
        self.canvas_alpha = MplCanvas()
        self.ax_alpha = self.canvas_alpha.fig.add_subplot(111)
        lay = QVBoxLayout(self.tab_alpha)
        lay.addWidget(self.canvas_alpha)

        # v
        self.canvas_v = MplCanvas()
        self.ax_v = self.canvas_v.fig.add_subplot(111)
        lay = QVBoxLayout(self.tab_v)
        lay.addWidget(self.canvas_v)

        # a
        self.canvas_a = MplCanvas()
        self.ax_a = self.canvas_a.fig.add_subplot(111)
        lay = QVBoxLayout(self.tab_a)
        lay.addWidget(self.canvas_a)

    # ----------------------------
    # Tab: Data table
    # ----------------------------
    def _build_tab_data(self):
        lay = QVBoxLayout(self.tab_data)

        top = QHBoxLayout()
        self.chk_only_failed = QCheckBox("Sadece ok=false satırları")
        self.chk_only_failed.stateChanged.connect(self._refresh_table_if_ready)
        top.addWidget(self.chk_only_failed)
        top.addStretch(1)

        self.table = QTableWidget()
        lay.addLayout(top)
        lay.addWidget(self.table, 1)

    def _refresh_table_if_ready(self):
        if self.out is not None:
            self._update_table(self.out)

    # ----------------------------
    # Run simulation and update tabs
    # ----------------------------
    def on_run(self):
        try:
            self.lbl_status.setText("Hesaplanıyor...")
            self.btn_run.setEnabled(False)
            QApplication.processEvents()

            p = Params(
                L1=to_float(self.le_L1, "L1"),
                L2=to_float(self.le_L2, "L2"),
                L_tip=to_float(self.le_Ltip, "L_tip"),
                D1=to_float(self.le_D1, "D1"),
                D2=to_float(self.le_D2, "D2"),
                H1=to_float(self.le_H1, "H1"),
                H2=to_float(self.le_H2, "H2"),
                delta_tip=np.deg2rad(to_float(self.le_delta_tip_deg, "delta_tip")),
            )
            cfg = StrokeCfg(
                t_end=to_float(self.le_t_end, "t_end"),
                Nt=to_int(self.le_Nt, "Nt"),
                S34_0=to_float(self.le_S0, "S34_0"),
                S34_f=to_float(self.le_Sf, "S34_f"),
            )

            # validation
            if cfg.t_end <= 0:
                raise ValueError("t_end > 0 olmalı.")
            if cfg.Nt < 50:
                raise ValueError("Nt en az 50 olmalı (öneri: 1000+).")
            if p.L1 <= 0 or p.L2 <= 0 or p.L_tip <= 0:
                raise ValueError("L1, L2, L_tip > 0 olmalı.")

            out = simulate(p, cfg)
            self.out = out

            ok_ratio = float(np.mean(out["ok"])) * 100.0
            s_end = float(out["S34"][-1])
            th_end = float(np.rad2deg(out["theta_tip"][-1]))

            self.lbl_status.setText(
                f"Son: S34={s_end:.3f} mm | θ_tip={th_end:.3f} deg | başarı={ok_ratio:.1f}%"
            )

            # enable result tabs
            for i in range(1, self.tabs.count()):
                self.tabs.setTabEnabled(i, True)

            # update all outputs
            self._update_summary(p, cfg, out)
            self._update_plots(out)
            self._update_table(out)

            # jump to summary
            self.tabs.setCurrentIndex(1)

            if ok_ratio < 99.0:
                QMessageBox.warning(
                    self,
                    "Uyarı",
                    "Bazı adımlarda kök bulma başarısız. 'Veriler' sekmesinde ok=false satırlarını kontrol edin."
                )

        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))
            self.lbl_status.setText("Hata oluştu.")
        finally:
            self.btn_run.setEnabled(True)

    def _update_summary(self, p: Params, cfg: StrokeCfg, out: dict):
        S = out["S34"]
        th = np.rad2deg(out["theta_tip"])
        w = np.rad2deg(out["dtheta_tip"])
        a = np.rad2deg(out["ddtheta_tip"])

        txt = []
        txt.append("Parametre Özeti")
        txt.append("----------------")
        txt.append(f"Geometri: L1={p.L1:.6g}, L2={p.L2:.6g}, L_tip={p.L_tip:.6g} (mm)")
        txt.append(f"Ofsetler: D1={p.D1:.6g}, D2={p.D2:.6g}, H1={p.H1:.6g}, H2={p.H2:.6g}, Btot={(p.H1+p.H2):.6g} (mm)")
        txt.append(f"delta_tip = {np.rad2deg(p.delta_tip):.6g} (deg)")
        txt.append("")
        txt.append("Stroke/Zaman")
        txt.append("------------")
        txt.append(f"t_end={cfg.t_end:.6g} s, Nt={cfg.Nt}, S34_0={cfg.S34_0:.6g} mm, S34_f={cfg.S34_f:.6g} mm")
        txt.append("")
        txt.append("Sonuç Özeti")
        txt.append("-----------")
        txt.append(f"S34_end = {S[-1]:.6g} mm")
        txt.append(f"θ_tip_end = {th[-1]:.6g} deg")
        txt.append(f"θ_tip min/max = {th.min():.6g} / {th.max():.6g} deg")
        txt.append(f"ω_tip min/max = {w.min():.6g} / {w.max():.6g} deg/s")
        txt.append(f"α_tip min/max = {a.min():.6g} / {a.max():.6g} deg/s²")
        txt.append(f"Çözüm başarı oranı = {np.mean(out['ok'])*100:.3f}%")

        self.lbl_summary.setText("\n".join(txt))

    def _update_plots(self, out: dict):
        S = out["S34"]
        th = np.rad2deg(out["theta_tip"])
        w = np.rad2deg(out["dtheta_tip"])
        a = np.rad2deg(out["ddtheta_tip"])

        # θ
        self.ax_theta.clear()
        self.ax_theta.plot(S, th, linewidth=2)
        self.ax_theta.set_xlabel("S34 (mm)")
        self.ax_theta.set_ylabel("θ_tip (deg)")
        self.ax_theta.grid(True)
        self.ax_theta.set_title(f"θ_tip vs S34 | Son: {th[-1]:.2f} deg")
        self.canvas_theta.fig.tight_layout()
        self.canvas_theta.draw()

        # ω
        self.ax_omega.clear()
        self.ax_omega.plot(S, w, linewidth=2)
        self.ax_omega.set_xlabel("S34 (mm)")
        self.ax_omega.set_ylabel("ω_tip (deg/s)")
        self.ax_omega.grid(True)
        self.ax_omega.set_title("ω_tip vs S34")
        self.canvas_omega.fig.tight_layout()
        self.canvas_omega.draw()

        # α
        self.ax_alpha.clear()
        self.ax_alpha.plot(S, a, linewidth=2)
        self.ax_alpha.set_xlabel("S34 (mm)")
        self.ax_alpha.set_ylabel("α_tip (deg/s²)")
        self.ax_alpha.grid(True)
        self.ax_alpha.set_title("α_tip vs S34")
        self.canvas_alpha.fig.tight_layout()
        self.canvas_alpha.draw()

        # v components
        self.ax_v.clear()
        self.ax_v.plot(S, out["v_tan"], linewidth=2, label="v_tan")
        self.ax_v.plot(S, out["v_norm"], linewidth=1.5, linestyle="--", label="v_norm")
        self.ax_v.set_xlabel("S34 (mm)")
        self.ax_v.set_ylabel("v (mm/s)")
        self.ax_v.grid(True)
        self.ax_v.set_title("Tip Hız Bileşenleri (t-n)")
        self.ax_v.legend(loc="best")
        self.canvas_v.fig.tight_layout()
        self.canvas_v.draw()

        # a components
        self.ax_a.clear()
        self.ax_a.plot(S, out["a_tan"], linewidth=2, label="a_tan")
        self.ax_a.plot(S, out["a_norm"], linewidth=1.5, linestyle="--", label="a_norm")
        self.ax_a.set_xlabel("S34 (mm)")
        self.ax_a.set_ylabel("a (mm/s²)")
        self.ax_a.grid(True)
        self.ax_a.set_title("Tip İvme Bileşenleri (t-n)")
        self.ax_a.legend(loc="best")
        self.canvas_a.fig.tight_layout()
        self.canvas_a.draw()

    def _update_table(self, out: dict):
        S = out["S34"]
        th = np.rad2deg(out["theta_tip"])
        w = np.rad2deg(out["dtheta_tip"])
        a = np.rad2deg(out["ddtheta_tip"])
        v_t = out["v_tan"]
        a_t = out["a_tan"]
        a_n = out["a_norm"]
        ok = out["ok"]

        only_failed = self.chk_only_failed.isChecked()
        if only_failed:
            idx = np.where(~ok)[0]
        else:
            idx = np.arange(len(S))

        cols = [
            "S34(mm)",
            "theta_tip(deg)",
            "omega_tip(deg/s)",
            "alpha_tip(deg/s^2)",
            "v_tan(mm/s)",
            "a_tan(mm/s^2)",
            "a_norm(mm/s^2)",
            "ok",
        ]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(idx))

        for r, i in enumerate(idx):
            vals = [S[i], th[i], w[i], a[i], v_t[i], a_t[i], a_n[i], ok[i]]
            for c, v in enumerate(vals):
                if c == 7:
                    text = "false" if (not v) else "true"
                else:
                    text = f"{float(v):.6g}"
                item = QTableWidgetItem(text)
                if c == 7 and (not ok[i]):
                    item.setBackground(Qt.red)
                self.table.setItem(r, c, item)

        self.table.resizeColumnsToContents()


def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
