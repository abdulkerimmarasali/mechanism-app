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
    QFileDialog,
    QSlider,
    QTextBrowser,
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
    # one-file bundle
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / relative

    # one-dir bundle: next to exe
    exe_dir = Path(sys.executable).resolve().parent
    p1 = exe_dir / relative
    if p1.exists():
        return p1

    # dev: project root assumed one above src/
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
        self.resize(1280, 820)

        # last outputs
        self.out: dict | None = None

        # precomputed simulation coordinates
        self.sim_A = None
        self.sim_B = None
        self.sim_Tip = None
        self.sim_xlim = None
        self.sim_ylim = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Banner (logo + title)
        root.addLayout(self._build_banner())

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # Tabs
        self.tab_input = QWidget()
        self.tab_sim = QWidget()
        self.tab_summary = QWidget()
        self.tab_angle = QWidget()
        self.tab_w = QWidget()
        self.tab_alpha = QWidget()
        self.tab_v = QWidget()
        self.tab_a = QWidget()
        self.tab_data = QWidget()
        self.tab_eq = QWidget()

        self.tabs.addTab(self.tab_input, "Giriş")
        self.tabs.addTab(self.tab_sim, "Simülasyon")
        self.tabs.addTab(self.tab_summary, "Özet")
        self.tabs.addTab(self.tab_angle, "Bıçak Açısı")
        self.tabs.addTab(self.tab_w, "Açısal Hız")
        self.tabs.addTab(self.tab_alpha, "Açısal İvme")
        self.tabs.addTab(self.tab_v, "Bıçak Hızı")
        self.tabs.addTab(self.tab_a, "Bıçak İvmesi")
        self.tabs.addTab(self.tab_data, "Veriler")
        self.tabs.addTab(self.tab_eq, "Denklemler")

        # disable result tabs until first run (everything except giriş)
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, False)

        self._build_tab_input()
        self._build_tab_summary()
        self._build_tab_plots()
        self._build_tab_data()
        self._build_tab_sim()
        self._build_tab_equations()

    # ----------------------------
    # Theme + banner
    # ----------------------------
    def _build_banner(self):
        lay = QHBoxLayout()
        lay.setSpacing(10)

        self.logo = QLabel()
        self.logo.setFixedHeight(46)
        self.logo.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        logo_path = resource_path("assets/roketsan_logo.png")
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            pix2 = pix.scaledToHeight(40, Qt.SmoothTransformation)
            self.logo.setPixmap(pix2)
        else:
            self.logo.setText("roketsan")

        title = QLabel("Mechanism Kinematic Analyzer (Teknik Demo)")
        title.setStyleSheet("font-weight:600; font-size: 14pt; color:#005F2C;")

        lay.addWidget(self.logo, 0)
        lay.addWidget(title, 0)
        lay.addStretch(1)
        return lay

    # ----------------------------
    # Tab 1: Input + Image
    # ----------------------------
    def _build_tab_input(self):
        layout = QHBoxLayout(self.tab_input)
        layout.setSpacing(10)

        # Left: compact inputs
        left = QVBoxLayout()
        left.setSpacing(8)

        box_geo = QGroupBox("Geometri")
        form_geo = QFormLayout(box_geo)
        form_geo.setVerticalSpacing(6)
        form_geo.setHorizontalSpacing(10)

        self.le_L1 = QLineEdit("28")
        self.le_L2 = QLineEdit("30.017")
        self.le_Ltip = QLineEdit("565")
        form_geo.addRow("Kol-1 Uzunluğu (A-B) [mm]", self.le_L1)
        form_geo.addRow("Kol-2 Uzunluğu (O-B) [mm]", self.le_L2)
        form_geo.addRow("Bıçak Uzunluğu [mm]", self.le_Ltip)
        note = QLabel("Not: Bıçak uzunluğu O noktasından itibaren ölçülür.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#444; font-weight:400;")
        form_geo.addRow(note)

        box_off = QGroupBox("Ofsetler")
        form_off = QFormLayout(box_off)
        form_off.setVerticalSpacing(6)
        form_off.setHorizontalSpacing(10)

        self.le_D1 = QLineEdit("9.042")
        self.le_D2 = QLineEdit("30")
        self.le_H1 = QLineEdit("26.5")
        self.le_H2 = QLineEdit("1")
        form_off.addRow("A X-Ofset (D1) [mm]", self.le_D1)
        form_off.addRow("O X-Ofset (D2) [mm]", self.le_D2)
        form_off.addRow("H1 [mm]", self.le_H1)
        form_off.addRow("H2 [mm]", self.le_H2)

        box_motion = QGroupBox("Hareket Tanımı")
        form_m = QFormLayout(box_motion)
        form_m.setVerticalSpacing(6)
        form_m.setHorizontalSpacing(10)

        self.le_t_end = QLineEdit("0.160")
        self.le_Nt = QLineEdit("3000")
        self.le_stroke = QLineEdit("50")
        form_m.addRow("Hareket Süresi [s]", self.le_t_end)
        form_m.addRow("Zaman Adım Sayısı", self.le_Nt)
        form_m.addRow("Stroke [mm]", self.le_stroke)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Hesapla")
        self.btn_reset = QPushButton("Reset")
        self.btn_run.clicked.connect(self.on_run)
        self.btn_reset.clicked.connect(self.on_reset)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_reset)

        self.lbl_status = QLabel("Hazır.")
        self.lbl_status.setWordWrap(True)

        left_container = QWidget()
        left_container.setLayout(left)
        left_container.setMaximumWidth(440)

        left.addWidget(box_geo)
        left.addWidget(box_off)
        left.addWidget(box_motion)
        left.addLayout(btn_row)
        left.addWidget(self.lbl_status)
        left.addStretch(1)

        # Make left panel scrollable so it never overlaps
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        left_scroll.setMaximumWidth(470)

        # Right: mechanism image
        right = QVBoxLayout()
        title = QLabel("Mekanizma Görseli")
        title.setStyleSheet("font-weight:600; color:#005F2C;")
        right.addWidget(title, 0, Qt.AlignLeft)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.img_label)
        right.addWidget(scroll, 1)

        layout.addWidget(left_scroll, 0)
        layout.addLayout(right, 1)

        self._load_mech_image()

    def _load_mech_image(self):
        img_path = resource_path("assets/mechanism.png")
        if img_path.exists():
            pix = QPixmap(str(img_path))
            # scale reasonably for the right pane
            self.img_label.setPixmap(pix.scaledToWidth(920, Qt.SmoothTransformation))
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
        self.le_t_end.setText("0.160")
        self.le_Nt.setText("3000")
        self.le_stroke.setText("50")
        self.lbl_status.setText("Hazır.")
        self.tabs.setCurrentIndex(0)

    # ----------------------------
    # Tab: Summary
    # ----------------------------
    def _build_tab_summary(self):
        lay = QVBoxLayout(self.tab_summary)
        self.lbl_summary = QLabel("Henüz sonuç yok. Giriş sekmesinden Hesapla.")
        self.lbl_summary.setWordWrap(True)
        lay.addWidget(self.lbl_summary)
        lay.addStretch(1)

    # ----------------------------
    # Plot tabs
    # ----------------------------
    def _build_tab_plots(self):
        self.canvas_angle = MplCanvas()
        self.ax_angle = self.canvas_angle.fig.add_subplot(111)
        QVBoxLayout(self.tab_angle).addWidget(self.canvas_angle)

        self.canvas_w = MplCanvas()
        self.ax_w = self.canvas_w.fig.add_subplot(111)
        QVBoxLayout(self.tab_w).addWidget(self.canvas_w)

        self.canvas_alpha = MplCanvas()
        self.ax_alpha = self.canvas_alpha.fig.add_subplot(111)
        QVBoxLayout(self.tab_alpha).addWidget(self.canvas_alpha)

        self.canvas_v = MplCanvas()
        self.ax_v = self.canvas_v.fig.add_subplot(111)
        QVBoxLayout(self.tab_v).addWidget(self.canvas_v)

        self.canvas_a = MplCanvas()
        self.ax_a = self.canvas_a.fig.add_subplot(111)
        QVBoxLayout(self.tab_a).addWidget(self.canvas_a)

    # ----------------------------
    # Tab: Data + export
    # ----------------------------
    def _build_tab_data(self):
        lay = QVBoxLayout(self.tab_data)

        top = QHBoxLayout()
        self.btn_export = QPushButton("Excel'e Aktar (.xlsx)")
        self.btn_export.clicked.connect(self.export_xlsx)
        top.addWidget(self.btn_export)
        top.addStretch(1)

        self.table = QTableWidget()
        lay.addLayout(top)
        lay.addWidget(self.table, 1)

    def export_xlsx(self):
        if self.out is None:
            QMessageBox.information(self, "Bilgi", "Önce hesaplama yapın.")
            return
        try:
            from openpyxl import Workbook
            from openpyxl.utils import get_column_letter

            path, _ = QFileDialog.getSaveFileName(
                self, "Excel Kaydet", "mechanism_results.xlsx", "Excel Files (*.xlsx)"
            )
            if not path:
                return

            wb = Workbook()
            ws = wb.active
            ws.title = "Results"

            headers = self._table_headers()
            ws.append(headers)

            rows = self._table_rows()
            for r in rows:
                ws.append(r)

            for c in range(1, len(headers) + 1):
                col = get_column_letter(c)
                ws.column_dimensions[col].width = max(12, min(30, len(headers[c - 1]) + 2))

            wb.save(path)
            QMessageBox.information(self, "Tamam", "Excel dosyası kaydedildi.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Excel export başarısız: {e}")

    # ----------------------------
    # Tab: Simulation (slider controlled) - FIXED VIEW + RAIL
    # ----------------------------
    def _build_tab_sim(self):
        lay = QVBoxLayout(self.tab_sim)

        top = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._sim_update)
        self.sim_info = QLabel("Hesaplama sonrası aktif olur.")
        self.sim_info.setStyleSheet("color:#333;")
        top.addWidget(QLabel("Konum:"))
        top.addWidget(self.slider, 1)
        top.addWidget(self.sim_info, 0)

        lay.addLayout(top)

        self.canvas_sim = MplCanvas()
        self.ax_sim = self.canvas_sim.fig.add_subplot(111)
        lay.addWidget(self.canvas_sim, 1)

        self.ax_sim.set_aspect("equal", adjustable="box")
        self.ax_sim.grid(True, alpha=0.12)
        self.ax_sim.set_title("Mekanizma Simülasyonu (Slider Kontrollü)")
        self.canvas_sim.fig.tight_layout()

        # Artists (consistent colors)
        self.line_rail, = self.ax_sim.plot([], [], linewidth=2, linestyle="--", alpha=0.35)

        self.line_L2, = self.ax_sim.plot([], [], linewidth=4)   # O-B
        self.line_L1, = self.ax_sim.plot([], [], linewidth=4)   # A-B
        self.line_blade, = self.ax_sim.plot([], [], linewidth=6)  # B-Tip

        self.pt_O = self.ax_sim.scatter([], [], s=80)
        self.pt_A = self.ax_sim.scatter([], [], s=80)
        self.pt_B = self.ax_sim.scatter([], [], s=90)
        self.pt_T = self.ax_sim.scatter([], [], s=90)

        # legend-like text (small)
        self.ax_sim.text(0.01, 0.01, "O: Mavi | A: Turuncu | B: Yeşil | Tip: Kırmızı",
                         transform=self.ax_sim.transAxes, fontsize=9, alpha=0.6)

    def _sim_update(self, k: int):
        if self.out is None or self.sim_A is None:
            return
        k = int(k)

        A = self.sim_A[k]
        B = self.sim_B[k]
        T = self.sim_Tip[k]
        O = np.array([0.0, 0.0])

        # stable limits (do NOT rescale each frame)
        if self.sim_xlim is not None and self.sim_ylim is not None:
            self.ax_sim.set_xlim(*self.sim_xlim)
            self.ax_sim.set_ylim(*self.sim_ylim)

        # rail at y=0 across whole view
        if self.sim_xlim is not None:
            x0, x1 = self.sim_xlim
            self.line_rail.set_data([x0, x1], [0.0, 0.0])

        # update lines
        self.line_L2.set_data([O[0], B[0]], [O[1], B[1]])
        self.line_L1.set_data([A[0], B[0]], [A[1], B[1]])
        self.line_blade.set_data([B[0], T[0]], [B[1], T[1]])

        # points
        self.pt_O.set_offsets([O])
        self.pt_A.set_offsets([A])
        self.pt_B.set_offsets([B])
        self.pt_T.set_offsets([T])

        # point styles (colors)
        self.pt_O.set_color("#1f77b4")  # blue
        self.pt_A.set_color("#ff7f0e")  # orange
        self.pt_B.set_color("#2ca02c")  # green
        self.pt_T.set_color("#d62728")  # red

        # line colors: links gray, blade roketsan green
        self.line_L1.set_color("#555555")
        self.line_L2.set_color("#555555")
        self.line_blade.set_color("#00843D")
        self.line_rail.set_color("#888888")

        # info
        t = float(self.out["t"][k])
        s = float(self.out["S34"][k])
        ang = float(np.rad2deg(self.out["theta_tip"][k]))
        self.sim_info.setText(f"t={t:.4f}s | Stroke={s:.2f}mm | Bıçak Açısı={ang:.2f}°")

        self.canvas_sim.draw()

    # ----------------------------
    # Tab: Equations (HTML, clean layout)
    # ----------------------------
    def _build_tab_equations(self):
        lay = QVBoxLayout(self.tab_eq)

        box = QGroupBox("Denklemler (Sabit İvme Varsayımı)")
        v = QVBoxLayout(box)

        tb = QTextBrowser()
        tb.setOpenExternalLinks(False)

        html = """
        <div style="font-family:Segoe UI, Arial; font-size:12pt; line-height:1.55; padding:10px;">
          <h2 style="margin:0 0 10px 0; color:#005F2C;">Konum Denklemleri</h2>
          <div style="margin-left:10px;">
            <div><b>(1)</b> D2 − (S + D1) + L1·cos(θ1) + L2·cos(θ2) = 0</div>
            <div><b>(2)</b> −(H1 + H2) + L1·sin(θ1) + L2·sin(θ2) = 0</div>
          </div>

          <h2 style="margin:16px 0 10px 0; color:#005F2C;">Hız</h2>
          <div style="margin-left:10px;">
            <div>J(θ) · [θ̇1, θ̇2]ᵀ = [Ṡ, 0]ᵀ</div>
          </div>

          <h2 style="margin:16px 0 10px 0; color:#005F2C;">İvme (dkd formu)</h2>
          <div style="margin-left:10px;">
            <div>term_x = L1·(θ̇1²)·cos(θ1) + L2·(θ̇2²)·cos(θ2)</div>
            <div>term_y = L1·(θ̇1²)·sin(θ1) + L2·(θ̇2²)·sin(θ2)</div>
            <div>[θ̈1, θ̈2]ᵀ = J(θ)⁻¹ · [S̈ + term_x, term_y]ᵀ</div>
          </div>

          <h2 style="margin:16px 0 10px 0; color:#005F2C;">Bıçak</h2>
          <div style="margin-left:10px;">
            <div>δ_tip = arctan(H2 / D2)</div>
            <div>θ_bıçak = θ2 + δ_tip</div>
            <div>v = L_bıçak·θ̇_bıçak</div>
            <div>a_t = L_bıçak·θ̈_bıçak</div>
            <div>a_n = L_bıçak·(θ̇_bıçak²)</div>
          </div>

          <h2 style="margin:16px 0 10px 0; color:#005F2C;">Kısa Metod Özeti</h2>
          <div style="margin-left:10px;">
            <div>Her zaman adımında (θ1, θ2) konum denklemlerinden sayısal kök bulma ile çözülür.</div>
            <div>Ardından hız/ivme, Jacobian ile oluşan lineer sistemlerin çözümüyle elde edilir.</div>
          </div>
        </div>
        """
        tb.setHtml(html)

        v.addWidget(tb, 1)
        lay.addWidget(box, 1)

    # ----------------------------
    # Run simulation and update tabs
    # ----------------------------
    def on_run(self):
        try:
            self.lbl_status.setText("Hesaplanıyor...")
            self.btn_run.setEnabled(False)
            QApplication.processEvents()

            p = Params(
                L1=to_float(self.le_L1, "Kol-1 Uzunluğu"),
                L2=to_float(self.le_L2, "Kol-2 Uzunluğu"),
                L_tip=to_float(self.le_Ltip, "Bıçak Uzunluğu"),
                D1=to_float(self.le_D1, "D1"),
                D2=to_float(self.le_D2, "D2"),
                H1=to_float(self.le_H1, "H1"),
                H2=to_float(self.le_H2, "H2"),
            )

            cfg = StrokeCfg(
                t_end=to_float(self.le_t_end, "Hareket Süresi"),
                Nt=to_int(self.le_Nt, "Zaman Adım Sayısı"),
                stroke_mm=to_float(self.le_stroke, "Stroke"),
            )

            # validation
            if cfg.t_end <= 0:
                raise ValueError("Hareket Süresi > 0 olmalı.")
            if cfg.Nt < 50:
                raise ValueError("Zaman Adım Sayısı en az 50 olmalı (öneri: 1000+).")
            if cfg.stroke_mm <= 0:
                raise ValueError("Stroke > 0 olmalı.")
            if p.L1 <= 0 or p.L2 <= 0 or p.L_tip <= 0:
                raise ValueError("Uzunluklar > 0 olmalı.")

            out = simulate(p, cfg)
            self.out = out

            ok_ratio = float(np.mean(out["ok"])) * 100.0
            s_end = float(out["S34"][-1])
            ang_end = float(np.rad2deg(out["theta_tip"][-1]))

            self.lbl_status.setText(
                f"Son: Stroke={s_end:.2f} mm | Bıçak Açısı={ang_end:.2f}° | başarı={ok_ratio:.1f}%"
            )

            # enable result tabs
            for i in range(1, self.tabs.count()):
                self.tabs.setTabEnabled(i, True)

            self._update_summary(p, cfg, out)
            self._update_plots(out)
            self._update_table(out)
            self._prepare_sim(out, p)

            # go to summary
            self.tabs.setCurrentIndex(2)

            if ok_ratio < 99.0:
                QMessageBox.warning(
                    self,
                    "Uyarı",
                    "Bazı adımlarda kök bulma başarısız. 'Veriler' sekmesindeki ok sütununu kontrol edin."
                )

        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))
            self.lbl_status.setText("Hata oluştu.")
        finally:
            self.btn_run.setEnabled(True)

    def _update_summary(self, p: Params, cfg: StrokeCfg, out: dict):
        S = out["S34"]
        ang = np.rad2deg(out["theta_tip"])
        w = np.rad2deg(out["dtheta_tip"])
        alpha = np.rad2deg(out["ddtheta_tip"])

        txt = []
        txt.append("Parametre Özeti")
        txt.append("----------------")
        txt.append(f"Kol-1 (A-B) = {p.L1:.6g} mm")
        txt.append(f"Kol-2 (O-B) = {p.L2:.6g} mm")
        txt.append(f"Bıçak Uzunluğu = {p.L_tip:.6g} mm (O noktasından itibaren)")
        txt.append(f"D1 = {p.D1:.6g} mm, D2 = {p.D2:.6g} mm")
        txt.append(f"H1 = {p.H1:.6g} mm, H2 = {p.H2:.6g} mm, (H1+H2) = {(p.H1+p.H2):.6g} mm")
        txt.append("")
        txt.append("Hareket Tanımı")
        txt.append("--------------")
        txt.append(f"Hareket Süresi = {cfg.t_end:.6g} s")
        txt.append(f"Zaman Adım Sayısı = {cfg.Nt}")
        txt.append(f"Stroke = {cfg.stroke_mm:.6g} mm")
        txt.append("")
        txt.append("Sonuç Özeti")
        txt.append("-----------")
        txt.append(f"Stroke (son) = {S[-1]:.6g} mm")
        txt.append(f"Bıçak Açısı (son) = {ang[-1]:.6g} °")
        txt.append(f"Bıçak Açısı min/max = {ang.min():.6g} / {ang.max():.6g} °")
        txt.append(f"Açısal Hız min/max = {w.min():.6g} / {w.max():.6g} °/s")
        txt.append(f"Açısal İvme min/max = {alpha.min():.6g} / {alpha.max():.6g} °/s²")
        txt.append(f"Çözüm başarı oranı = {np.mean(out['ok'])*100:.3f} %")

        self.lbl_summary.setText("\n".join(txt))

    def _update_plots(self, out: dict):
        S = out["S34"]
        ang = np.rad2deg(out["theta_tip"])
        w = np.rad2deg(out["dtheta_tip"])
        alpha = np.rad2deg(out["ddtheta_tip"])

        self.ax_angle.clear()
        self.ax_angle.plot(S, ang, linewidth=2)
        self.ax_angle.set_xlabel("Stroke (mm)")
        self.ax_angle.set_ylabel("Bıçak Açısı (°)")
        self.ax_angle.grid(True, alpha=0.20)
        self.ax_angle.set_title(f"Bıçak Açısı vs Stroke | Son: {ang[-1]:.2f}°")
        self.canvas_angle.fig.tight_layout()
        self.canvas_angle.draw()

        self.ax_w.clear()
        self.ax_w.plot(S, w, linewidth=2)
        self.ax_w.set_xlabel("Stroke (mm)")
        self.ax_w.set_ylabel("Açısal Hız (°/s)")
        self.ax_w.grid(True, alpha=0.20)
        self.ax_w.set_title("Açısal Hız vs Stroke")
        self.canvas_w.fig.tight_layout()
        self.canvas_w.draw()

        self.ax_alpha.clear()
        self.ax_alpha.plot(S, alpha, linewidth=2)
        self.ax_alpha.set_xlabel("Stroke (mm)")
        self.ax_alpha.set_ylabel("Açısal İvme (°/s²)")
        self.ax_alpha.grid(True, alpha=0.20)
        self.ax_alpha.set_title("Açısal İvme vs Stroke")
        self.canvas_alpha.fig.tight_layout()
        self.canvas_alpha.draw()

        self.ax_v.clear()
        self.ax_v.plot(S, out["v_tan"], linewidth=2)
        self.ax_v.set_xlabel("Stroke (mm)")
        self.ax_v.set_ylabel("Bıçak Hızı (mm/s)")
        self.ax_v.grid(True, alpha=0.20)
        self.ax_v.set_title("Bıçak Hızı vs Stroke")
        self.canvas_v.fig.tight_layout()
        self.canvas_v.draw()

        self.ax_a.clear()
        self.ax_a.plot(S, out["a_tan"], linewidth=2, label="Bıçak İvmesi")
        self.ax_a.plot(S, out["a_norm"], linewidth=1.5, linestyle="--", label="Merkezcil İvme")
        self.ax_a.set_xlabel("Stroke (mm)")
        self.ax_a.set_ylabel("İvme (mm/s²)")
        self.ax_a.grid(True, alpha=0.20)
        self.ax_a.set_title("Bıçak İvmesi vs Stroke")
        self.ax_a.legend(loc="best")
        self.canvas_a.fig.tight_layout()
        self.canvas_a.draw()

    def _table_headers(self):
        return [
            "Stroke (mm)",
            "Bıçak Açısı (deg)",
            "Açısal Hız (deg/s)",
            "Açısal İvme (deg/s^2)",
            "Bıçak Hızı (mm/s)",
            "Bıçak İvmesi (mm/s^2)",
            "Merkezcil İvme (mm/s^2)",
            "ok",
        ]

    def _table_rows(self):
        out = self.out
        S = out["S34"]
        ang = np.rad2deg(out["theta_tip"])
        w = np.rad2deg(out["dtheta_tip"])
        alpha = np.rad2deg(out["ddtheta_tip"])
        v_t = out["v_tan"]
        a_t = out["a_tan"]
        a_n = out["a_norm"]
        ok = out["ok"]

        rows = []
        for i in range(len(S)):
            rows.append([
                float(S[i]),
                float(ang[i]),
                float(w[i]),
                float(alpha[i]),
                float(v_t[i]),
                float(a_t[i]),
                float(a_n[i]),
                bool(ok[i]),
            ])
        return rows

    def _update_table(self, out: dict):
        headers = self._table_headers()
        rows = self._table_rows()

        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(rows))

        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                if c == 7:
                    text = "true" if v else "false"
                else:
                    text = f"{v:.6g}"
                item = QTableWidgetItem(text)
                if c == 7 and (not v):
                    item.setBackground(Qt.red)
                self.table.setItem(r, c, item)

        self.table.resizeColumnsToContents()

    def _prepare_sim(self, out: dict, p: Params):
        # Precompute A, B, Tip for all k for fast slider updates.
        # A is sliding point on x-axis: x = S34 + D1, y = 0
        S = out["S34"]
        th2 = out["theta2"]
        theta_tip = out["theta_tip"]
        Btot = p.H1 + p.H2

        A = np.column_stack([S + p.D1, np.zeros_like(S)])

        # B computed from theta2
        Bx = p.D2 + p.L2 * np.cos(th2)
        By = Btot - p.L2 * np.sin(th2)
        B = np.column_stack([Bx, By])

        # Tip
        Tx = Bx + p.L_tip * np.cos(theta_tip)
        Ty = By + p.L_tip * np.sin(theta_tip)
        Tip = np.column_stack([Tx, Ty])

        self.sim_A = A
        self.sim_B = B
        self.sim_Tip = Tip

        # Fixed limits over whole motion (no resizing jitter)
        all_pts = np.vstack([A, B, Tip, np.zeros((len(S), 2))])  # include O=(0,0)
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        pad = 0.10 * max(1.0, float((xmax - xmin) + (ymax - ymin)))

        self.sim_xlim = (xmin - pad, xmax + pad)
        self.sim_ylim = (ymin - pad, ymax + pad)
        self.ax_sim.set_xlim(*self.sim_xlim)
        self.ax_sim.set_ylim(*self.sim_ylim)

        # slider range
        self.slider.setMaximum(len(S) - 1)
        self.slider.setValue(0)
        self._sim_update(0)

    # ----------------------------
    # Apply Roketsan Theme (less pure white)
    # ----------------------------
    def apply_theme(self):
        self.setStyleSheet("""
        QWidget {
            background-color: #F5F7F6;
            color: #1A1A1A;
            font-size: 11pt;
        }
        QTabWidget::pane {
            background: #FFFFFF;
            border: 1px solid #E0E0E0;
        }
        QTabBar::tab {
            background: #005F2C;
            color: white;
            padding: 8px 12px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #00843D;
        }
        QPushButton {
            background-color: #00843D;
            color: white;
            padding: 6px 10px;
            border-radius: 4px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #005F2C;
        }
        QGroupBox {
            background-color: #FFFFFF;
            font-weight: 600;
            border: 1px solid #E0E0E0;
            margin-top: 8px;
            border-radius: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
            color: #005F2C;
        }
        QLineEdit, QTableWidget, QTextBrowser {
            background-color: #FFFFFF;
        }
        """)


def main():
    app = QApplication(sys.argv)
    w = App()
    w.apply_theme()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
