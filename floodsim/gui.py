"""หน้าจอโปรแกรม (Tkinter + Matplotlib): แม่น้ำ 1D + ช่วงลำน้ำ 3 มิติสำหรับทดสอบเรือดันน้ำ"""
from __future__ import annotations

import csv
import math
import os
import queue
import threading
import time
import tkinter as tk
import tkinter.font  # noqa: F401
import traceback
from copy import deepcopy
from dataclasses import fields, replace
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from . import hydraulics1d as h1  # noqa: E402
from .backend import get_backend  # noqa: E402
from .boat_tests import BOAT_TESTS, run_test  # noqa: E402
from .experiments import EXPERIMENTS, guide_text  # noqa: E402
from .geometry import RiverParams, Section, build_imported, generate_river  # noqa: E402
from .io_formats import SUPPORTED, export_kml, read_river_file  # noqa: E402
from .reach3d import (BOAT_TYPES, Fleet, ReachConfig, Run3D, auto_dx, bed_level, boat_type,  # noqa: E402
                      build_reach, summary_text)

plt.rcParams["font.family"] = ["Tahoma", "Leelawadee UI", "Segoe UI", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 8.5

PALETTE = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]
fmt_time = h1.fmt_time

RIVER_FIELDS = [("D_km", "ระยะทางเส้นตรง (กม.)", 0.2, 500, 0.5),
                ("sigma", "ความคดเคี้ยว σ (≥1)", 1.0, 3.0, 0.05),
                ("Sv_m_per_km", "ความชันหุบเขา (ม./กม.)", 0.01, 100, 0.05),
                ("wavelength_B", "ความยาวคลื่นโค้ง (×ความกว้าง)", 4, 40, 1),
                ("irregularity", "ความไม่สม่ำเสมอของโค้ง (0–1)", 0, 1, 0.05),
                ("seed", "รูปแบบสุ่ม (seed)", 0, 9999, 1),
                ("z_down", "ระดับท้องน้ำปลายน้ำ (ม.)", -50, 3000, 1)]
SECTION_FIELDS = [("B", "ความกว้างท้องน้ำ B (ม.)", 1, 2000, 1),
                  ("Hb", "ความสูงตลิ่ง Hb (ม.)", 0.2, 40, 0.1),
                  ("z", "ความลาดตลิ่ง z (นอน:ตั้ง)", 0, 10, 0.1),
                  ("Bf", "ที่ราบน้ำท่วมแต่ละฝั่ง (ม.)", 0, 20000, 10),
                  ("n", "แมนนิ่ง n ร่องน้ำ", 0.010, 0.2, 0.001),
                  ("nf", "แมนนิ่ง n ที่ราบ", 0.010, 0.3, 0.005)]
REACH_FIELDS = [("length", "ความยาวช่วงจำลอง (ม.)", 30, 2000, 10),
                ("dx", "ขนาดเซลล์ (ม., 0=อัตโนมัติ)", 0, 5, 0.05),
                ("max_cells", "จำนวนเซลล์สูงสุด", 1e5, 2e7, 2.5e5),
                ("fp_max", "ที่ราบน้ำท่วมในโดเมน (ม./ฝั่ง)", 0, 500, 5),
                ("spacing_km", "ระยะห่างกลุ่มเรือ (กม.) สำหรับประมาณ ΔQ", 0.1, 100, 0.5),
                ("tau_crit", "แรงเฉือนวิกฤตท้องน้ำ (N/ม²)", 0.1, 50, 0.5)]
BOAT_FIELDS = [("hull_L", "ความยาวเรือ (ม.)", 1, 80, 0.5), ("hull_B", "ความกว้างเรือ (ม.)", 0.5, 20, 0.1),
               ("draft", "ระยะกินน้ำ (ม.)", 0.1, 6, 0.05), ("n_props", "จำนวนใบพัด/หัวฉีดต่อลำ", 1, 12, 1),
               ("prop_D", "เส้นผ่านศูนย์กลางใบพัด (ม.)", 0.1, 4, 0.05),
               ("prop_depth", "ความลึกใบพัด (ม.)", 0.1, 10, 0.05),
               ("tilt_deg", "มุมกดใบพัด (องศา)", -10, 45, 1),
               ("prop_x", "ใบพัดหลังท้ายเรือ (ม.)", -10, 10, 0.1),
               ("prop_spacing", "ระยะห่างใบพัด (ม.)", 0, 10, 0.1),
               ("power_kW", "กำลังเครื่อง/ใบพัด (kW)", 1, 5000, 5),
               ("merit", "ประสิทธิภาพใบพัด (0–1)", 0.2, 0.9, 0.05),
               ("thrust_N", "แรงขับ/ใบพัด (N, 0=คำนวณ)", 0, 1e6, 500),
               ("jet_bar", "หัวฉีด: แรงดัน (บาร์, 0=ใบพัด)", 0, 500, 1),
               ("spread_deg", "หัวฉีด: มุมฉีดกระจาย ± (องศา)", 0, 80, 5)]
FLEET_FIELDS = [("rows", "จำนวนแถว (ตามลำน้ำ)", 1, 10, 1), ("per_row", "จำนวนลำต่อแถว", 1, 12, 1),
                ("row_spacing", "ระยะห่างแถว (ม.)", 5, 500, 5), ("x_first", "แถวแรกห่างทางเข้า (ม.)", 5, 1000, 5)]
INT_KEYS = {"seed", "n_props", "rows", "per_row", "max_cells"}


class App(tk.Tk):
    def __init__(self, backend_pref="auto"):
        super().__init__()
        self.title("Flood Sim — แบบจำลองแม่น้ำและการไหล 3 มิติสำหรับทดสอบเรือดันน้ำ (CUDA)")
        self.geometry("1560x940")
        self.minsize(1100, 700)
        for f in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tk.font.nametofont(f).configure(family="Tahoma", size=9)
            except Exception:  # noqa: BLE001
                pass
        # LBM 3D ใช้ CuPy (GPU) หรือ Numba (CPU)
        self.be, notes = get_backend("numpy" if backend_pref in ("torch", "numpy") else backend_pref)
        if self.be.name == "torch":
            self.be, n2 = get_backend("numpy")
            notes += n2
        self.backend_notes = notes
        self.trace = None
        self.reverse = tk.BooleanVar(value=False)
        self.use_elev = tk.BooleanVar(value=True)
        self.saved = []
        self.active_exp = None
        self.river = None
        self.run = None
        self.running = False
        self.results = []
        self.worker = None
        self.cancel_flag = False
        self.q = queue.Queue()
        self._rebuild_job = None
        self.boat_edits = {}
        self._frame = 0
        self._loading = False

        self._build_vars()
        self._build_layout()
        self.load_experiment(2, quiet=True)
        self._load_boat_fields()
        draw_results_fig(self.figR, [])
        self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================ variables
    def _build_vars(self):
        rp, sec, rc = RiverParams(), Section(), ReachConfig()
        mk = lambda k, v: (tk.IntVar if k in INT_KEYS else tk.DoubleVar)(value=v)  # noqa: E731
        self.rvars = {k: mk(k, getattr(rp, k)) for k, *_ in RIVER_FIELDS}
        self.svars = {k: mk(k, getattr(sec, k)) for k, *_ in SECTION_FIELDS}
        self.cvars = {k: mk(k, getattr(rc, k)) for k, *_ in REACH_FIELDS}
        self.bvars = {k: mk(k, 0) for k, *_ in BOAT_FIELDS}
        fl = Fleet(boat_type("longtail"))
        self.fvars = {k: mk(k, getattr(fl, k)) for k, *_ in FLEET_FIELDS}
        self.boat_key = tk.StringVar(value=BOAT_TYPES[0].name)
        self.has_boats = tk.BooleanVar(value=True)
        self.wall_model = tk.BooleanVar(value=True)
        self.levels_var = tk.StringVar(value="1, 2, 3, 4, 5")
        self.sel_level = tk.StringVar(value="3")
        self.n_particles = tk.IntVar(value=3000)
        self.show_mean = tk.BooleanVar(value=False)
        self.plan_depth = tk.DoubleVar(value=0.5)
        self.long_y = tk.DoubleVar(value=0.0)
        self.cross_x = tk.DoubleVar(value=80.0)
        for v in list(self.rvars.values()) + list(self.svars.values()):
            v.trace_add("write", lambda *a: self._schedule_rebuild())
        self.levels_var.trace_add("write", lambda *a: self._schedule_rebuild())
        for v in list(self.cvars.values()) + list(self.bvars.values()) + list(self.fvars.values()):
            v.trace_add("write", lambda *a: self._on_reach_change())

    # ================================================================ layout
    def _build_layout(self):
        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        pw.add(left, weight=0)
        right = ttk.Frame(pw)
        pw.add(right, weight=1)
        self.ui_scale = self.winfo_fpixels("1i") / 72.0
        cv = tk.Canvas(left, width=int(340 * self.ui_scale), highlightthickness=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=cv.yview)
        self.ctrl = ttk.Frame(cv)
        self.ctrl.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        win_id = cv.create_window((0, 0), window=self.ctrl, anchor="nw")
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win_id, width=e.width))
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(-int(e.delta / 120), "units")
                    if str(e.widget).startswith(str(left)) else None)
        self._build_controls(self.ctrl)

        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        self.status = tk.StringVar(value="พร้อม")
        ttk.Label(right, textvariable=self.status, anchor="w", relief="sunken").pack(fill="x")
        self._tab_3d_slices()
        self._tab_3d_view()
        self._tab_compare()
        self._tab_1d()
        self._tab_geom()
        self._tab_guide()
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._refresh_visible())

    def _section(self, parent, title):
        f = ttk.LabelFrame(parent, text=title, padding=6)
        f.pack(fill="x", padx=6, pady=4)
        f.columnconfigure(0, weight=1)
        return f

    def _num(self, parent, row, label, var, lo, hi, inc, width=8):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        sp = ttk.Spinbox(parent, from_=lo, to=hi, increment=inc, textvariable=var, width=width)
        sp.grid(row=row, column=1, sticky="e", pady=1)
        return sp

    def _wrap(self):
        return int(300 * self.ui_scale)

    def _build_controls(self, c):
        dev = f"{'GPU' if self.be.is_gpu else 'CPU'}: {self.be.device_name}  [{self.be.name}]"
        f = self._section(c, "อุปกรณ์คำนวณ")
        ttk.Label(f, text=dev, foreground="#0a7a2f" if self.be.is_gpu else "#a05a00",
                  wraplength=self._wrap()).grid(row=0, column=0, columnspan=2, sticky="w")
        if not self.be.is_gpu:
            ttk.Label(f, text="ไม่พบ CUDA — แบบจำลอง 3D บน CPU ช้ามาก ควรลดจำนวนเซลล์", foreground="#a05a00",
                      wraplength=self._wrap()).grid(row=1, column=0, columnspan=2, sticky="w")

        f = self._section(c, "เรือดันน้ำ")
        ttk.Checkbutton(f, text="มีเรือในช่วงจำลอง", variable=self.has_boats,
                        command=self._on_reach_change).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(f, text="ชนิดเรือ").grid(row=1, column=0, sticky="w")
        cb = ttk.Combobox(f, textvariable=self.boat_key, state="readonly", width=24,
                          values=[b.name for b in BOAT_TYPES])
        cb.grid(row=2, column=0, columnspan=2, sticky="ew")
        cb.bind("<<ComboboxSelected>>", lambda e: self._load_boat_fields())
        self.boat_note = tk.StringVar()
        ttk.Label(f, textvariable=self.boat_note, foreground="#555", wraplength=self._wrap()).grid(
            row=3, column=0, columnspan=2, sticky="w")
        for r, (k, lab, lo, hi, inc) in enumerate(BOAT_FIELDS):
            self._num(f, 4 + r, lab, self.bvars[k], lo, hi, inc)
        self.thrust_lbl = tk.StringVar()
        ttk.Label(f, textvariable=self.thrust_lbl, foreground="#225", wraplength=self._wrap()).grid(
            row=30, column=0, columnspan=2, sticky="w")
        ttk.Separator(f).grid(row=31, column=0, columnspan=2, sticky="ew", pady=4)
        for r, (k, lab, lo, hi, inc) in enumerate(FLEET_FIELDS):
            self._num(f, 32 + r, lab, self.fvars[k], lo, hi, inc)

        f = self._section(c, "ช่วงลำน้ำจำลอง 3 มิติ")
        ttk.Label(f, text="ระดับน้ำ (ความลึก ม.)").grid(row=0, column=0, sticky="w")
        self.level_cb = ttk.Combobox(f, textvariable=self.sel_level, width=8)
        self.level_cb.grid(row=0, column=1, sticky="e")
        self.level_cb.bind("<<ComboboxSelected>>", lambda e: self._on_reach_change())
        self.level_cb.bind("<Return>", lambda e: self._on_reach_change())
        for r, (k, lab, lo, hi, inc) in enumerate(REACH_FIELDS):
            self._num(f, 1 + r, lab, self.cvars[k], lo, hi, inc, width=9)
        ttk.Checkbutton(f, text="wall model (แรงเสียดทานท้องน้ำตามค่า n ของแมนนิ่ง)", variable=self.wall_model,
                        command=self._on_reach_change).grid(row=1 + len(REACH_FIELDS), column=0, columnspan=2, sticky="w")
        self.reach_info = tk.StringVar()
        ttk.Label(f, textvariable=self.reach_info, foreground="#225", wraplength=self._wrap(), justify="left").grid(
            row=20, column=0, columnspan=2, sticky="w", pady=(4, 0))

        f = self._section(c, "จำลอง")
        self._num(f, 0, "จำนวนอนุภาคติดตามการไหล", self.n_particles, 0, 20000, 500)
        bf = ttk.Frame(f)
        bf.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.run_btn = ttk.Button(bf, text="▶ เริ่ม", command=self.toggle_run)
        self.run_btn.pack(side="left", expand=True, fill="x")
        ttk.Button(bf, text="⟲ รีเซ็ต", command=self.reset_run).pack(side="left", expand=True, fill="x")
        self.cmp_btn = ttk.Button(f, text="เปรียบเทียบ: ไม่มีเรือ vs เรือชุดนี้", command=self.run_compare)
        self.cmp_btn.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        self.prog = ttk.Progressbar(f, maximum=1.0)
        self.prog.grid(row=3, column=0, sticky="ew")
        ttk.Button(f, text="ยกเลิก", width=7, command=self._cancel).grid(row=3, column=1, sticky="e")

        f = self._section(c, "การทดสอบเรือแนะนำ (รันอัตโนมัติ)")
        ttk.Style(self).configure("Left.TButton", anchor="w")
        self.test_btns = []
        for i, t in enumerate(BOAT_TESTS):
            b = ttk.Button(f, text=t["title"], style="Left.TButton", command=lambda i=i: self.run_boat_test(i))
            b.grid(row=i, column=0, columnspan=2, sticky="ew", pady=1)
            self.test_btns.append(b)
        ttk.Label(f, text="ใช้หน้าตัด ระดับน้ำ และค่าเรือที่แก้ไขไว้ด้านบน — แต่ละกรณีใช้เวลาประมาณ 1–5 นาทีบน GPU",
                  foreground="#666", wraplength=self._wrap()).grid(row=10, column=0, columnspan=2, sticky="w")

        f = self._section(c, "แม่น้ำ: การทดลองรูปทรง (1D)")
        for i, e in enumerate(EXPERIMENTS):
            ttk.Button(f, text=e["title"], style="Left.TButton",
                       command=lambda i=i: self.load_experiment(i)).grid(row=i, column=0, columnspan=2,
                                                                         sticky="ew", pady=1)
        f = self._section(c, "รูปทรงแม่น้ำ")
        self.river_spins = []
        for r, (k, lab, lo, hi, inc) in enumerate(RIVER_FIELDS):
            self.river_spins.append(self._num(f, r, lab, self.rvars[k], lo, hi, inc))
        self.derived = tk.StringVar()
        ttk.Label(f, textvariable=self.derived, foreground="#225", wraplength=self._wrap(), justify="left").grid(
            row=20, column=0, columnspan=2, sticky="w", pady=(4, 0))
        f = self._section(c, "หน้าตัดลำน้ำ (ใช้ทั้ง 1D และ 3D)")
        for r, (k, lab, lo, hi, inc) in enumerate(SECTION_FIELDS):
            self._num(f, r, lab, self.svars[k], lo, hi, inc)
        f = self._section(c, "นำเข้าไฟล์แม่น้ำ (KML / KMZ / GPX / GeoJSON / CSV)")
        ttk.Button(f, text="เลือกไฟล์…", command=self.import_file).grid(row=0, column=0, sticky="ew")
        ttk.Button(f, text="ใช้แม่น้ำจำลอง", command=self.clear_import).grid(row=0, column=1, sticky="ew")
        ttk.Checkbutton(f, text="ใช้ค่าความสูงจากไฟล์ (ถ้ามี)", variable=self.use_elev,
                        command=self._schedule_rebuild).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(f, text="กลับทิศการไหล (ใช้เมื่อไม่มีค่าระดับ)", variable=self.reverse,
                        command=self._schedule_rebuild).grid(row=2, column=0, columnspan=2, sticky="w")
        self.import_info = tk.StringVar(value="ยังไม่ได้นำเข้าไฟล์ — ใช้แม่น้ำจำลอง")
        ttk.Label(f, textvariable=self.import_info, wraplength=self._wrap(), justify="left", foreground="#444").grid(
            row=3, column=0, columnspan=2, sticky="w")
        f = self._section(c, "ระดับน้ำที่เปรียบเทียบ (ความลึก ม.)")
        ttk.Entry(f, textvariable=self.levels_var).grid(row=0, column=0, columnspan=2, sticky="ew")

        f = self._section(c, "ส่งออก")
        ttk.Button(f, text="บันทึกรูปทุกแท็บ (PNG)", command=self.save_figs).grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(f, text="ผลเปรียบเทียบ CSV", command=self.export_csv).grid(row=1, column=0, sticky="ew")
        ttk.Button(f, text="สนามความเร็ว 3D (.vtk)", command=self.export_vtk).grid(row=1, column=1, sticky="ew")
        ttk.Button(f, text="เส้นแม่น้ำ KML", command=self.export_kml).grid(row=2, column=0, columnspan=2, sticky="ew")

    def _fig(self, parent, figsize=(5, 3.5), toolbar=True):
        fig = Figure(figsize=figsize, dpi=100, layout="constrained")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        tb = None
        if toolbar:
            tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
            tb.update()
            tb.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, canvas, tb

    def _tab_3d_slices(self):
        fr = ttk.Frame(self.nb)
        self.nb.add(fr, text="จำลอง 3D (GPU)")
        self.tab_slices = fr
        bar = ttk.Frame(fr, padding=3)
        bar.pack(fill="x")
        ttk.Checkbutton(bar, text="แสดงค่าเฉลี่ยตามเวลา", variable=self.show_mean,
                        command=self._refresh_visible).pack(side="left")
        for lab, var, lo, hi, inc in (("  ระนาบแนวนอนลึก (ม.)", self.plan_depth, 0, 50, 0.1),
                                      ("  แนวตัดตามยาวที่ y (ม.)", self.long_y, -500, 500, 0.5),
                                      ("  หน้าตัดขวางที่ x (ม.)", self.cross_x, 0, 5000, 5)):
            ttk.Label(bar, text=lab).pack(side="left")
            sp = ttk.Spinbox(bar, from_=lo, to=hi, increment=inc, textvariable=var, width=6,
                             command=self._refresh_visible)
            sp.pack(side="left")
            sp.bind("<Return>", lambda e: self._refresh_visible())
        ttk.Label(bar, text="  (คลิกบนระนาบแนวนอนเพื่อเลือกแนวตัด)", foreground="#666").pack(side="left")
        self.figS, self.cvS, self.tbS = self._fig(fr)
        gs = self.figS.add_gridspec(3, 2, height_ratios=[1.25, 1, 1])
        self.axP = self.figS.add_subplot(gs[0, :])
        self.axL = self.figS.add_subplot(gs[1, :])
        self.axC = self.figS.add_subplot(gs[2, 0])
        self.axE = self.figS.add_subplot(gs[2, 1])
        self.axE2 = self.axE.twinx()
        self.cbS = None
        self.axP.text(0.5, 0.5, "ตั้งค่าเรือแล้วกด ▶ เริ่ม เพื่อจำลองการไหล 3 มิติ", ha="center", va="center",
                      transform=self.axP.transAxes, fontsize=12, color="#555")
        self.cvS.mpl_connect("button_press_event", self._on_slice_click)

    def _tab_3d_view(self):
        fr = ttk.Frame(self.nb)
        self.nb.add(fr, text="มุมมอง 3D")
        self.tab_view = fr
        ttk.Label(fr, text="ลากเมาส์เพื่อหมุน • จุดสี = อนุภาคที่ไหลไปกับน้ำ (สีตามความเร็ว) • กล่องเทา = ตัวเรือ • วงแดง = ใบพัด",
                  foreground="#666", padding=3).pack(anchor="w")
        self.fig3, self.cv3, _ = self._fig(fr, toolbar=False)
        self.ax3 = self.fig3.add_subplot(projection="3d")

    def _tab_compare(self):
        fr = ttk.Frame(self.nb)
        self.nb.add(fr, text="ผลเปรียบเทียบเรือ")
        self.tab_cmp = fr
        top = ttk.Frame(fr)
        top.pack(fill="x")
        cols = ("label", "h", "T", "P", "deta", "reach", "us", "umax", "tau", "dq", "dqn")
        heads = ("กรณี", "ระดับน้ำ (ม.)", "แรงขับ (kN)", "กำลัง (kW)", "Δระดับน้ำ (ซม.)",
                 "ระยะสายน้ำ (ม.)", "ผิวน้ำเร็วขึ้น (ม./วิ)", "V สูงสุด", "τ ท้องน้ำสูงสุด",
                 "ΔQ จากแรงขับ (%)", "ΔQ สุทธิ (%)")
        self.tree = ttk.Treeview(top, columns=cols, show="headings", height=6)
        for c_, h_ in zip(cols, heads):
            self.tree.heading(c_, text=h_)
            self.tree.column(c_, width=70, minwidth=40, stretch=True, anchor="center")
        self.tree.column("label", width=160, anchor="w")
        self.tree.pack(side="left", fill="x", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._show_result_text())
        bf = ttk.Frame(top)
        bf.pack(side="left", fill="y", padx=4)
        ttk.Button(bf, text="ล้างผล", command=self.clear_results).pack(fill="x")
        pw = ttk.PanedWindow(fr, orient="vertical")
        pw.pack(fill="both", expand=True)
        f2 = ttk.Frame(pw)
        pw.add(f2, weight=4)
        self.figR, self.cvR, _ = self._fig(f2)
        f3 = ttk.Frame(pw)
        pw.add(f3, weight=1)
        self.res_text = tk.Text(f3, height=6, wrap="word", font=("Tahoma", 9), background="#fbfaf4")
        self.res_text.pack(fill="both", expand=True)

    def _tab_1d(self):
        fr = ttk.Frame(self.nb)
        self.nb.add(fr, text="ระดับน้ำ (1D)")
        self.tab_1d = fr
        top = ttk.Frame(fr)
        top.pack(fill="x")
        cols = ("h", "Q", "V", "Vm", "Fr", "tau", "tt", "T", "state")
        heads = ("h (ม.)", "Q (ม³/วิ)", "V เฉลี่ย", "V ร่องน้ำ", "Fr สูงสุด", "τ (N/ม²)", "เวลาไหลผ่าน",
                 "กว้างผิวน้ำ (ม.)", "สภาพการไหล")
        self.tree1 = ttk.Treeview(top, columns=cols, show="headings", height=7)
        for c_, h_ in zip(cols, heads):
            self.tree1.heading(c_, text=h_)
            self.tree1.column(c_, width=60, minwidth=40, stretch=True, anchor="center")
        self.tree1.column("state", width=140)
        self.tree1.pack(side="left", fill="x", expand=True)
        bf = ttk.Frame(top)
        bf.pack(side="left", fill="y", padx=4)
        ttk.Button(bf, text="บันทึกผลนี้ไว้เปรียบเทียบ", command=self.save_scenario).pack(fill="x", pady=1)
        ttk.Button(bf, text="รันชุดเปรียบเทียบของการทดลอง", command=self.run_sweep).pack(fill="x", pady=1)
        ttk.Button(bf, text="ล้างผลที่บันทึก", command=self.clear_saved).pack(fill="x", pady=1)
        mid = ttk.PanedWindow(fr, orient="vertical")
        mid.pack(fill="both", expand=True)
        f2 = ttk.Frame(mid)
        mid.add(f2, weight=4)
        self.fig1d, self.cv1d, _ = self._fig(f2, toolbar=False)
        self.ax1d = self.fig1d.subplots(2, 3).ravel()
        f3 = ttk.Frame(mid)
        mid.add(f3, weight=1)
        self.insight = tk.Text(f3, height=7, wrap="word", font=("Tahoma", 9), background="#fbfaf4")
        self.insight.pack(fill="both", expand=True)

    def _tab_geom(self):
        fr = ttk.Frame(self.nb)
        self.nb.add(fr, text="รูปทรงแม่น้ำ")
        self.tab_geo = fr
        self.figgeo, self.cvgeo, _ = self._fig(fr)
        gs = self.figgeo.add_gridspec(2, 2, height_ratios=[2, 1])
        self.axg_plan = self.figgeo.add_subplot(gs[0, :])
        self.axg_prof = self.figgeo.add_subplot(gs[1, 0])
        self.axg_sig = self.figgeo.add_subplot(gs[1, 1])
        self.axg_sig2 = self.axg_sig.twinx()

    def _tab_guide(self):
        fr = ttk.Frame(self.nb)
        self.nb.add(fr, text="คู่มือ")
        t = tk.Text(fr, wrap="word", font=("Tahoma", 10), padx=12, pady=8)
        t.insert("1.0", guide_3d() + "\n\n" + guide_text())
        t.configure(state="disabled")
        t.pack(fill="both", expand=True)

    # ================================================================ parameters
    def _get(self, v, k):
        return int(float(v.get())) if k in INT_KEYS else float(v.get())

    def read_params(self):
        rp = RiverParams(**{k: self._get(v, k) for k, v in self.rvars.items()})
        sec = Section(**{k: self._get(v, k) for k, v in self.svars.items()})
        if min(sec.B, sec.Hb, sec.n, sec.nf, rp.D_km, rp.Sv_m_per_km) <= 0:
            raise ValueError("ค่าพารามิเตอร์ต้องมากกว่า 0")
        rp.sigma = max(1.0, min(rp.sigma, 3.0))
        return rp, sec

    def read_levels(self):
        vals = sorted({float(t) for t in self.levels_var.get().replace(";", ",").split(",") if t.strip()})
        vals = [v for v in vals if v > 0]
        if not vals:
            raise ValueError("ใส่ระดับน้ำอย่างน้อย 1 ค่า")
        return vals

    def read_reach_cfg(self):
        return ReachConfig(**{k: self._get(v, k) for k, v in self.cvars.items()},
                           wall_model=bool(self.wall_model.get()))

    def _boat_key(self):
        name = self.boat_key.get()
        return next(b.key for b in BOAT_TYPES if b.name == name)

    def _boat_from_vars(self):
        return replace(boat_type(self._boat_key()), **{k: self._get(v, k) for k, v in self.bvars.items()})

    def read_fleet(self):
        fl = {k: self._get(v, k) for k, v in self.fvars.items()}
        return Fleet(self._boat_from_vars(), enabled=bool(self.has_boats.get()), **fl)

    def _load_boat_fields(self):
        key = self._boat_key()
        b = boat_type(key)
        edits = self.boat_edits.get(key, {})
        self._loading = True
        for k, v in self.bvars.items():
            v.set(edits.get(k, getattr(b, k)))
        self._loading = False
        self.boat_note.set(b.note)
        self._on_reach_change()

    def _sel_h(self):
        try:
            return float(self.sel_level.get())
        except ValueError:
            return self.levels[len(self.levels) // 2]

    def _schedule_rebuild(self):
        if self._rebuild_job:
            self.after_cancel(self._rebuild_job)
        self._rebuild_job = self.after(450, self.rebuild)

    def _on_reach_change(self):
        """ค่าเรือ/ช่วงจำลองเปลี่ยน: เก็บค่าที่แก้ไข ล้างการจำลองปัจจุบัน และแสดงข้อมูลสรุป"""
        if self._loading or self.river is None:
            return
        try:
            b = self._boat_from_vars()
            if b.outlet_D() <= 0 or b.hull_L <= 0:
                return
            self.boat_edits[b.key] = {k: self._get(v, k) for k, v in self.bvars.items()}
            fl = self.read_fleet()
            cfg = self.read_reach_cfg()
            h = self._sel_h()
        except (tk.TclError, ValueError, StopIteration):
            return
        what = f"หัวฉีด Ø{b.outlet_D()*100:.1f} ซม. ที่ {b.jet_bar:g} บาร์" if b.is_jet else "ใบพัด"
        self.thrust_lbl.set(f"แรงขับต่อ{what} ≈ {b.thrust()/1000:.2f} kN ({b.thrust()/max(b.power_kW, 1e-9):.0f} N/kW), "
                            f"ความเร็วสายน้ำ ≈ {b.jet_speed():.1f} ม./วิ\n"
                            f"กลุ่มเรือ: {fl.n_boats} ลำ, แรงขับรวม {fl.total_thrust()/1000:.1f} kN, "
                            f"กำลังรวม {fl.total_power():.0f} kW")
        sec = self.sec
        Q = float(h1.q_ref(h, self.river, sec))
        fp = min(sec.Bf, cfg.fp_max) if h > sec.Hb else 0
        W = sec.B + 2 * sec.z * min(h, sec.Hb) + 2 * fp
        dx = cfg.dx if cfg.dx > 0 else auto_dx(cfg.length, W, h, cfg)
        nz = max(int(round(h / dx)), 4)
        dx = h / nz
        nx, ny = int(round(cfg.length / dx)), int(math.ceil(W / dx)) + 2
        A = sec.B * min(h, sec.Hb) + sec.z * min(h, sec.Hb) ** 2 + max(h - sec.Hb, 0) * W
        self.reach_info.set(
            f"Q = {Q:,.1f} ม³/วิ (จาก 1D), ความเร็วเฉลี่ย ≈ {Q/max(A, 1e-9):.2f} ม./วิ\n"
            f"กริด {nx}×{ny}×{nz} = {nx*ny*nz/1e6:.2f} ล้านเซลล์, Δx = {dx:.3f} ม., "
            f"หน่วยความจำ GPU ≈ {nx*ny*nz*190/1e6:,.0f} MB")
        if self.run is not None:
            self.stop()
            self.run = None

    def load_experiment(self, i, quiet=False):
        e = EXPERIMENTS[i]
        self.active_exp = e
        self.stop()
        self.trace = None
        self.import_info.set("ยังไม่ได้นำเข้าไฟล์ — ใช้แม่น้ำจำลอง")
        for k, v in self.rvars.items():
            v.set(getattr(e["river"], k))
        for k, v in self.svars.items():
            v.set(getattr(e["section"], k))
        self.levels_var.set(", ".join(f"{x:g}" for x in e["levels"]))
        self.sel_level.set(f"{e['levels'][len(e['levels']) // 2]:g}")
        self.saved = []
        if self._rebuild_job:
            self.after_cancel(self._rebuild_job)
        self.rebuild()
        if not quiet:
            self.nb.select(self.tab_1d)

    def rebuild(self):
        self._rebuild_job = None
        try:
            rp, sec = self.read_params()
            levels = self.read_levels()
        except (ValueError, tk.TclError) as e:
            self.status.set(f"ค่าพารามิเตอร์ไม่ถูกต้อง: {e}")
            return
        self.stop()
        try:
            river = (build_imported(self.trace, sec, rp, self.use_elev.get(), self.reverse.get())
                     if self.trace is not None else generate_river(rp, sec))
        except Exception as e:  # noqa: BLE001
            self.status.set(f"สร้างแม่น้ำไม่สำเร็จ: {e}")
            return
        self.river, self.sec, self.rp, self.levels = river, sec, rp, levels
        for sp in self.river_spins[:6]:
            sp.configure(state="disabled" if self.trace is not None else "normal")
        self.level_cb.configure(values=[f"{x:g}" for x in levels])
        Sc = river.mean_slope
        self.derived.set(
            f"ความยาวลำน้ำ {river.L/1000:.2f} กม. | ระยะทางตรง {river.D/1000:.2f} กม.\n"
            f"ความคดเคี้ยว {river.sigma:.2f} | ความชันท้องน้ำ {Sc*1000:.3f} ม./กม. (1:{1/Sc:,.0f})"
            + ("\n" + " / ".join(river.notes) if river.notes else ""))
        self.compute_1d()
        self.draw_geometry()
        self.run = None
        self._on_reach_change()
        self.status.set(f"แม่น้ำ '{river.name}' พร้อม — ตั้งค่าเรือแล้วกด ▶ เพื่อจำลอง 3 มิติบน {self.be.device_name}")

    # ================================================================ 1D
    def compute_1d(self):
        r, sec = self.river, self.sec
        self.rows1d = [h1.summarize(h, r, sec) for h in self.levels]
        self.curves = h1.rating_curves(r, sec, max(self.levels) * 1.15)
        self.tree1.delete(*self.tree1.get_children())
        for x in self.rows1d:
            self.tree1.insert("", "end", values=(
                f"{x['h0']:.2f}", f"{x['Q']:,.1f}", f"{x['V']:.2f}", f"{x['Vm']:.2f}", f"{x['Frmax']:.2f}",
                f"{x['tau']:.1f}", fmt_time(x["tt"]), f"{x['T']:,.0f}", h1.regime_text(x, sec)))
        self.insight.delete("1.0", "end")
        self.insight.insert("1.0", "ข้อสังเกตอัตโนมัติ\n" + "\n".join("• " + s for s in h1.insights(self.rows1d, sec, r)))
        self.draw_1d()

    def save_scenario(self):
        r = self.river
        self.saved.append(dict(label=f"{r.name[:22]} | S={r.mean_slope*1000:.2f}‰ B={self.sec.B:g} n={self.sec.n:g}",
                               curves=self.curves))
        self.draw_1d()

    def clear_saved(self):
        self.saved = []
        self.draw_1d()

    def run_sweep(self):
        e = self.active_exp
        if e is None or self.trace is not None:
            messagebox.showinfo("ชุดเปรียบเทียบ", "เลือกการทดลองรูปทรงแม่น้ำก่อน (ใช้ได้กับแม่น้ำจำลอง)")
            return
        sw = e["sweep"]
        rp, sec = self.read_params()
        self.saved = []
        for v, lab in zip(sw["values"], sw["labels"]):
            rp2, sec2 = deepcopy(rp), deepcopy(sec)
            setattr(rp2 if sw["target"] == "river" else sec2, sw["param"], v)
            r2 = generate_river(rp2, sec2)
            self.saved.append(dict(label=lab, curves=h1.rating_curves(r2, sec2, max(self.levels) * 1.15)))
        self.draw_1d()

    def draw_1d(self):
        specs = [("Q", "อัตราการไหล Q (ม³/วิ)", "Rating curve"), ("V", "ความเร็วเฉลี่ย (ม./วิ)", "ความเร็วเฉลี่ย"),
                 ("Frmax", "Froude สูงสุด", "ระบอบการไหล"), ("tau", "แรงเฉือน (N/ม²)", "แรงกัดเซาะท้องน้ำ"),
                 ("tt", "เวลา (ชม.)", "เวลาที่น้ำไหลผ่านลำน้ำ"), ("T", "ความกว้างผิวน้ำ (ม.)", "ความกว้างผิวน้ำ")]
        for ax, (k, ylab, title) in zip(self.ax1d, specs):
            ax.clear()
            sc = 1 / 3600 if k == "tt" else 1
            for j, s in enumerate(self.saved):
                ax.plot(s["curves"]["h0"], s["curves"][k] * sc, color=PALETTE[j % 8], lw=1.6, label=s["label"])
            ax.plot(self.curves["h0"], self.curves[k] * sc, color="k", lw=2.2, label="ปัจจุบัน")
            ax.plot([x["h0"] for x in self.rows1d], [x[k] * sc for x in self.rows1d], "o", color="k", ms=4)
            ax.axvline(self.sec.Hb, color="#2a8a2a", ls="--", lw=1)
            if k == "Frmax":
                ax.axhline(1, color="#c33", ls=":", lw=1)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("ระดับน้ำ (ม.)")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
        if self.saved:
            self.ax1d[0].legend(fontsize=7)
        self.cv1d.draw_idle()

    def draw_geometry(self):
        r, sec = self.river, self.sec
        ax = self.axg_plan
        ax.clear()
        hw = sec.Tb / 2
        for sgn in (1, -1):
            ax.plot(r.x + sgn * r.nx * hw, r.y + sgn * r.ny * hw, color="#8a6a3a", lw=0.8)
        ax.plot(r.x, r.y, color="#1f77b4", lw=0.8)
        ax.plot(r.x[0], r.y[0], "o", color="#1f77b4")
        ax.text(r.x[0], r.y[0], "  ต้นน้ำ", va="center")
        ax.plot(r.x[-1], r.y[-1], "s", color="#d62728")
        ax.text(r.x[-1], r.y[-1], "  ปลายน้ำ", va="center")
        ax.set_aspect("equal")
        ax.set_title(f"{r.name} — ยาว {r.L/1000:.2f} กม., ระยะตรง {r.D/1000:.2f} กม., σ={r.sigma:.2f}", fontsize=9)
        ax = self.axg_prof
        ax.clear()
        km = r.s / 1000
        ax.fill_between(km, r.zb.min() - 1, r.zb, color="#9c7b55")
        ax.plot(km, r.zb + sec.Hb, color="#2a8a2a", ls="--", lw=1, label="ระดับตลิ่ง")
        for j, h in enumerate(self.levels):
            st = h1.node_profile(h, r, sec)
            ax.plot(km, r.zb + st["h"], lw=1, color=plt.cm.Blues(0.4 + 0.6 * j / max(len(self.levels) - 1, 1)),
                    label=f"h={h:g}")
        ax.legend(fontsize=7, ncol=2)
        ax.set_xlabel("ระยะตามลำน้ำ (กม.)")
        ax.set_ylabel("ระดับ (ม.)")
        ax.grid(alpha=0.3)
        ax, ax2 = self.axg_sig, self.axg_sig2
        ax.clear()
        ax2.clear()
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax.plot(km, r.sig_loc, color="#9467bd")
        sl = r.slope * 1000
        ax2.plot(km, sl, color="#ff7f0e")
        pad = max(0.1 * np.ptp(sl), 0.05 * sl.mean())
        ax2.set_ylim(sl.min() - pad, sl.max() + pad)
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax2.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax.set_ylabel("ความคดเคี้ยวเฉพาะที่", color="#9467bd")
        ax2.set_ylabel("ความชัน (ม./กม.)", color="#ff7f0e")
        ax.set_xlabel("ระยะตามลำน้ำ (กม.)")
        ax.grid(alpha=0.3)
        self.cvgeo.draw_idle()

    # ================================================================ import / export
    def import_file(self):
        path = filedialog.askopenfilename(title="เลือกไฟล์แม่น้ำ", filetypes=SUPPORTED)
        if not path:
            return
        try:
            tr = read_river_file(path)
            if len(tr["candidates"]) > 1:
                i = self._choose_line(tr["candidates"], tr["picked"])
                if i is None:
                    return
                if i != tr["picked"]:
                    tr = read_river_file(path, pick=i)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("นำเข้าไม่สำเร็จ", f"{type(e).__name__}: {e}")
            return
        self.trace = tr
        zv = tr["z"][np.isfinite(tr["z"])]
        el = f"ความสูง {zv.min():.1f}–{zv.max():.1f} ม." if len(zv) and np.ptp(zv) > 0 else "ไม่มีข้อมูลความสูง"
        self.import_info.set(f"{os.path.basename(path)}: '{tr['name']}' {tr['n_points']} จุด, {el}"
                             + (f"\n{tr['info']}" if tr["info"] else ""))
        self.active_exp = None
        self.rebuild()
        self.nb.select(self.tab_geo)

    def _choose_line(self, cands, default):
        win = tk.Toplevel(self)
        win.title("เลือกเส้นแม่น้ำ")
        win.transient(self)
        win.grab_set()
        ttk.Label(win, text="ไฟล์นี้มีหลายเส้น — เลือกเส้นที่เป็นแนวร่องน้ำ:", padding=8).pack(anchor="w")
        lb = tk.Listbox(win, width=60, height=min(12, len(cands)), font=("Tahoma", 10))
        for n, L in cands:
            lb.insert("end", f"{n}  —  {L:,.2f} กม.")
        lb.selection_set(default)
        lb.pack(padx=8, fill="both", expand=True)
        res = {"i": None}

        def ok(*_):
            sel = lb.curselection()
            res["i"] = sel[0] if sel else default
            win.destroy()

        lb.bind("<Double-Button-1>", ok)
        bf = ttk.Frame(win, padding=8)
        bf.pack(fill="x")
        ttk.Button(bf, text="ตกลง", command=ok).pack(side="right")
        ttk.Button(bf, text="ยกเลิก", command=win.destroy).pack(side="right", padx=4)
        self.wait_window(win)
        return res["i"]

    def clear_import(self):
        self.trace = None
        self.import_info.set("ยังไม่ได้นำเข้าไฟล์ — ใช้แม่น้ำจำลอง")
        self.rebuild()

    def export_kml(self):
        path = filedialog.asksaveasfilename(defaultextension=".kml", filetypes=[("KML", "*.kml")], initialfile="river.kml")
        if path:
            export_kml(self.river, self.sec, path)
            self.status.set(f"บันทึก {path}")

    def export_csv(self):
        if not self.results:
            messagebox.showinfo("ส่งออก", "ยังไม่มีผลเปรียบเทียบเรือ")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="boat_results.csv")
        if path:
            write_results_csv(path, self.results)
            self.status.set(f"บันทึก {path}")

    def export_vtk(self):
        if self.run is None:
            messagebox.showinfo("ส่งออก", "ยังไม่มีผลจำลอง 3D — กด ▶ ก่อน")
            return
        path = filedialog.asksaveasfilename(defaultextension=".vtk", filetypes=[("VTK (ParaView)", "*.vtk")],
                                            initialfile="flow3d.vtk")
        if path:
            write_vtk(path, self.run, mean=self.show_mean.get())
            self.status.set(f"บันทึก {path} — เปิดด้วย ParaView เพื่อดู streamline / isosurface")

    def save_figs(self):
        d = filedialog.askdirectory(title="เลือกโฟลเดอร์สำหรับบันทึกรูป")
        if not d:
            return
        for fig, name in ((self.figS, "slices3d"), (self.fig3, "view3d"), (self.figR, "boat_compare"),
                          (self.fig1d, "levels1d"), (self.figgeo, "geometry")):
            fig.savefig(os.path.join(d, f"{name}.png"), dpi=150)
        self.status.set(f"บันทึกรูปที่ {d}")

    # ================================================================ 3D interactive run
    def ensure_run(self):
        if self.run is not None:
            return True
        try:
            h = self._sel_h()
            Q = float(h1.q_ref(h, self.river, self.sec))
            reach = build_reach(self.sec, h, Q, self.river.mean_slope, self.read_fleet(), self.read_reach_cfg())
            self.status.set("กำลังเตรียมแบบจำลอง 3D …")
            self.update_idletasks()
            self.run = Run3D(reach, self.be)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            messagebox.showerror("สร้างแบบจำลอง 3D ไม่สำเร็จ", f"{type(e).__name__}: {e}")
            return False
        n = int(self.n_particles.get())
        if n > 0:
            self.run.init_particles(n)
        r = reach
        if r.props:
            self.long_y.set(round(r.props[0]["y"], 2))
            self.cross_x.set(round(min(max(p["x"] for p in r.props) + 20, r.cfg.length - 2), 1))
            self.plan_depth.set(round(r.h - r.props[0]["z"], 2))
        else:
            self.long_y.set(0.0)
            self.cross_x.set(round(r.cfg.length / 2, 1))
            self.plan_depth.set(round(0.3 * r.h, 2))
        self._3d_view_set = False
        self.status.set(f"3D: กริด {r.nx}×{r.ny}×{r.nz} = {r.cells/1e6:.2f} ล้านเซลล์, Δx={r.dx:.3f} ม., "
                        f"Δt={r.dt*1000:.2f} มิลลิวินาที, Q={r.Q:,.1f} ม³/วิ, V={r.V:.2f} ม./วิ"
                        + (" | " + "; ".join(r.notes) if r.notes else ""))
        return True

    def toggle_run(self):
        if self.running:
            self.stop()
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("กำลังทำงาน", "รอการเปรียบเทียบที่รันอยู่ให้เสร็จก่อน (หรือกดยกเลิก)")
            return
        if not self.ensure_run():
            return
        self.running = True
        self.run_btn.configure(text="⏸ หยุด")
        if self.nb.select() not in (str(self.tab_slices), str(self.tab_view)):
            self.nb.select(self.tab_slices)
        self.after(1, self._tick)

    def stop(self):
        self.running = False
        if hasattr(self, "run_btn"):
            self.run_btn.configure(text="▶ เริ่ม")

    def reset_run(self):
        self.stop()
        self.run = None
        if self.ensure_run():
            self._refresh_visible()

    def _tick(self):
        if not self.running or self.run is None:
            return
        run = self.run
        t0 = time.time()
        chunk = 20 if self.be.is_gpu else 1
        nst = 0
        while time.time() - t0 < (0.08 if self.be.is_gpu else 0.3):
            run.advance(chunk)
            nst += chunk
        run.move_particles(nst * run.r.dt)
        self._frame += 1
        cur = self.nb.select()
        if cur == str(self.tab_slices):
            self.draw_slices()
        elif cur == str(self.tab_view) and self._frame % 3 == 0:
            self.draw_view3d()
        self.status.set(f"เวลาในแม่น้ำ {run.t:,.1f} วินาที (ครบเวลาเฉลี่ยที่ {run.t_total:,.0f}) | "
                        f"{run.lbm.nstep:,} ก้าว | {run.mlups():,.0f} ล้านเซลล์/วินาที | ค่าเฉลี่ย {run.navg} ภาพ")
        self.after(1, self._tick)

    def _refresh_visible(self):
        if self.run is None:
            return
        cur = self.nb.select()
        if cur == str(self.tab_slices):
            self.draw_slices()
        elif cur == str(self.tab_view):
            self.draw_view3d()

    def _on_slice_click(self, ev):
        if self.run is None or ev.inaxes is not self.axP or (self.tbS and self.tbS.mode):
            return
        self.cross_x.set(round(ev.xdata, 1))
        self.long_y.set(round(ev.ydata, 2))
        self.draw_slices()

    def _speed_norm(self):
        r = self.run.r
        jet = max([p["Vjet"] for p in r.props], default=0)
        return Normalize(0, max(1.6 * float(r.uin_phys.max()), 0.45 * jet + r.V, 0.3))

    def draw_slices(self):
        draw_slices_fig(self, self.run, bool(self.show_mean.get()))
        self.cvS.draw_idle()

    def draw_view3d(self):
        ax = self.ax3
        if getattr(self, "_3d_view_set", False):
            elev, azim = ax.elev, ax.azim
        else:
            elev, azim = 28, -62
        draw_view3d_ax(ax, self.run, self._speed_norm())
        ax.view_init(elev=elev, azim=azim)
        self._3d_view_set = True
        self.cv3.draw_idle()

    # ================================================================ comparisons (background)
    def _busy(self, on):
        st = "disabled" if on else "normal"
        self.cmp_btn.configure(state=st)
        for b in self.test_btns:
            b.configure(state=st)

    def _cancel(self):
        self.cancel_flag = True

    def run_compare(self):
        """รันกรณีไม่มีเรือ + เรือชุดปัจจุบัน แล้วเพิ่มผลในตาราง"""
        if self.worker and self.worker.is_alive():
            return
        self.stop()
        try:
            fleet = self.read_fleet()
            cfg = self.read_reach_cfg()
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("ค่าไม่ถูกต้อง", str(e))
            return
        fleet = replace(fleet, enabled=True)
        test = dict(title="กลุ่มเรือปัจจุบัน",
                    variants=[dict(label=fleet.label(), boat=fleet.boat.key, rows=fleet.rows,
                                   per_row=fleet.per_row, row_spacing=fleet.row_spacing)])
        edits = {fleet.boat.key: {f.name: getattr(fleet.boat, f.name) for f in fields(fleet.boat)
                                  if f.name not in ("key", "name", "note")}}
        self._start_test(test, self._sel_h(), cfg, fleet.x_first, edits)

    def run_boat_test(self, i):
        if self.worker and self.worker.is_alive():
            return
        self.stop()
        t = BOAT_TESTS[i]
        n = len(t["variants"])
        msg = (f"{t['title']}\n\n{t['objective']}\n\nจะรัน {n} กรณี + กรณีไม่มีเรือ "
               f"(ประมาณ {2*n}–{6*n} นาทีบน GPU) ต้องการเริ่มหรือไม่?")
        if not messagebox.askyesno("การทดสอบเรือ", msg):
            return
        try:
            cfg = self.read_reach_cfg()
            x_first = float(self.fvars["x_first"].get())
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("ค่าไม่ถูกต้อง", str(e))
            return
        self.res_text.delete("1.0", "end")
        self.res_text.insert("1.0", f"{t['title']}\nวัตถุประสงค์: {t['objective']}\nสังเกต:\n"
                             + "\n".join("• " + o for o in t["observe"]) + "\n")
        self._start_test(t, self._sel_h(), cfg, x_first, dict(self.boat_edits))

    def _start_test(self, test, h, cfg, x_first, edits):
        self.cancel_flag = False
        self._busy(True)
        self.nb.select(self.tab_cmp)
        river, sec, levels = self.river, self.sec, list(self.levels)

        def flow(hh):
            return float(h1.q_ref(hh, river, sec)), float(river.mean_slope)

        def work():
            try:
                run_test(test, river, sec, levels, h, cfg, self.be, flow, x_first=x_first,
                         progress=lambda f, lab: self.q.put(("prog", (f, lab))),
                         cancel=lambda: self.cancel_flag,
                         on_result=lambda c, run: self.q.put(("res", c)), boat_edits=edits)
                self.q.put(("done", test["title"]))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self.q.put(("err", f"{type(e).__name__}: {e}"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _poll_queue(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "prog":
                    f, lab = val
                    self.prog["value"] = f
                    self.status.set(f"กำลังรัน: {lab} ({f*100:.0f}%)")
                elif kind == "res":
                    self.add_result(val)
                elif kind == "done":
                    self._busy(False)
                    self.prog["value"] = 1.0
                    self.status.set(f"เสร็จ: {val}" + (" (ยกเลิก)" if self.cancel_flag else ""))
                elif kind == "err":
                    self._busy(False)
                    messagebox.showerror("ผิดพลาด", val)
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def add_result(self, c):
        self.results.append(c)
        self.tree.insert("", "end", values=result_row(c))
        self.draw_results()
        self.res_text.insert("end", "\n" + "\n".join(summary_text(c, self.read_reach_cfg())) + "\n")
        self.res_text.see("end")

    def clear_results(self):
        self.results = []
        self.tree.delete(*self.tree.get_children())
        self.draw_results()

    def _show_result_text(self):
        sel = self.tree.selection()
        if not sel:
            return
        c = self.results[self.tree.index(sel[0])]
        self.res_text.delete("1.0", "end")
        self.res_text.insert("1.0", "\n".join(summary_text(c, self.read_reach_cfg()))
                             + ("\nหมายเหตุ: " + "; ".join(c["notes"]) if c.get("notes") else ""))

    def draw_results(self):
        draw_results_fig(self.figR, self.results)
        self.cvR.draw_idle()

    def _on_close(self):
        self.running = False
        self.cancel_flag = True
        self.destroy()


# ==================================================================== drawing helpers
def result_row(c):
    return (c["label"], f"{c['h']:g}", f"{c['T_kN']:.1f}", f"{c['P_kW']:.0f}", f"{c['d_eta_cm']:+.2f}",
            f"{c['jet_reach']:.0f}" + ("+" if c["reach_capped"] else ""), f"{c['us_gain']:+.3f}",
            f"{c['umax']:.2f}", f"{c['tau_max']:.1f}", f"{c['dQ_pct']:.2f}", f"{c['dQ_net_pct']:.2f}")


def run_line_stats(run, mean):
    """ความเร็วสูงสุดในแต่ละหน้าตัด และระดับน้ำ (จากความดันใต้ผิวน้ำ) — ลดข้อมูลบน GPU ก่อนโอน"""
    r, L, xp = run.r, run.lbm, run.xp
    if not hasattr(run, "_fluid_dev"):
        run._fluid_dev = xp.asarray((~r.solid).astype(np.float32).ravel())
    fl = run._fluid_dev.reshape(r.nz, r.ny, r.nx)
    if mean and run.navg > 0:
        sp = (run.sum["sp"] / run.navg).reshape(r.nz, r.ny, r.nx)
        rho = (run.sum["rho"] / run.navg).reshape(r.nz, r.ny, r.nx)
    else:
        sp = xp.sqrt(L.ux * L.ux + L.uy * L.uy + L.uz * L.uz).reshape(r.nz, r.ny, r.nx)
        rho = L.rho.reshape(r.nz, r.ny, r.nx)
    umax = run.be.to_np((sp * fl).max(axis=(0, 1))) * r.vel
    top = fl[-1]
    eta = r.eta_from_rho(run.be.to_np((rho[-1] * top).sum(0) / xp.maximum(top.sum(0), 1)))
    return eta - eta[-1], umax


def _speed_norm(r):
    jet = max([p["Vjet"] for p in r.props], default=0)
    return Normalize(0, max(1.6 * float(r.uin_phys.max()), 0.45 * jet + r.V, 0.3))


def draw_slices_fig(app, run, mean):
    """วาดระนาบแนวนอน แนวตัดตามยาว หน้าตัดขวาง และค่าตามแนวลำน้ำ"""
    r = run.r
    mean = mean and run.navg > 0
    norm = _speed_norm(r)
    x = r.x
    y0, y1 = r.y[0] - r.dx / 2, r.y[-1] + r.dx / 2
    try:
        dep, ys, xs = float(app.plan_depth.get()), float(app.long_y.get()), float(app.cross_x.get())
    except (tk.TclError, ValueError):
        return
    kz = int(np.clip(r.nz - 1 - round(dep / r.dx - 0.5), 0, r.nz - 1))
    jy = int(np.clip(np.searchsorted(r.y, ys), 0, r.ny - 1))
    ix = int(np.clip(xs / r.dx, 0, r.nx - 1))
    tag = "เฉลี่ยตามเวลา" if mean else "ขณะนั้น"
    grey = dict(cmap="Greys", vmin=0, vmax=2, aspect="auto")

    def panel(ax, sp, sol, ext, u, v, X, Y, scale, w=0.0015):
        im = ax.imshow(np.ma.masked_where(sol, sp), origin="lower", extent=ext, cmap="turbo", norm=norm,
                       aspect="auto", interpolation="bilinear")
        ax.imshow(np.ma.masked_where(~sol, np.ones_like(sp)), origin="lower", extent=ext, **grey)
        ax.quiver(X, Y, u, v, color="white", alpha=0.65, scale=scale, width=w)
        return im

    # ---- ระนาบแนวนอน
    ax = app.axP
    ax.clear()
    sp, ux, uy = (run.slice_np(k, 0, kz, mean) for k in ("sp", "ux", "uy"))
    sol = r.solid[kz]
    st = max(1, int(max(r.nx, r.ny) / 70))
    XX, YY = np.meshgrid(x[::st], r.y[::st])
    im = panel(ax, sp, sol, (0, r.nx * r.dx, y0, y1), np.where(sol, 0, ux)[::st, ::st],
               np.where(sol, 0, uy)[::st, ::st], XX, YY, norm.vmax * 45)
    for (a0, a1, b0, b1, z0, z1) in r.hulls:
        ax.add_patch(Rectangle((a0, b0), a1 - a0, b1 - b0, fc="none", ec="k", lw=1))
    for p in r.props:
        ax.plot([p["x"]], [p["y"]], marker="|", color="red", ms=8, mew=2)
    ax.axhline(r.y[jy], color="magenta", lw=0.8, ls="--")
    ax.axvline(x[ix], color="magenta", lw=0.8, ls="--")
    ax.set_xlim(0, r.nx * r.dx)
    ax.set_ylim(y0, y1)
    ax.set_title(f"ระนาบแนวนอน ลึก {r.h - r.zc[kz]:.2f} ม. จากผิวน้ำ — ความเร็ว ({tag}) | น้ำไหลจากซ้ายไปขวา | "
                 f"{r.fleet.label()}", fontsize=9)
    ax.set_ylabel("y (ม.)")
    if app.cbS is None or app.cbS.ax not in app.figS.axes:
        app.cbS = app.figS.colorbar(im, ax=[app.axP, app.axL], shrink=0.9, pad=0.01, label="ความเร็ว (ม./วิ)")
    else:
        app.cbS.update_normal(im)

    # ---- แนวตัดตามยาว
    ax = app.axL
    ax.clear()
    spL, uxL, uzL = (run.slice_np(k, 1, jy, mean) for k in ("sp", "ux", "uz"))
    solL = r.solid[:, jy, :]
    stx, stz = max(1, r.nx // 70), max(1, r.nz // 10)
    XX, ZZ = np.meshgrid(x[::stx], r.zc[::stz])
    panel(ax, spL, solL, (0, r.nx * r.dx, 0, r.h), np.where(solL, 0, uxL)[::stz, ::stx],
          np.where(solL, 0, uzL)[::stz, ::stx], XX, ZZ, norm.vmax * 45)
    for p in r.props:
        if abs(p["y"] - r.y[jy]) < p["Deff"]:
            ax.plot([p["x"]], [p["z"]], marker="o", mfc="none", mec="red", ms=6)
    ax.axvline(x[ix], color="magenta", lw=0.8, ls="--")
    ax.set_title(f"แนวตัดตามยาวที่ y = {r.y[jy]:.2f} ม. (มองด้านข้าง ผิวน้ำอยู่บน)", fontsize=9)
    ax.set_ylabel("สูงจากท้องน้ำ (ม.)")
    ax.set_xlabel("ระยะจากทางเข้า (ม.)")

    # ---- หน้าตัดขวาง
    ax = app.axC
    ax.clear()
    uxC, uyC, uzC = (run.slice_np(k, 2, ix, mean) for k in ("ux", "uy", "uz"))
    solC = r.solid[:, :, ix]
    sty, stz = max(1, r.ny // 30), max(1, r.nz // 10)
    YY, ZZ = np.meshgrid(r.y[::sty], r.zc[::stz])
    vmag = float(np.sqrt(uyC ** 2 + uzC ** 2).max())
    panel(ax, uxC, solC, (y0, y1, 0, r.h), np.where(solC, 0, uyC)[::stz, ::sty],
          np.where(solC, 0, uzC)[::stz, ::sty], YY, ZZ, max(vmag, 0.02) * 25, w=0.003)
    ax.set_title(f"หน้าตัดขวางที่ x = {x[ix]:.1f} ม. — สี: ความเร็วตามลำน้ำ, ลูกศร: การไหลวนในหน้าตัด", fontsize=9)
    ax.set_xlabel("y (ม.)  ← ตลิ่งซ้าย | ตลิ่งขวา →")
    ax.set_ylabel("z (ม.)")

    # ---- ตามแนวลำน้ำ
    ax, ax2 = app.axE, app.axE2
    ax.clear()
    ax2.clear()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    eta, umax = run_line_stats(run, mean)
    ax.plot(x, umax, color="#d62728", lw=1.4)
    ax.set_ylabel("ความเร็วสูงสุดในหน้าตัด (ม./วิ)", color="#d62728")
    ax.set_xlabel("ระยะจากทางเข้า (ม.)")
    ax2.plot(x, eta * 100, color="#1f77b4", lw=1.4)
    ax2.set_ylabel("ระดับน้ำเทียบท้ายช่วง (ซม.)", color="#1f77b4")
    for p in r.props:
        ax.axvline(p["x"], color="#999", lw=0.6, ls=":")
    ax.set_title(f"ตามแนวลำน้ำ ({tag}): ความเร็วสูงสุด และระดับผิวน้ำ", fontsize=9)
    ax.grid(alpha=0.3)


def draw_view3d_ax(ax, run, norm):
    r = run.r
    ax.clear()
    ve = float(np.clip(0.35 * r.W / r.h, 1, 8))
    yy = np.linspace(r.y[0], r.y[-1], 40)
    xx = np.linspace(0, r.nx * r.dx, 40)
    X, Y = np.meshgrid(xx, yy)
    Zb = np.broadcast_to(np.minimum(bed_level(yy, r.sec), r.h)[:, None], X.shape)
    ax.plot_surface(X, Y, Zb * ve, color="#a08560", alpha=0.5, linewidth=0, shade=True)
    for (a0, a1, b0, b1, z0, z1) in r.hulls:
        ax.add_collection3d(Poly3DCollection(_box(a0, a1, b0, b1, z0 * ve, (z1 + 0.3 * (z1 - z0)) * ve),
                                             facecolor="#6b6b6b", edgecolor="#333", alpha=0.95))
    th = np.linspace(0, 2 * np.pi, 24)
    for p in r.props:
        t = math.radians(p["tilt"])
        rr = max(p["D"], 0.3) / 2
        ax.plot(p["x"] + rr * np.cos(th) * math.sin(t), p["y"] + rr * np.sin(th),
                (p["z"] + rr * np.cos(th) * math.cos(t)) * ve, color="red", lw=1.5)
    if run.particles:
        px, py, pz = run.particles_xyz()
        ix = np.clip((px / r.dx).astype(int), 0, r.nx - 1)
        iy = np.clip(((py - r.y[0]) / r.dx + 0.5).astype(int), 0, r.ny - 1)
        iz = np.clip((pz / r.dx).astype(int), 0, r.nz - 1)
        ci = run.xp.asarray(ix + r.nx * (iy + r.ny * iz))
        L = run.lbm
        spd = np.sqrt(run.be.to_np(L.ux[ci]) ** 2 + run.be.to_np(L.uy[ci]) ** 2
                      + run.be.to_np(L.uz[ci]) ** 2) * r.vel
        ax.scatter(px, py, pz * ve, c=spd, cmap="turbo", norm=norm, s=2, depthshade=False)
    ax.set_xlim(0, r.nx * r.dx)
    ax.set_ylim(r.y[0], r.y[-1])
    ax.set_zlim(0, r.h * ve * 1.15)
    ax.set_box_aspect((r.nx * r.dx, r.y[-1] - r.y[0], r.h * ve * 1.15))
    ax.set_xlabel("ระยะตามลำน้ำ (ม.)")
    ax.set_ylabel("y (ม.)")
    ax.set_zlabel(f"z (ขยาย ×{ve:.1f})")
    ax.set_zticks([])
    ax.set_title(f"ช่วงลำน้ำ 3 มิติ — {r.fleet.label()} | ระดับน้ำ {r.h:g} ม. | เวลา {run.t:,.1f} วินาที", fontsize=9)


def _box(x0, x1, y0, y1, z0, z1):
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return [[v[i] for i in q] for q in f]


def draw_results_fig(fig, results):
    fig.clear()
    if not results:
        ax = fig.add_subplot()
        ax.text(0.5, 0.5, "กด 'เปรียบเทียบ' หรือเลือก 'การทดสอบเรือแนะนำ' เพื่อเริ่มเปรียบเทียบ",
                ha="center", va="center", transform=ax.transAxes, color="#555")
        ax.set_axis_off()
        return
    gs = fig.add_gridspec(2, 5)
    labs = [c["label"] for c in results]
    cols = [PALETTE[i % 8] for i in range(len(results))]
    specs = [("d_eta_cm", "Δระดับน้ำเหนือเรือ (ซม.)\nลบ = ลดระดับน้ำได้"),
             ("drop_cm_per_100kW", "ลดระดับน้ำได้ต่อกำลัง\n(ซม. ต่อ 100 kW) — สูง = คุ้มค่า"),
             ("jet_reach", "ระยะสายน้ำเร็ว (ม.)"),
             ("us_gain_pct", "ผิวน้ำท้ายเรือเร็วขึ้น (%)"), ("tau_max", "แรงเฉือนท้องน้ำสูงสุด (N/ม²)"),
             ("dQ_net_pct", "ΔQ สุทธิ (%)\nถ้าวางซ้ำทุกระยะที่กำหนด"), ("T_per_kW", "แรงขับต่อกำลัง (N/kW)")]
    pos = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1)]
    for (k, title), (i, j) in zip(specs, pos):
        ax = fig.add_subplot(gs[i, j])
        vals = [c[k] for c in results]
        ax.bar(range(len(vals)), vals, color=cols)
        if k == "tau_max":
            ax.axhline(results[0]["tau_base"], color="k", ls="--", lw=1, label="ไม่มีเรือ")
            ax.legend(fontsize=7)
        if k == "dQ_net_pct":
            ax.plot(range(len(vals)), [c["dQ_pct"] for c in results], "k_", ms=14, mew=2, label="จากแรงขับล้วน")
            ax.legend(fontsize=7)
        ax.axhline(0, color="#444", lw=0.6)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=0.3, axis="y")
    ax = fig.add_subplot(gs[1, 2:])
    for c, col in zip(results, cols):
        p = c["profiles"]
        ax.plot(p["x"], p["umax"], color=col, lw=1.3, label=c["label"])
    b = results[-1]["base_profiles"]
    ax.plot(b["x"], b["umax"], color="k", ls="--", lw=1.2, label="ไม่มีเรือ")
    ax.set_title("ความเร็วสูงสุดในหน้าตัด ตามแนวลำน้ำ (เฉลี่ยตามเวลา)", fontsize=8.5)
    ax.set_xlabel("ระยะจากทางเข้า (ม.)")
    ax.set_ylabel("ม./วิ")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def write_results_csv(path, results):
    keys = ["label", "h", "Q", "V", "n", "T_kN", "P_kW", "T_per_kW", "F_net_kN", "d_eta_cm", "jet_reach",
            "reach_capped", "us_gain", "us_gain_pct", "umax", "tau_max", "tau_base", "tau_area", "tau_area_base",
            "dQ_pct", "dQ_net_pct", "dQ_per_100kW", "drop_cm_per_100kW", "kW_per_cm"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for c in results:
            w.writerow([c[k] if isinstance(c[k], str) else f"{c[k]:.5g}" for k in keys])


def write_vtk(path, run, mean=False):
    """ส่งออกสนามความเร็ว 3D เป็น VTK legacy (binary) สำหรับ ParaView"""
    r = run.r
    ux, uy, uz = (run.field(k, mean) for k in ("ux", "uy", "uz"))
    sp = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    with open(path, "wb") as f:
        f.write((f"# vtk DataFile Version 3.0\nFlood Sim 3D\nBINARY\nDATASET STRUCTURED_POINTS\n"
                 f"DIMENSIONS {r.nx} {r.ny} {r.nz}\nORIGIN {0.5*r.dx} {r.y[0]} {0.5*r.dx}\n"
                 f"SPACING {r.dx} {r.dx} {r.dx}\nPOINT_DATA {r.cells}\n").encode())
        f.write(b"VECTORS velocity float\n")
        f.write(np.stack([ux, uy, uz], -1).astype(">f4").tobytes())
        f.write(b"\nSCALARS speed float 1\nLOOKUP_TABLE default\n")
        f.write(sp.astype(">f4").tobytes())
        f.write(b"\nSCALARS solid float 1\nLOOKUP_TABLE default\n")
        f.write(r.solid.astype(">f4").tobytes())


def guide_3d() -> str:
    lines = ["คู่มือแบบจำลอง 3 มิติสำหรับทดสอบเรือดันน้ำ", "=" * 44, "",
             "ขั้นตอน",
             "  1. กำหนดแม่น้ำและหน้าตัด (หรือนำเข้าไฟล์) แล้วดูแท็บ 'ระดับน้ำ (1D)' เพื่อเลือกระดับน้ำ",
             "  2. เลือกชนิดเรือ ปรับค่า (ขนาด ใบพัด กำลังเครื่อง) และการจัดกลุ่มเรือ",
             "  3. กด ▶ เพื่อดูการไหล 3 มิติแบบเคลื่อนไหว (ระนาบแนวนอน แนวตัดตามยาว หน้าตัดขวาง มุมมอง 3D)",
             "  4. กด 'เปรียบเทียบ' หรือเลือกการทดสอบแนะนำ เพื่อรันกรณีไม่มีเรือ/มีเรือ และสรุปผลเป็นตัวเลข",
             "", "การทดสอบเรือแนะนำ"]
    for t in BOAT_TESTS:
        lines.append(f"  {t['title']}: {t['objective']}")
        lines += ["     • " + o for o in t["observe"]]
    lines += ["", "ความหมายของผล",
              "  • Δระดับน้ำ: ระดับน้ำต้นช่วงเทียบท้ายช่วง เปลี่ยนไปจากกรณีไม่มีเรือ (ลบ = เรือช่วยดึงระดับน้ำด้านเหนือลง)",
              "    ในช่วงสั้น ๆ ค่านี้ยังไม่รวมโมเมนตัมที่สายน้ำพาออกไปท้ายช่วง",
              "  • ระยะสายน้ำ: ระยะหลังใบพัดที่น้ำยังเร็วกว่ากรณีไม่มีเรือเกินเกณฑ์ ('+' = ยาวเกินช่วงจำลอง)",
              "  • แรงเฉือนท้องน้ำ: ประเมินความเสี่ยงกัดเซาะท้องน้ำ/ตลิ่ง/ฐานรากใกล้เรือ",
              "  • ΔQ จากแรงขับ: ถ้าแรงขับทั้งหมดกลายเป็นแรงผลักน้ำ (ขอบเขตบน)",
              "  • ΔQ สุทธิ: ใช้แรงสุทธิที่วัดได้จากแบบจำลอง (แรงขับ − แรงต้านตัวเรือ − แรงเสียดทานเพิ่ม)",
              "    ทั้งสองค่าคิดแบบ 'วางกลุ่มเรือนี้ซ้ำทุกระยะที่กำหนด' ด้วยสมดุลโมเมนตัมและสมการแมนนิ่ง",
              "", "หลักการและข้อจำกัด",
              "  • Lattice Boltzmann D3Q19 + LES (Smagorinsky) บน GPU (CUDA kernel เขียนเฉพาะ)",
              "  • ผิวน้ำแบบ rigid lid: ระดับน้ำคำนวณจากความดันใต้ผิวน้ำ — เหมาะกับแม่น้ำที่ราบ (Froude ต่ำ)",
              "    คลื่นผิวน้ำและการยุบตัวของผิวน้ำรอบใบพัดตื้น ๆ จะไม่ถูกจำลอง",
              "  • ใบพัดเป็น actuator disk (แรงขับกระจายในจาน ไม่มีการหมุนวน) — ใบพัดที่เล็กกว่า 3 เซลล์ถูกขยาย",
              "    โดยคงแรงขับเท่าเดิม ความเร็วใกล้ใบพัดจึงต่ำกว่าจริง แต่โมเมนตัมและผลระยะไกลถูกต้อง",
              "  • แรงขับคำนวณจากกำลังเครื่องด้วยทฤษฎี actuator disk (ประสิทธิภาพ 0.6) — ใส่ค่าแรงขับจริงได้ถ้าทราบ",
              "  • เรือจอดนิ่ง (ผูกทุ่น) ตัวเรือเป็นกล่องหัวเรียว; ท้องน้ำเป็นผนังแบบขั้นบันได",
              "  • ช่วงจำลองเป็นลำน้ำตรงตามหน้าตัดที่กำหนด ทางเข้าใช้โปรไฟล์ความเร็วกำลัง 1/7 ตาม Q จาก 1D"]
    return "\n".join(lines)


def main(backend="auto"):
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001
        pass
    App(backend).mainloop()
