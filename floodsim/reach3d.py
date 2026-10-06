"""ช่วงลำน้ำ 3 มิติสำหรับทดสอบเรือดันน้ำ: เรขาคณิต, เรือ/ใบพัด, หน่วย, การรัน และสรุปผล"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from . import hydraulics1d as h1
from .geometry import Section
from .lbm3d import LBM3D

G = 9.81
RHO = 1000.0
N_SLOSH = 10   # จำนวนรอบคลื่นความดันไป-กลับขั้นต่ำในช่วงหาค่าเฉลี่ยเวลา
# สัดส่วนเวลาที่ให้ GPU ทำงาน (1 = เต็มที่, 0.5 = พักเท่ากับเวลาที่คำนวณ → ใช้พลังงาน/ความร้อนเฉลี่ยครึ่งหนึ่ง แต่ช้าลง 2 เท่า)
GPU_DUTY = float(os.environ.get("FLOODSIM_GPU_DUTY", "1"))


# ================================================================ เรือ
@dataclass
class BoatType:
    key: str
    name: str
    hull_L: float          # ความยาวลำเรือ (ม.)
    hull_B: float          # ความกว้างลำเรือ (ม.)
    draft: float           # ระยะกินน้ำ (ม.)
    n_props: int           # จำนวนใบพัดต่อลำ
    prop_D: float          # เส้นผ่านศูนย์กลางใบพัด (ม.)
    prop_depth: float      # ความลึกจุดศูนย์กลางใบพัดจากผิวน้ำ (ม.)
    tilt_deg: float        # มุมกดใบพัดลง (องศา, 0 = แนวนอน)
    prop_x: float          # ตำแหน่งใบพัดหลังท้ายเรือ (ม.; ลบ = ใต้ท้องเรือ)
    prop_spacing: float    # ระยะห่างใบพัดแนวขวาง (ม.)
    power_kW: float        # กำลังเครื่องต่อใบพัด (kW)
    merit: float = 0.6     # ประสิทธิภาพใบพัด (figure of merit)
    thrust_N: float = 0.0  # กำหนดแรงขับเอง (0 = คำนวณจากกำลัง)
    note: str = ""
    jet_bar: float = 0.0   # หัวฉีดน้ำ: แรงดันที่หัวฉีด (บาร์); 0 = ใบพัด  (ความเร็วสายน้ำ v = √(2Δp/ρ))
    spread_deg: float = 0.0  # ฉีดกระจาย: แตกสายน้ำเป็น 3 ทิศ −θ, 0, +θ ในแนวราบ (องศา)

    @property
    def is_jet(self) -> bool:
        return self.jet_bar > 0

    def thrust(self) -> float:
        """แรงขับต่อใบพัด/หัวฉีด (N)
        ใบพัด: ทฤษฎี actuator disk  T = (2ρA·(ηP)²)^(1/3)
        หัวฉีด: ปั๊มให้กำลังน้ำ ηP = ½·ṁ·v²  และ T = ṁ·v  →  T = 2ηP / v   (แรงดันสูง = v สูง = แรงขับต่อกำลังต่ำ)
        """
        if self.thrust_N > 0:
            return self.thrust_N
        if self.is_jet:
            return 2 * self.merit * self.power_kW * 1000 / self.jet_speed()
        A = math.pi * self.prop_D ** 2 / 4
        return (2 * RHO * A * (self.merit * self.power_kW * 1000) ** 2) ** (1 / 3)

    def jet_speed(self) -> float:
        """ความเร็วสายน้ำ (ม./วิ): หัวฉีด √(2Δp/ρ), ใบพัด ≈ 2·√(T / 2ρA)"""
        if self.is_jet:
            return math.sqrt(2 * self.jet_bar * 1e5 / RHO)
        A = max(math.pi * self.prop_D ** 2 / 4, 1e-6)
        return 2 * math.sqrt(self.thrust() / (2 * RHO * A))

    def outlet_D(self) -> float:
        """เส้นผ่านศูนย์กลางใบพัด หรือขนาดหัวฉีดที่ได้แรงขับตามกำลังและแรงดัน (T = ρ·A·v²)"""
        if self.is_jet:
            v = self.jet_speed()
            return math.sqrt(4 * self.thrust() / (math.pi * RHO * v * v))
        return self.prop_D


BOAT_TYPES = [
    BoatType("longtail", "เรือหางยาว", 9.0, 1.6, 0.35, 1, 0.30, 0.35, 12, 2.5, 0, 75,
             note="เครื่องยนต์รถยนต์ ~100 แรงม้า เพลายาว ใบพัดเล็กหมุนเร็ว อยู่ตื้นและเอียงลง"),
    BoatType("outboard", "เรือเครื่องท้าย (outboard)", 7.0, 2.4, 0.5, 1, 0.36, 0.6, 0, 0.3, 0, 110,
             note="เครื่องติดท้าย ~150 แรงม้า ใบพัดแนวนอน"),
    BoatType("tug", "เรือลากจูง/เรือผลักดันน้ำ (กองทัพเรือ)", 16.0, 5.0, 1.8, 2, 1.2, 1.6, 0, -1.0, 2.4, 300,
             note="2 ใบพัดใหญ่ใต้ท้ายเรือ ~400 แรงม้าต่อเครื่อง"),
    BoatType("pontoon", "เครื่องผลักดันน้ำบนทุ่นลอย", 6.0, 4.0, 0.5, 1, 1.6, 1.3, 0, 0.5, 0, 30,
             note="ใบพัดใหญ่หมุนช้า กำลังน้อย แต่แรงขับต่อกำลังสูง"),
    BoatType("jet_hp", "เครื่องฉีดน้ำแรงดันสูง (100 บาร์)", 6.0, 4.0, 0.5, 4, 0.0, 1.2, 0, 0.3, 0.8, 25,
             merit=0.7, jet_bar=100,
             note="ปั๊มแรงดันสูง 4 หัวฉีด สายน้ำ ~141 ม./วิ หัวฉีดเล็กมาก — แรงขับต่อกำลังต่ำ (T = 2ηP/v)"),
    BoatType("jet_mp", "เครื่องฉีดน้ำแรงดันปานกลาง (10 บาร์)", 6.0, 4.0, 0.5, 4, 0.0, 1.2, 0, 0.3, 0.8, 25,
             merit=0.7, jet_bar=10, note="ปั๊มหอยโข่ง 4 หัวฉีด สายน้ำ ~45 ม./วิ"),
    BoatType("curtain", "ม่านน้ำความเร็วต่ำ (ท่อพ่นใหญ่)", 6.0, 6.0, 0.5, 4, 0.0, 1.3, 0, 0.3, 1.4, 25,
             merit=0.7, jet_bar=0.045,
             note="ปั๊มไหลตามแกน 4 ท่อ ~1.3 ม. เรียงขวาง พ่นน้ำปริมาณมากที่ ~3 ม./วิ — แรงขับต่อกำลังสูงสุด"),
    BoatType("custom", "กำหนดเอง", 10.0, 3.0, 0.8, 1, 0.8, 1.0, 0, 0.5, 0, 100),
]


def boat_type(key: str) -> BoatType:
    return replace(next(b for b in BOAT_TYPES if b.key == key))


@dataclass
class Fleet:
    boat: BoatType
    rows: int = 1           # จำนวนแถว (ตามลำน้ำ)
    per_row: int = 3        # จำนวนลำต่อแถว (ขวางลำน้ำ)
    row_spacing: float = 30.0
    x_first: float = 40.0   # ตำแหน่งหัวเรือแถวแรก นับจากทางเข้า (ม.)
    enabled: bool = True

    @property
    def n_boats(self):
        return self.rows * self.per_row if self.enabled else 0

    def total_thrust(self):
        return self.n_boats * self.boat.n_props * self.boat.thrust()

    def total_power(self):
        return self.n_boats * self.boat.n_props * self.boat.power_kW

    def label(self):
        if not self.enabled or self.n_boats == 0:
            return "ไม่มีเรือ"
        return f"{self.boat.name} {self.rows}×{self.per_row} ลำ"


@dataclass
class ReachConfig:
    length: float = 150.0      # ความยาวช่วงจำลอง (ม.)
    dx: float = 0.0            # ขนาดเซลล์ (ม.) 0 = อัตโนมัติ
    max_cells: int = 2_000_000
    fp_max: float = 40.0       # ความกว้างที่ราบน้ำท่วมที่รวมในโดเมนสูงสุด (ม./ฝั่ง)
    u_lb: float = 0.14         # ความเร็วสูงสุดในหน่วยแลตทิซ (ควบคุม Mach)
    cs: float = 0.17           # ค่าคงที่ Smagorinsky
    run_flows: float = 1.0     # เวลาจำลอง = run_flows × (ความยาว / ความเร็วเฉลี่ย)
    avg_frac: float = 0.4      # ช่วงท้ายที่ใช้หาค่าเฉลี่ยเวลา
    spacing_km: float = 1.0    # ระยะห่างกลุ่มเรือตามลำน้ำ (สำหรับประมาณ ΔQ)
    tau_crit: float = 2.0      # แรงเฉือนวิกฤตของท้องน้ำ/ตลิ่ง (N/ม²)
    wall_model: bool = True    # wall model: ท้องน้ำ/ตลิ่งตามแมนนิ่ง (ค่า n), ตัวเรือตามกฎลอการิทึม; False = ผนังไม่ลื่นธรรมดา
    hull_ks: float = 0.002     # ความขรุขระผิวตัวเรือ (ม.)


# ================================================================ เรขาคณิต
def bed_level(yy, sec: Section):
    a = np.abs(yy)
    if sec.z > 0:
        side = np.minimum((a - sec.B / 2) / sec.z, sec.Hb)
    else:
        side = np.where(a > sec.B / 2, sec.Hb, 0.0)
    return np.where(a <= sec.B / 2, 0.0, np.where(a <= sec.Tb / 2, side, sec.Hb))


@dataclass
class Reach:
    sec: Section
    cfg: ReachConfig
    fleet: Fleet
    h: float
    Q: float
    S: float
    dx: float
    dt: float
    nx: int
    ny: int
    nz: int
    y: np.ndarray
    zc: np.ndarray
    solid: np.ndarray
    uin_phys: np.ndarray
    Fx: np.ndarray
    Fy: np.ndarray
    Fz: np.ndarray
    cf: np.ndarray = None                       # สัมประสิทธิ์แรงเสียดทานของเซลล์ติดผนัง (nz, ny, nx)
    bedwall: np.ndarray = None                  # เซลล์ติดท้องน้ำ/ตลิ่ง (ใช้ความเร็วเฉลี่ยตามความลึกเป็นความเร็วอ้างอิง)
    hulls: list = field(default_factory=list)   # (x0, x1, y0, y1, z0, z1)
    props: list = field(default_factory=list)   # dict(x, y, z, D, Deff, tilt, T)
    notes: list = field(default_factory=list)
    W: float = 0.0
    A: float = 0.0
    V: float = 0.0

    @property
    def vel(self):  # แปลงความเร็วแลตทิซ → ม./วิ
        return self.dx / self.dt

    @property
    def x(self):
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def cells(self):
        return self.nx * self.ny * self.nz

    def eta_from_rho(self, rho):
        return (rho - 1.0) / 3.0 * self.vel ** 2 / G


def auto_dx(L, Wd, h, cfg: ReachConfig):
    budget = (L * Wd * h / cfg.max_cells) ** (1 / 3)
    return float(min(max(budget, h / 40), max(h / 8, budget)))


def build_reach(sec: Section, h: float, Q: float, S: float, fleet: Fleet, cfg: ReachConfig) -> Reach:
    notes = []
    fp = min(sec.Bf, cfg.fp_max) if h > sec.Hb else 0.0
    Wtop = (sec.B + 2 * sec.z * min(h, sec.Hb)) + 2 * fp
    dx = cfg.dx if cfg.dx > 0 else auto_dx(cfg.length, Wtop, h, cfg)
    nz = max(int(round(h / dx)), 4)
    dx = h / nz                          # ให้ผิวน้ำตรงกับขอบเซลล์พอดี
    ny = int(math.ceil(Wtop / dx)) + 2
    nx = max(int(round(cfg.length / dx)), 16)
    y = (np.arange(ny) - ny / 2 + 0.5) * dx
    zc = (np.arange(nz) + 0.5) * dx
    zb = bed_level(y, sec)
    sol2 = (zc[:, None] < zb[None, :]) | (np.abs(y)[None, :] > Wtop / 2)
    if fp > 0 and sec.Bf > fp:
        notes.append(f"ที่ราบน้ำท่วมจำลองแค่ {fp:g} ม./ฝั่ง (ขอบโดเมนเป็นผนัง)")
    solid = np.broadcast_to(sol2[:, :, None], (nz, ny, nx)).copy()

    # ---- โปรไฟล์ความเร็วขาเข้า (กฎกำลัง 1/7 จากระยะห่างถึงท้องน้ำ/ตลิ่ง)
    fluid2 = ~sol2
    pad = np.concatenate([fluid2, fluid2[::-1]], axis=0)   # สะท้อนที่ผิวน้ำ (ไม่ใช่ผนัง)
    d = distance_transform_edt(pad)[:nz] * dx
    prof = np.where(fluid2, np.maximum(d, 0.5 * dx) ** (1 / 7), 0.0)
    if cfg.wall_model:
        # ความเร็วเฉลี่ยของแต่ละคอลัมน์ตามแมนนิ่ง U ∝ h^(2/3) (ตลิ่งตื้นไหลช้ากว่ากลางลำน้ำ) — ตรงกับสมดุลของ
        # wall model จึงไม่ต้องใช้ระยะทางยาวให้การไหลปรับตัวจากทางเข้า
        ncol = fluid2.sum(0)
        cmean = np.where(ncol > 0, prof.sum(0) / np.maximum(ncol, 1), 1.0)
        prof = prof / cmean[None, :] * (np.maximum(ncol, 1) * dx) ** (2 / 3)
        prof = np.where(fluid2, prof, 0.0)
    A = fluid2.sum() * dx * dx
    V = Q / A
    uin = prof * (V * fluid2.sum() / prof.sum())

    # ---- เรือและใบพัด
    Fx = np.zeros(solid.shape, np.float32)
    Fy = np.zeros(solid.shape, np.float32)
    Fz = np.zeros(solid.shape, np.float32)
    hulls, props = [], []
    b = fleet.boat
    Xc = (np.arange(nx) + 0.5) * dx
    if fleet.enabled and fleet.n_boats > 0:
        usable = (sec.B + 2 * sec.z * min(h, sec.Hb)) * 0.9
        if b.draft >= h * 0.9:
            notes.append(f"ระยะกินน้ำ {b.draft:g} ม. เกือบเท่าความลึก {h:g} ม. — ลดเหลือ {0.6*h:.2f} ม.")
        draft = min(b.draft, 0.6 * h)
        for r in range(fleet.rows):
            xb = fleet.x_first + r * fleet.row_spacing
            for k in range(fleet.per_row):
                yb = -usable / 2 + (k + 0.5) * usable / fleet.per_row
                x0, x1 = xb, xb + b.hull_L
                hulls.append((x0, x1, yb - b.hull_B / 2, yb + b.hull_B / 2, h - draft, h))
                ix = (Xc >= x0) & (Xc <= x1)
                frac = np.clip((Xc[ix] - x0) / (0.25 * b.hull_L), 0.35, 1.0)   # หัวเรือเรียว
                for jx, fr in zip(np.flatnonzero(ix), frac):
                    jy = np.abs(y - yb) <= b.hull_B / 2 * fr
                    jz = zc >= h - draft
                    solid[np.ix_(jz, jy, [jx])] = True
                D = b.outlet_D()
                yaws = (-b.spread_deg, 0.0, b.spread_deg) if b.spread_deg > 0 else (0.0,)
                for p in range(b.n_props):
                    yp = yb + (p - (b.n_props - 1) / 2) * b.prop_spacing
                    depth = b.prop_depth
                    if depth > h - 0.6 * D:
                        depth = max(h - 0.6 * D, 0.5 * D)
                        notes.append(f"ใบพัด/หัวฉีดลึกเกินความลึกน้ำ — ยกขึ้นเป็น {depth:.2f} ม.")
                    if b.prop_x < 0 and depth < draft + 0.5 * D:
                        depth = min(draft + 0.5 * D, h - 0.5 * D)
                    xp = x1 + b.prop_x
                    T = b.thrust()
                    for yaw in yaws:
                        pr = _add_prop(Fx, Fy, Fz, solid, Xc, y, zc, dx, xp, yp, h - depth, D,
                                       b.tilt_deg, T / len(yaws), yaw)
                        if b.is_jet:      # สายน้ำจากหัวฉีดมีความเร็วตามแรงดัน (ใช้เลือก Δt ให้ Mach ต่ำพอ)
                            pr["Vjet"] = min(b.jet_speed(), pr["Vjet"] * 4)
                        props.append(pr)
        notes = list(dict.fromkeys(notes))

    # ---- หน่วยเวลา: Mach ต่ำพอสำหรับสายน้ำจากใบพัด
    Vmax_in = float(uin.max())
    Ujet = 0.8 * max([p["Vjet"] for p in props], default=0.0) + V   # ความเร็วจริงต่ำกว่าทฤษฎีจากการผสม
    Umax = max(1.6 * Vmax_in, Ujet, 0.2)
    dt = cfg.u_lb * dx / Umax
    # ความเร็วเสียงเทียมของ LBM = dx/(Δt·√3): คลื่นความดันต้องสะท้อนไป-กลับในช่วงหลายรอบภายในช่วงหาค่าเฉลี่ย
    # ไม่เช่นนั้นระดับน้ำ/อัตราการไหลจะแกว่งเป็นคลื่นช้า ๆ ทำให้ผลเปรียบเทียบคลาดเคลื่อน (1 ซม. ≈ ρgA·0.01 N)
    t_win = cfg.run_flows * cfg.length / max(V, 0.05) * cfg.avg_frac
    dt = float(min(dt, t_win * dx / (N_SLOSH * 2 * math.sqrt(3) * nx * dx)))
    # แปลงแรงเป็นหน่วยแลตทิซ: f_lb = f [N/m³] · dt² / (ρ·dx)
    # ถ้าแรงต่อเซลล์ใหญ่เกินขีดเสถียรภาพ ให้ลด Δt (แรงแลตทิซ ∝ Δt²)
    F_LB_MAX = 1.5e-3
    fmax = float(np.sqrt(Fx ** 2 + Fy ** 2 + Fz ** 2).max()) if props else 0.0
    if fmax > 0 and fmax * dt * dt / (RHO * dx) > F_LB_MAX:
        dt = math.sqrt(F_LB_MAX * RHO * dx / fmax)
    k = dt * dt / (RHO * dx)
    cf, bedwall = wall_cf(solid, sol2, dx, sec.n, cfg) if cfg.wall_model else (None, None)
    return Reach(sec=sec, cfg=cfg, fleet=fleet, h=h, Q=Q, S=S, dx=dx, dt=dt, nx=nx, ny=ny, nz=nz, y=y, zc=zc,
                 solid=solid, uin_phys=uin, Fx=(Fx * k).astype(np.float32), Fy=(Fy * k).astype(np.float32),
                 Fz=(Fz * k).astype(np.float32), cf=cf, bedwall=bedwall, hulls=hulls, props=props, notes=notes, W=Wtop, A=A, V=V)


# D3Q19: เพื่อนบ้าน 18 ทิศ (ไม่รวมมุม 8 ทิศ)
_D3Q19 = np.ones((3, 3, 3), bool)
_D3Q19[::2, ::2, ::2] = False


def log_cf(z, ks):
    """สัมประสิทธิ์แรงเสียดทานผนังขรุขระ  τ = ρ·cf·u²  ที่ระยะ z จากผนัง (กฎลอการิทึม, z0 = ks/30)"""
    return (0.41 / math.log(max(30.0 * z / max(ks, 1e-6), 2.0))) ** 2


def manning_ks(n):
    """ความขรุขระเทียบเท่า (ม.) จากค่า n ของแมนนิ่ง (สูตร Strickler)"""
    return (21.1 * n) ** 6


def wall_cf(solid, bed2, dx, n, cfg: ReachConfig):
    """cf ต่อเซลล์ของเหลวที่ติดผนัง  τ = ρ·cf·U²

    ท้องน้ำ/ตลิ่ง: cf = g·n²/h^(1/3) ตามแมนนิ่ง (h = ความลึกของคอลัมน์น้ำนั้น) กับ U = ความเร็วเฉลี่ยตามความลึก
      → ในการไหลสม่ำเสมอให้แรงเสียดทานตรงกับแบบจำลอง 1D ไม่ขึ้นกับรูปโปรไฟล์ความเร็วของ LES บนกริดหยาบ
    ตัวเรือ: กฎลอการิทึมจาก hull_ks กับความเร็วขนานผิวของเซลล์เอง
    คืน (cf, bedwall) — bedwall = เซลล์ที่ใช้ความเร็วเฉลี่ยตามความลึก
    """
    nz, ny, nx = solid.shape
    bed = np.broadcast_to(bed2[:, :, None], solid.shape)
    hull = solid & ~bed

    def touching(m, floor):
        pad = np.pad(m, ((1, 1), (1, 1), (0, 0)), constant_values=False)
        pad[0] = floor                      # z = -1 (ใต้ท้องน้ำ)
        pad[:, 0] = pad[:, -1] = floor      # ขอบข้างโดเมน
        pad = np.pad(pad, ((0, 0), (0, 0), (1, 1)), mode="edge")
        return binary_dilation(pad, _D3Q19)[1:-1, 1:-1, 1:-1] & ~solid

    cf = np.zeros(solid.shape, np.float32)
    cf[touching(hull, False)] = log_cf(0.5 * dx, cfg.hull_ks)
    bedwall = touching(bed, True)
    hcol = np.maximum((~bed2).sum(0) * dx, dx)                          # ความลึกน้ำของแต่ละคอลัมน์ (ny,)
    cf_col = (G * n * n / hcol ** (1 / 3)).astype(np.float32)
    cf = np.where(bedwall, cf_col[None, :, None], cf)
    for a in (cf, bedwall):
        a[:, :, :2] = 0
        a[:, :, -2:] = 0                    # คอลัมน์ทางเข้า/ออกถูกกำหนดโดยเงื่อนไขขอบ — ใช้ bounce-back ธรรมดา
    return cf.astype(np.float32), bedwall


def _add_prop(Fx, Fy, Fz, solid, Xc, y, zc, dx, xp, yp, zp, D, tilt, T, yaw=0.0):
    """กระจายแรงขับใบพัดลงในเซลล์รูปจาน (ขยายให้กว้างอย่างน้อย 3 เซลล์ โดยคงแรงขับรวมเท่าเดิม)
    tilt = มุมกดลง, yaw = มุมเบนในแนวราบ (+ = ไปทาง y บวก)"""
    Deff = max(D, 3 * dx)
    t, w = math.radians(tilt), math.radians(yaw)
    ax, ay, az = math.cos(t) * math.cos(w), math.cos(t) * math.sin(w), -math.sin(t)
    R = Deff / 2 + 2 * dx
    ix = np.flatnonzero(np.abs(Xc - xp) <= R)
    iy = np.flatnonzero(np.abs(y - yp) <= R)
    iz = np.flatnonzero(np.abs(zc - zp) <= R)
    X, Yy, Z = np.meshgrid(Xc[ix] - xp, y[iy] - yp, zc[iz] - zp, indexing="ij")
    axial = X * ax + Yy * ay + Z * az
    rx, ry, rz = X - axial * ax, Yy - axial * ay, Z - axial * az
    rad = np.sqrt(rx * rx + ry * ry + rz * rz)
    m = (np.abs(axial) <= dx * 0.75) & (rad <= Deff / 2)
    m &= ~solid[np.ix_(iz, iy, ix)].transpose(2, 1, 0)
    n = int(m.sum())
    if n == 0:
        return dict(x=xp, y=yp, z=zp, D=D, Deff=Deff, tilt=tilt, yaw=yaw, T=0.0, Vjet=0.0, cells=0)
    fcell = T / (n * dx ** 3)              # N/m³
    sub = np.zeros(m.shape, np.float32)
    sub[m] = fcell
    sub = sub.transpose(2, 1, 0)           # (z, y, x)
    Fx[np.ix_(iz, iy, ix)] += sub * ax
    Fy[np.ix_(iz, iy, ix)] += sub * ay
    Fz[np.ix_(iz, iy, ix)] += sub * az
    Aeff = math.pi * Deff ** 2 / 4
    return dict(x=xp, y=yp, z=zp, D=D, Deff=Deff, tilt=tilt, yaw=yaw, T=T, cells=n,
                Vjet=2 * math.sqrt(T / (2 * RHO * Aeff)))


# ================================================================ การรัน
class Run3D:
    """ควบคุมการรัน LBM หนึ่งกรณี: เดินเวลา, หาค่าเฉลี่ยเวลา, อนุภาคติดตาม"""

    def __init__(self, reach: Reach, be):
        self.r, self.be = reach, be
        nu_lb = max(1e-6 * reach.dt / reach.dx ** 2, 2e-4)
        uin_lb = (reach.uin_phys / reach.vel).astype(np.float32)
        self.lbm = LBM3D(be, reach.solid, reach.Fx, reach.Fy, reach.Fz, uin_lb, tau0=0.5 + 3 * nu_lb, cs=reach.cfg.cs,
                         cf_wall=reach.cf)
        xp = self.lbm.xp
        self.xp = xp
        if reach.bedwall is not None:
            fl = ~reach.solid
            self._fl = xp.asarray(fl.astype(np.float32))
            self._ncol = xp.asarray(np.maximum(fl.sum(0), 1).astype(np.float32))
            self._bedwall = xp.asarray(reach.bedwall)
            self._update_uref()
        self.sum = {k: xp.zeros(reach.cells, np.float32) for k in ("ux", "uy", "uz", "rho", "sp", "uxx")}
        self.navg = 0
        self.t = 0.0
        self.t_total = reach.cfg.run_flows * reach.cfg.length / max(reach.V, 0.05)
        self.t_avg = self.t_total * (1 - reach.cfg.avg_frac)
        # เร่งแรงขับใบพัดขึ้นทีละน้อย เพื่อไม่ให้เกิดคลื่นความดันกระเพื่อมตอนเริ่ม
        self.t_ramp = min(0.15 * self.t_total, 20.0) if reach.props else 0.0
        self._F0 = (self.lbm.Fx.copy(), self.lbm.Fy.copy(), self.lbm.Fz.copy())
        self._ramp_done = self.t_ramp <= 0
        if not self._ramp_done:
            self.lbm.Fx *= 0
            self.lbm.Fy *= 0
            self.lbm.Fz *= 0
        self.wall0 = time.time()
        self.steps_wall = 0.0
        self.particles = None

    def _update_uref(self):
        """ความเร็วเฉลี่ยตามความลึกของแต่ละคอลัมน์ → ความเร็วอ้างอิงของ wall model ที่ท้องน้ำ/ตลิ่ง"""
        L, r, xp = self.lbm, self.r, self.xp
        ux = L.ux.reshape(r.nz, r.ny, r.nx)
        uy = L.uy.reshape(r.nz, r.ny, r.nx)
        U = xp.sqrt((ux * self._fl).sum(0) ** 2 + (uy * self._fl).sum(0) ** 2) / self._ncol
        L.uref = xp.where(self._bedwall, U[None], 0).astype(np.float32).ravel()

    @property
    def done(self):
        return self.t >= self.t_total

    def advance(self, nsteps: int):
        t0 = time.time()
        L = self.lbm
        avg_every = 20
        k = 0
        while k < nsteps:
            m = min(avg_every, nsteps - k)
            if not self._ramp_done:
                f = min(self.t / self.t_ramp, 1.0)
                f = np.float32(f * f * (3 - 2 * f))   # คงเป็น float32 — เคอร์เนล CUDA อ่านเป็น float*
                L.Fx = self._F0[0] * f
                L.Fy = self._F0[1] * f
                L.Fz = self._F0[2] * f
                self._ramp_done = f >= 1.0
            if self.r.bedwall is not None:
                self._update_uref()
            L.step(m)
            k += m
            self.t += m * self.r.dt
            if self.t >= self.t_avg:
                s = self.sum
                s["ux"] += L.ux
                s["uy"] += L.uy
                s["uz"] += L.uz
                s["rho"] += L.rho
                s["sp"] += self.xp.sqrt(L.ux * L.ux + L.uy * L.uy + L.uz * L.uz)
                s["uxx"] += L.ux * L.ux
                self.navg += 1
        self.be.sync()
        self.steps_wall += time.time() - t0

    def run_to_end(self, progress=None, cancel=None, chunk=400):
        duty = min(max(GPU_DUTY, 0.05), 1.0)
        if duty < 1:
            chunk = 100                     # ช่วงสั้น ๆ สลับพัก ให้ภาระ GPU เรียบขึ้น
        while not self.done:
            t0 = time.time()
            self.advance(chunk)
            if duty < 1:
                time.sleep((time.time() - t0) * (1 / duty - 1))
            if progress:
                progress(self.t / self.t_total)
            if cancel and cancel():
                break

    def mlups(self):
        return self.lbm.nstep * self.r.cells / max(self.steps_wall, 1e-9) / 1e6

    # ---- ฟิลด์ (หน่วยจริง) เป็น NumPy (nz, ny, nx)
    def field(self, name: str, mean: bool = False) -> np.ndarray:
        L, r = self.lbm, self.r
        if mean and self.navg > 0:
            a = self.be.to_np(self.sum[name]) / self.navg
        elif name == "sp":
            a = self.be.to_np(self.xp.sqrt(L.ux * L.ux + L.uy * L.uy + L.uz * L.uz))
        else:
            a = self.be.to_np(getattr(L, name))
        a = a.reshape(r.nz, r.ny, r.nx)
        if name == "rho":
            return r.eta_from_rho(a)          # ระดับน้ำเทียบเท่า (ม.)
        if name == "uxx":
            return a * r.vel ** 2
        return a * r.vel

    def slice_np(self, name, axis, idx, mean=False):
        """ตัดเฉพาะหน้าตัด (โอนข้อมูลจาก GPU น้อย): axis 0=z, 1=y, 2=x"""
        L, r = self.lbm, self.r
        src = (self.sum[name] / max(self.navg, 1)) if (mean and self.navg > 0) else None
        if src is None:
            src = self.xp.sqrt(L.ux * L.ux + L.uy * L.uy + L.uz * L.uz) if name == "sp" else getattr(L, name)
        a3 = src.reshape(r.nz, r.ny, r.nx)
        a = a3[idx] if axis == 0 else (a3[:, idx, :] if axis == 1 else a3[:, :, idx])
        a = self.be.to_np(a)
        return r.eta_from_rho(a) if name == "rho" else a * r.vel

    # ---- อนุภาคติดตามการไหล 3 มิติ
    def init_particles(self, n):
        r = self.r
        rng = np.random.default_rng(0)
        fluid = np.argwhere(~r.solid)
        pick = fluid[rng.integers(0, len(fluid), n)]
        self.P = (pick[:, ::-1] + rng.random((n, 3))) * r.dx   # (x, y_index, z) ในหน่วยเมตรจากมุมโดเมน
        self.particles = True

    def move_particles(self, dt_phys):
        if not self.particles:
            return
        r = self.r
        L = self.lbm
        P = self.P
        ix = np.clip((P[:, 0] / r.dx).astype(int), 0, r.nx - 1)
        iy = np.clip((P[:, 1] / r.dx).astype(int), 0, r.ny - 1)
        iz = np.clip((P[:, 2] / r.dx).astype(int), 0, r.nz - 1)
        cell = ix + r.nx * (iy + r.ny * iz)
        ci = self.xp.asarray(cell)
        u = np.stack([self.be.to_np(L.ux[ci]), self.be.to_np(L.uy[ci]), self.be.to_np(L.uz[ci])], 1) * r.vel
        P += np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0) * dt_phys
        fin = np.isfinite(P).all(1)
        Q = np.where(fin[:, None], P, -1.0)
        ix = (Q[:, 0] / r.dx).astype(int)
        iy = (Q[:, 1] / r.dx).astype(int)
        iz = (Q[:, 2] / r.dx).astype(int)
        out = ~fin | (ix < 0) | (ix >= r.nx - 1) | (iy < 0) | (iy >= r.ny) | (iz < 0) | (iz >= r.nz)
        ok = ~out
        bad = out.copy()
        bad[ok] = r.solid[iz[ok], iy[ok], ix[ok]]
        nb = int(bad.sum())
        if nb:
            rng = np.random.default_rng()
            inlet = np.argwhere(~r.solid[:, :, 1])
            pk = inlet[rng.integers(0, len(inlet), nb)]
            P[bad] = np.column_stack([rng.random(nb) * 2 * r.dx, (pk[:, 1] + rng.random(nb)) * r.dx,
                                      (pk[:, 0] + rng.random(nb)) * r.dx])

    def particles_xyz(self):
        """พิกัดจริง: x จากทางเข้า, y จากแนวกลางลำน้ำ, z จากท้องน้ำ"""
        r = self.r
        return self.P[:, 0], self.P[:, 1] + r.y[0] - 0.5 * r.dx, self.P[:, 2]


# ================================================================ สรุปผล
def profiles(run: Run3D) -> dict:
    """ค่าตามแนวลำน้ำจากสนามเฉลี่ยเวลา"""
    r = run.r
    ux = run.field("ux", mean=True)
    sp = run.field("sp", mean=True)
    eta = run.field("rho", mean=True)
    fl = ~r.solid
    top = fl[-1]                                                     # เซลล์ใต้ผิวน้ำ
    n_top = np.maximum(top.sum(0), 1)
    eta_x = (eta[-1] * top).sum(0) / n_top
    us_x = (ux[-1] * top).sum(0) / n_top
    umax_x = np.where(fl, sp, 0).max(axis=(0, 1))
    q_x = (ux * fl).sum(axis=(0, 1)) * r.dx * r.dx
    # ฟลักซ์โมเมนตัม + แรงดันต่อหน้าตัด (N): Σ (ρ·u² + ρ·g·η) dA  — ใช้หาแรงสุทธิที่เรือเติมให้น้ำ
    uxx = run.field("uxx", mean=True)
    mom_x = ((RHO * uxx + RHO * G * eta) * fl).sum(axis=(0, 1)) * r.dx * r.dx
    # แรงเฉือนท้องน้ำจากเซลล์ของเหลวชั้นแรกเหนือพื้น (กฎลอการิทึม, ks จาก Manning)
    below = np.concatenate([np.ones((1,) + fl.shape[1:], bool), ~fl[:-1]], 0)
    first = fl & below
    cf = log_cf(0.5 * r.dx, manning_ks(r.sec.n))
    tau = np.where(first, RHO * cf * sp ** 2, 0.0)
    tau_bed = tau.max(axis=0)                                         # (ny, nx)
    return dict(x=r.x, eta=eta_x - eta_x[-1], us=us_x, umax=umax_x, q=q_x, tau_bed=tau_bed,
                tau_max_x=tau_bed.max(0), mom=mom_x)


def compare(res: dict, base: dict, reach: Reach, cfg: ReachConfig) -> dict:
    """เปรียบเทียบกรณีมีเรือกับไม่มีเรือ"""
    x = res["x"]
    fl = reach.fleet
    T = fl.total_thrust()
    P = fl.total_power()
    xs = [p["x"] for p in reach.props]
    x_last = max(xs) if xs else 0.0
    i_up = max(int(0.08 * len(x)), 1)
    if reach.hulls:   # จุดวัดระดับน้ำด้านเหนือต้องอยู่หน้ากลุ่มเรือเสมอ (ช่วงจำลองยาว 8% อาจเลยหัวเรือไปแล้ว)
        i_up = max(min(i_up, int(np.searchsorted(x, min(hb[0] for hb in reach.hulls) - 10.0))), 1)
    d_eta = (res["eta"][i_up] - res["eta"][-1]) - (base["eta"][i_up] - base["eta"][-1])
    excess = res["umax"] - base["umax"]
    thr = max(0.2, 0.15 * reach.V)
    down = x > x_last
    hit = np.flatnonzero(down & (excess > thr))
    reach_m = float(x[hit[-1]] - x_last) if len(hit) else 0.0
    reach_capped = bool(len(hit) and hit[-1] >= len(x) - 3)
    m = down & (x < x[-1] - 2 * reach.dx)
    us_gain = float((res["us"][m] - base["us"][m]).mean()) if m.any() else 0.0
    a_cell = reach.dx * reach.dx
    # แรงเฉือนท้องน้ำ: พิจารณาเฉพาะช่วงตั้งแต่หน้ากลุ่มเรือถึงก่อนทางออก (ตัดผลของขอบทางเข้า/ออก)
    x0 = min([hb[0] for hb in reach.hulls], default=0.15 * x[-1]) - 5.0
    zone = (x >= max(x0, 0.1 * x[-1])) & (x <= x[-1] - 3 * reach.dx)
    tb, tbb = res["tau_bed"][:, zone], base["tau_bed"][:, zone]
    over = float((tb > cfg.tau_crit).sum() * a_cell)
    over_b = float((tbb > cfg.tau_crit).sum() * a_cell)
    # แรงสุทธิที่กลุ่มเรือเติมให้น้ำ = Δ(โมเมนตัม+แรงดัน) ท้ายช่วง − Δ ต้นช่วง (เทียบกรณีไม่มีเรือ)
    i_out = len(x) - 4
    dm = res["mom"] - base["mom"]
    F_net = float(dm[i_out] - dm[i_up])
    # ประมาณการเพิ่มอัตราการไหล หากวางกลุ่มเรือนี้ทุก ๆ spacing_km (สมดุลโมเมนตัม + แมนนิ่ง)
    Lsp = cfg.spacing_km * 1000
    gravity = RHO * G * reach.A * Lsp * max(reach.S, 1e-7)

    def dq(F):
        return math.sqrt(max(1 + F / gravity, 0.0)) - 1
    dQ = dq(T)
    dQn = dq(F_net)
    return dict(label=fl.label(), n=fl.n_boats, T_kN=T / 1000, P_kW=P, h=reach.h, Q=reach.Q, V=reach.V,
                d_eta_cm=100 * d_eta, jet_reach=reach_m, reach_capped=reach_capped,
                us_gain=us_gain, us_gain_pct=100 * us_gain / max(reach.V, 1e-6),
                umax=float(res["umax"].max()), tau_max=float(tb.max()), tau_base=float(tbb.max()), tau_area=over, tau_area_base=over_b,
                dQ_pct=100 * dQ, dQ_net_pct=100 * dQn, F_net_kN=F_net / 1000,
                dQ_per_100kW=100 * dQn / max(P / 100, 1e-9) if P > 0 else 0.0,
                drop_cm_per_100kW=-100 * d_eta / (P / 100) if P > 0 else 0.0,
                kW_per_cm=P / (-100 * d_eta) if (P > 0 and d_eta < -1e-5) else float("nan"),
                T_per_kW=T / max(P, 1e-9) if P > 0 else 0.0)


def summary_text(c: dict, cfg: ReachConfig) -> list[str]:
    out = [f"■ {c['label']} (ระดับน้ำ {c['h']:g} ม., Q {c['Q']:,.1f} ม³/วิ): แรงขับรวม {c['T_kN']:.1f} kN "
           f"จากกำลัง {c['P_kW']:.0f} kW ({c['T_per_kW']:.0f} N/kW); "
           f"แรงสุทธิที่น้ำได้รับ (หักแรงต้านตัวเรือและแรงเสียดทานท้องน้ำที่เพิ่มขึ้นในช่วงจำลอง) ≈ {c['F_net_kN']:.1f} kN"]
    kpc = f"{c['kW_per_cm']:.0f} kW ต่อการลด 1 ซม." if c["kW_per_cm"] == c["kW_per_cm"] else "ไม่ได้ลดระดับน้ำ"
    out.append(f"ระดับน้ำด้านเหนือกลุ่มเรือเทียบท้ายช่วง เปลี่ยนไป {c['d_eta_cm']:+.2f} ซม. "
               f"(ค่าลบ = เรือช่วยลดระดับน้ำด้านเหนือ) — ลดได้ {c['drop_cm_per_100kW']:.2f} ซม. ต่อ 100 kW ({kpc})")
    cap = " (ยาวเกินช่วงจำลอง)" if c["reach_capped"] else ""
    out.append(f"สายน้ำจากใบพัดเร็วกว่าปกติ >{max(0.2, 0.15*c['V']):.2f} ม./วิ ไปได้ไกล {c['jet_reach']:.0f} ม.{cap}; "
               f"ความเร็วผิวน้ำท้ายเรือเพิ่มเฉลี่ย {c['us_gain']:+.3f} ม./วิ ({c['us_gain_pct']:+.1f}%)")
    out.append(f"แรงเฉือนท้องน้ำสูงสุด {c['tau_max']:.1f} N/ม² (ไม่มีเรือ {c['tau_base']:.1f}); "
               f"พื้นที่เกิน {cfg.tau_crit:g} N/ม² = {c['tau_area']:.0f} ม² (ไม่มีเรือ {c['tau_area_base']:.0f} ม²)")
    out.append(f"ถ้าวางกลุ่มเรือแบบนี้ทุก {cfg.spacing_km:g} กม.: อัตราการไหลเพิ่ม ≈ {c['dQ_net_pct']:.2f}% "
               f"จากแรงสุทธิ (ขอบเขตบนจากแรงขับล้วน {c['dQ_pct']:.2f}%) = {c['dQ_per_100kW']:.3f}% ต่อ 100 kW")
    return out


def level_flow(river, sec: Section, h: float):
    """Q และความชันจากแบบจำลอง 1D ที่ระดับน้ำ h"""
    return float(h1.q_ref(h, river, sec)), float(river.mean_slope)


def config_dict(fleet: Fleet) -> dict:
    d = asdict(fleet)
    return d
