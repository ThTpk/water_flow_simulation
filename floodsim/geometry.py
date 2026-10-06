"""รูปทรงแม่น้ำ: สร้างแม่น้ำจำลอง (sine-generated curve) และประมวลผลเส้นแม่น้ำที่นำเข้าจากไฟล์"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# จุดอ้างอิงพิกัดสำหรับแม่น้ำจำลอง (ใช้ตอนส่งออก KML) — ที่ราบภาคกลาง
DEFAULT_LAT0 = 14.35
DEFAULT_LON0 = 100.55
EARTH_R = 6371008.8


@dataclass
class RiverParams:
    D_km: float = 10.0          # ระยะทางเส้นตรงจากต้นน้ำถึงปลายน้ำ (ตามแนวหุบเขา)
    sigma: float = 1.6          # ความคดเคี้ยว = ความยาวลำน้ำ / ระยะทางเส้นตรง
    Sv_m_per_km: float = 0.5    # ความชันของหุบเขา (ม./กม.)
    wavelength_B: float = 12.0  # ความยาวคลื่นของโค้งน้ำ (เท่าของความกว้างลำน้ำ)
    irregularity: float = 0.25  # ความไม่สม่ำเสมอของโค้ง 0..1
    seed: int = 7
    z_down: float = 2.0         # ระดับท้องน้ำที่ปลายน้ำ (ม.)


@dataclass
class Section:
    """หน้าตัดลำน้ำแบบผสม: ร่องน้ำสี่เหลี่ยมคางหมู + ที่ราบน้ำท่วมสองฝั่ง"""
    B: float = 40.0     # ความกว้างท้องน้ำ (ม.)
    Hb: float = 3.0     # ความสูงตลิ่ง (ม.)
    z: float = 2.0      # ความลาดตลิ่ง (แนวนอน:แนวตั้ง)
    Bf: float = 200.0   # ความกว้างที่ราบน้ำท่วมแต่ละฝั่ง (ม.)
    n: float = 0.032    # สัมประสิทธิ์แมนนิ่งของร่องน้ำ
    nf: float = 0.06    # สัมประสิทธิ์แมนนิ่งของที่ราบน้ำท่วม

    @property
    def Tb(self) -> float:
        """ความกว้างที่ระดับตลิ่ง"""
        return self.B + 2 * self.z * self.Hb


@dataclass
class River:
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    zb: np.ndarray        # ระดับท้องน้ำตามแนวร่องน้ำ
    theta: np.ndarray     # ทิศทางการไหล (เรเดียน)
    kappa: np.ndarray     # ความโค้ง 1/R (บวก = เลี้ยวซ้าย)
    sig_loc: np.ndarray   # ความคดเคี้ยวเฉพาะที่
    slope: np.ndarray     # ความชันท้องน้ำเฉพาะที่
    ds: float
    L: float
    D: float
    sigma: float
    name: str = "แม่น้ำจำลอง"
    source: str = "generated"
    lat0: float = DEFAULT_LAT0
    lon0: float = DEFAULT_LON0
    has_elev: bool = False
    x_off: float = 0.0    # ค่าชดเชยพิกัด (ม.) สำหรับแปลงกลับเป็นละติจูด/ลองจิจูด
    y_off: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def nx(self):
        return -np.sin(self.theta)

    @property
    def ny(self):
        return np.cos(self.theta)

    @property
    def mean_slope(self) -> float:
        return max((self.zb[0] - self.zb[-1]) / self.L, 1e-6)

    @property
    def N(self) -> int:
        return len(self.s)


# ---------------------------------------------------------------- utilities
def moving_average(a: np.ndarray, w: int) -> np.ndarray:
    """ค่าเฉลี่ยเคลื่อนที่แบบสมมาตร ครึ่งหน้าต่าง w จุด (ขอบหดหน้าต่างทั้งสองข้างเท่ากัน
    จึงไม่ทำให้แนวโน้มเชิงเส้น เช่น ความชันท้องน้ำ ผิดเพี้ยนที่ปลายเส้น)"""
    a = np.asarray(a, float)
    if w < 1 or len(a) < 3:
        return a.copy()
    n = len(a)
    c = np.concatenate([[0.0], np.cumsum(a)])
    i = np.arange(n)
    k = np.minimum(np.minimum(i, n - 1 - i), w)
    return (c[i + k + 1] - c[i - k]) / (2 * k + 1)


def resample_polyline(x, y, z, ds):
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    n = max(int(round(L / ds)), 2) + 1
    si = np.linspace(0, L, n)
    return np.interp(si, s, x), np.interp(si, s, y), np.interp(si, s, z), si


def finish_geometry(x, y, zb, s, sec: Section, smooth_m: float) -> dict:
    """คำนวณทิศทาง ความโค้ง ความคดเคี้ยวเฉพาะที่ และความชัน จากเส้นที่สุ่มตัวอย่างสม่ำเสมอแล้ว"""
    ds = s[1] - s[0]
    w = max(1, int(round(smooth_m / ds)))
    xs, ys = moving_average(x, w), moving_average(y, w)
    th = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    th = moving_average(th, max(1, w // 2))
    kappa = np.gradient(th, ds)

    n = len(s)
    wl = int(np.clip(round(15 * sec.B / ds), 2, max(2, n // 4)))
    i = np.arange(n)
    lo, hi = np.clip(i - wl, 0, n - 1), np.clip(i + wl, 0, n - 1)
    chord = np.hypot(x[hi] - x[lo], y[hi] - y[lo])
    sig_loc = np.where(chord > 0, (s[hi] - s[lo]) / np.maximum(chord, 1e-9), 1.0)
    if n > 2 * wl + 1:  # ช่วงปลายเส้นที่หน้าต่างไม่ครบ ใช้ค่าจากหน้าต่างเต็มที่ใกล้ที่สุด
        sig_loc[:wl] = sig_loc[wl]
        sig_loc[n - wl:] = sig_loc[n - wl - 1]
    sig_loc = np.clip(moving_average(sig_loc, wl), 1.0, 10.0)

    wz = max(1, int(round(max(20 * sec.B, s[-1] / 60) / ds)))
    zs = moving_average(zb, wz)
    slope = np.clip(-np.gradient(zs, ds), 1e-5, None)
    return dict(theta=th, kappa=kappa, sig_loc=sig_loc, slope=slope)


# ---------------------------------------------------------------- generated
def _curve(L, ds_target, lam_arc, omega, om_mult, lam_mult):
    n = max(int(round(L / ds_target)), 2) + 1
    s = np.linspace(0, L, n)
    ds = s[1] - s[0]
    half = 0.5 * lam_arc * lam_mult                  # ความยาวครึ่งคลื่นแต่ละช่วง
    c = np.concatenate([[0.0], np.cumsum(half)])
    k = np.clip(np.searchsorted(c, s, side="right") - 1, 0, len(half) - 1)
    phase = np.pi * k + np.pi * (s - c[k]) / half[k]
    th = omega * om_mult[k] * np.sin(phase)
    thm = 0.5 * (th[1:] + th[:-1])
    x = np.concatenate([[0.0], np.cumsum(ds * np.cos(thm))])
    y = np.concatenate([[0.0], np.cumsum(ds * np.sin(thm))])
    return s, x, y


def generate_river(rp: RiverParams, sec: Section) -> River:
    D = rp.D_km * 1000.0
    sigma = max(1.0, rp.sigma)
    L = D * sigma
    lam_arc = max(rp.wavelength_B, 2.0) * sec.B * sigma
    ds = min(lam_arc / 40.0, L / 800.0)
    ds = max(ds, L / 20000.0)
    rng = np.random.default_rng(rp.seed)
    nh = int(2 * L / lam_arc) + 20
    irr = float(np.clip(rp.irregularity, 0, 1))
    om_mult = 1 + irr * 0.45 * (2 * rng.random(nh) - 1)
    lam_mult = 1 + irr * 0.5 * (2 * rng.random(nh) - 1)

    if sigma <= 1.0005:
        omega = 0.0
    else:
        lo, hi = 0.0, 2.25
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            s, x, y = _curve(L, ds, lam_arc, mid, om_mult, lam_mult)
            if np.hypot(x[-1], y[-1]) > D:
                lo = mid
            else:
                hi = mid
        omega = 0.5 * (lo + hi)
    s, x, y = _curve(L, ds, lam_arc, omega, om_mult, lam_mult)

    # หมุนให้แนวต้นน้ำ->ปลายน้ำอยู่ตามแกน +x
    ang = np.arctan2(y[-1], x[-1])
    ca, sa = np.cos(-ang), np.sin(-ang)
    x, y = x * ca - y * sa, x * sa + y * ca
    chord = np.hypot(x[-1], y[-1])
    sig_act = s[-1] / chord
    drop = rp.Sv_m_per_km / 1000.0 * chord
    zb = rp.z_down + drop * (1 - s / s[-1])

    g = finish_geometry(x, y, zb, s, sec, smooth_m=0)
    notes = []
    if abs(sig_act - sigma) > 0.05 and sigma > 1.0:
        notes.append(f"ความคดเคี้ยวที่ได้จริง {sig_act:.2f} (ขอ {sigma:.2f})")
    return River(x=x, y=y, s=s, zb=zb, ds=s[1] - s[0], L=s[-1], D=chord, sigma=sig_act,
                 name=f"แม่น้ำจำลอง σ={sig_act:.2f}", notes=notes, **g)


# ---------------------------------------------------------------- imported
def build_imported(trace: dict, sec: Section, rp: RiverParams, use_elev=True,
                   reverse=False) -> River:
    """trace: dict จาก io_formats.read_river_file (x, y เป็นเมตร, z อาจเป็น nan)"""
    x, y, z = (np.asarray(trace[k], float).copy() for k in ("x", "y", "z"))
    if reverse:
        x, y, z = x[::-1], y[::-1], z[::-1]
    keep = np.concatenate([[True], np.hypot(np.diff(x), np.diff(y)) > 0.01])
    x, y, z = x[keep], y[keep], z[keep]
    if len(x) < 2:
        raise ValueError("เส้นแม่น้ำมีจุดน้อยเกินไป")
    notes = []

    zv = z[np.isfinite(z)]
    elev_ok = len(zv) > 0.5 * len(z) and (zv.max() - zv.min()) > 0.5 and np.count_nonzero(zv) > 0.5 * len(z)
    if elev_ok:
        z = np.interp(np.arange(len(z)), np.flatnonzero(np.isfinite(z)), z[np.isfinite(z)])

    seg = np.hypot(np.diff(x), np.diff(y))
    L = seg.sum()
    ds = max(L / 6000.0, min(sec.B, L / 800.0))
    xr, yr, zr, s = resample_polyline(x, y, z if elev_ok else np.zeros_like(x), ds)
    chord = np.hypot(xr[-1] - xr[0], yr[-1] - yr[0])
    sigma = s[-1] / max(chord, 1e-6)

    if elev_ok and use_elev:
        if zr[0] < zr[-1]:  # น้ำไหลจากที่สูงลงที่ต่ำ: กลับทิศเส้นให้อัตโนมัติ
            xr, yr, zr = xr[::-1].copy(), yr[::-1].copy(), zr[::-1].copy()
            notes.append("กลับทิศเส้นอัตโนมัติตามค่าระดับ (ต้นน้ำ = ปลายที่สูงกว่า)")
        zb = np.minimum.accumulate(zr)  # ท้องน้ำต้องไม่สูงขึ้นตามทิศการไหล
        zb = moving_average(zb, max(1, int(round(max(10 * sec.B, s[-1] / 100) / ds))))
        has_elev = True
        notes.append(f"ใช้ระดับจากไฟล์: {zr.max():.1f} → {zr.min():.1f} ม.")
    else:
        drop = rp.Sv_m_per_km / 1000.0 * chord
        zb = rp.z_down + drop * (1 - s / s[-1])
        has_elev = False
        notes.append("ไม่ใช้ข้อมูลความสูงจากไฟล์ — คำนวณจากความชันหุบเขาที่ตั้งไว้")

    g = finish_geometry(xr, yr, zb, s, sec, smooth_m=max(2 * sec.B, 2 * ds))
    return River(x=xr - xr[0], y=yr - yr[0], s=s, zb=zb, ds=s[1] - s[0], L=s[-1], D=chord,
                 sigma=sigma, name=trace.get("name") or "แม่น้ำนำเข้า", source="imported",
                 lat0=trace["lat0"], lon0=trace["lon0"], has_elev=has_elev, notes=notes,
                 x_off=float(xr[0]), y_off=float(yr[0]), **g)

