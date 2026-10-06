"""ตัวควบคุมการจำลอง 2 มิติ: ตั้งค่าเริ่มต้น, รันจนเข้าสู่สภาวะคงตัว, คลื่นน้ำหลาก, และสรุปผล"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import hydraulics1d as h1
from .backend import Backend
from .geometry import River, Section
from .solver2d import Solver2D
from .terrain import Terrain, build_terrain


@dataclass
class Sim2DConfig:
    cells_per_B: float = 6.0
    max_cells: int = 2_000_000
    theta: float = 1.0
    fr_max: float = 1.2


def make_solver(river: River, sec: Section, be: Backend, cfg: Sim2DConfig, hmax: float):
    ter = build_terrain(river, sec, cfg.cells_per_B, cfg.max_cells, hmax)
    n_eff = sec.n  # ใช้ n เดียวกับกริด 2D (โค้งน้ำในกริดสร้างความต้านทานเอง)
    S_out = float(max(river.slope[-1], 1e-6))

    def rating(depth):
        return float(h1.conveyance(depth, sec, n_eff)) * np.sqrt(S_out)

    return ter, Solver2D(ter, be, theta=cfg.theta, fr_max=cfg.fr_max, outlet_rating=rating,
                         outlet_bed=float(river.zb[-1]))


def initial_depth(ter: Terrain, river: River, sec: Section, h0: float):
    """สภาพเริ่มต้นจากผล 1 มิติที่ระดับอ้างอิง h0: คืนค่า (ความลึก, qx กลางเซลล์, qy กลางเซลล์)
    การใส่ความเร็วเริ่มต้นตามแนวลำน้ำช่วยลดช่วงปรับตัว (transient) ตอนเริ่มจำลอง"""
    prof = h1.node_profile(h0, river, sec)
    hn = prof["h"][ter.idx]
    eta = ter.zc + hn
    h = np.clip(eta - ter.z, 0, None)
    h[ter.wall] = 0
    V = np.where(ter.channel, prof["Vm"][ter.idx], prof["Vf"][ter.idx])
    V = V * (np.clip(h / np.maximum(hn, 1e-6), 0, 3)) ** (2 / 3)
    q = V * h
    th = river.theta[ter.idx]
    return h.astype(np.float32), (q * np.cos(th)).astype(np.float32), (q * np.sin(th)).astype(np.float32)


def hydrograph(kind: str, Q0: float, Qp: float, Tr: float):
    """ฟังก์ชันอัตราการไหลเข้า Q(t): 'ramp' = ยกระดับแล้วคงที่, 'pulse' = ขึ้นแล้วลง (ยอดที่ t=Tr)"""
    if kind == "ramp":
        def f(t):
            x = min(max(t / Tr, 0.0), 1.0)
            return Q0 + (Qp - Q0) * x * x * (3 - 2 * x)
    else:
        def f(t):
            if t >= 2 * Tr:
                return Q0
            return Q0 + (Qp - Q0) * np.sin(np.pi * t / (2 * Tr)) ** 2
    return f


def centerline_sample(ter: Terrain, arr: np.ndarray, step=1):
    return arr[ter.river_iy[::step], ter.river_ix[::step]]


def metrics(sol: Solver2D, river: River, sec: Section, Q: float) -> dict:
    ter = sol.ter
    h, sp, fr = sol.fields_np()
    qmag = sp * h
    wet = h > 0.02
    ch = ter.channel & wet
    fp = wet & ~ter.channel & ~ter.wall
    a = ter.dx * ter.dx
    kap = river.kappa[ter.idx]
    curved = ch & (np.abs(kap) * sec.Tb > 0.08) & (ter.dist > 0.15 * sec.Tb / 2)
    outer = curved & (ter.lat * kap < 0)
    inner = curved & (ter.lat * kap > 0)
    sp_c = centerline_sample(ter, sp)
    tt = float(np.sum(river.ds / np.maximum(sp_c, 0.02)))
    return dict(
        h0=None, Q=Q,
        V_ch=float(qmag[ch].sum() / max(h[ch].sum(), 1e-9)) if ch.any() else 0.0,
        V_max=float(np.percentile(sp[h > 0.1], 99.5)) if (h > 0.1).any() else 0.0,
        V_fp=float(sp[fp].mean()) if fp.any() else 0.0,
        h_ch=float(h[ch].mean()) if ch.any() else 0.0,
        h_max=float(h.max()),
        Fr_max=float(np.percentile(fr[ch], 99)) if ch.any() else 0.0,
        wet_km2=float(wet.sum() * a / 1e6),
        fp_km2=float(fp.sum() * a / 1e6),
        outer_inner=float(sp[outer].mean() / max(sp[inner].mean(), 1e-6)) if outer.any() and inner.any() else float("nan"),
        tt=tt,
        t_sim=sol.t,
    )


def run_steady(sol: Solver2D, river: River, sec: Section, h0: float, tmax_factor=3.0,
               wall_limit=600.0, progress=None, init=True) -> dict:
    """รันที่อัตราการไหลคงที่จนน้ำเข้า≈น้ำออก (หรือครบเวลาสูงสุด) แล้วสรุปผล"""
    Q = float(h1.q_ref(h0, river, sec))
    if init:
        sol.set_depth(*initial_depth(sol.ter, river, sec, h0))
    tt1d = h1.summarize(h0, river, sec)["tt"]
    tmax = tmax_factor * tt1d
    chunk = max(tt1d / 40, 20 * sol.dt)
    t_start = time.time()
    ok = 0
    sol.flush_outflow()
    while sol.t < tmax:
        t0 = sol.t
        sol.advance(chunk, lambda t: Q)
        Qout = sol.flush_outflow() / max(sol.t - t0, 1e-9)
        err = abs(Qout - Q) / Q
        ok = ok + 1 if err < 0.02 else 0
        if progress:
            progress(min(sol.t / tmax, 1.0), Qout)
        if ok >= 3 and sol.t > 0.3 * tt1d:
            break
        if time.time() - t_start > wall_limit:
            break
    m = metrics(sol, river, sec, Q)
    m["h0"] = h0
    m["Qout"] = Qout
    m["steady_err"] = err
    m["wall_s"] = time.time() - t_start
    return m


def run_wave(sol: Solver2D, river: River, sec: Section, h_base: float, h_peak: float, Tr: float,
             kind="pulse", wall_limit=900.0, progress=None, record_every=None):
    """จำลองคลื่นน้ำหลาก: เริ่มจากน้ำฐาน แล้วป้อนไฮโดรกราฟ คืนอนุกรมเวลาที่จุดวัด"""
    Q0 = float(h1.q_ref(h_base, river, sec))
    Qp = float(h1.q_ref(h_peak, river, sec))
    f = hydrograph(kind, Q0, Qp, Tr)
    sol.set_depth(*initial_depth(sol.ter, river, sec, h_base))
    tt = h1.summarize(h_peak, river, sec)["tt"]
    t_end = 2 * Tr + 2.0 * tt if kind == "pulse" else Tr + 2.0 * tt
    rec_dt = record_every or t_end / 300
    gauges = gauge_cells(sol.ter, river)
    rec = dict(t=[], Qin=[], Qout=[], h=[[] for _ in gauges])
    t_start = time.time()
    sol.flush_outflow()
    while sol.t < t_end:
        t0 = sol.t
        sol.advance(rec_dt, f)
        rec["t"].append(sol.t)
        rec["Qin"].append(f(sol.t))
        rec["Qout"].append(sol.flush_outflow() / max(sol.t - t0, 1e-9))
        hh = sol.be.to_np(sol.h)
        for k, (iy, ix) in enumerate(gauges):
            rec["h"][k].append(float(hh[iy, ix]))
        if progress:
            progress(sol.t / t_end, rec["Qout"][-1])
        if time.time() - t_start > wall_limit:
            break
    out = {k: np.asarray(v) for k, v in rec.items() if k != "h"}
    out["h"] = [np.asarray(v) for v in rec["h"]]
    out.update(wave_stats(out))
    out["Q0"], out["Qp"] = Q0, Qp
    return out


def gauge_cells(ter: Terrain, river: River, fracs=(0.05, 0.5, 0.95)):
    res = []
    for f in fracs:
        i = int(f * (river.N - 1))
        res.append((int(ter.river_iy[i]), int(ter.river_ix[i])))
    return res


def wave_stats(rec: dict) -> dict:
    t, qi, qo = rec["t"], rec["Qin"], rec["Qout"]
    if len(t) < 3:
        return {}
    i_in, i_out = int(np.argmax(qi)), int(np.argmax(qo))
    return dict(Qin_peak=float(qi[i_in]), Qout_peak=float(qo[i_out]),
                lag=float(t[i_out] - t[i_in]),
                atten=float(1 - qo[i_out] / max(qi[i_in], 1e-9)))
