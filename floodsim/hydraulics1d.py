"""ไฮดรอลิก 1 มิติ: สมการแมนนิ่งบนหน้าตัดผสม (ร่องน้ำ + ที่ราบน้ำท่วม)

ระดับน้ำอ้างอิง h0 = ความลึกปกติ (normal depth) เฉลี่ยของลำน้ำ
  -> อัตราการไหล Q = K(h0)·√S  โดย K = (1/n)·A·R^(2/3) แยกคำนวณร่องน้ำ/ที่ราบ (divided channel)
  -> แต่ละจุดตามลำน้ำแก้หาความลึกปกติของ Q เดียวกันตามความชันและความคดเคี้ยวเฉพาะที่
"""
from __future__ import annotations

import numpy as np

from .geometry import River, Section

G = 9.81
RHO = 1000.0
NU = 1.0e-6


def meander_mult(sig):
    """ตัวคูณความขรุขระจากความคดเคี้ยว (Cowan 1956, ค่า m5): 1.0 / 1.15 / 1.30"""
    sig = np.asarray(sig, float)
    return np.where(sig <= 1.1, 1.0,
                    np.where(sig <= 1.5, 1.0 + 0.15 * (sig - 1.1) / 0.4,
                             np.minimum(1.30, 1.15 + 0.15 * (sig - 1.5) / 0.5)))


def xs_props(h, sec: Section) -> dict:
    """คุณสมบัติหน้าตัดที่ความลึก h (นับจากท้องน้ำ) — รองรับอาเรย์"""
    h = np.maximum(np.asarray(h, float), 0.0)
    B, Hb, z, Bf = sec.B, sec.Hb, sec.z, sec.Bf
    sq = np.sqrt(1 + z * z)
    Tb = sec.Tb
    inb = h <= Hb
    he = h - Hb
    Am = np.where(inb, (B + z * h) * h, (B + z * Hb) * Hb + Tb * he)
    Pm = np.where(inb, B + 2 * h * sq, B + 2 * Hb * sq + (0.0 if Bf > 0 else 2 * np.maximum(he, 0)))
    Tm = np.where(inb, B + 2 * z * h, Tb)
    if Bf > 0:
        Af = np.where(inb, 0.0, 2 * Bf * np.maximum(he, 0))
        Pf = np.where(inb, 0.0, 2 * Bf + 2 * np.maximum(he, 0))
        T = np.where(inb, Tm, Tb + 2 * Bf)
    else:
        Af = np.zeros_like(h)
        Pf = np.zeros_like(h)
        T = Tm
    Rm = Am / np.maximum(Pm, 1e-9)
    Rf = np.where(Pf > 0, Af / np.maximum(Pf, 1e-9), 0.0)
    return dict(A=Am + Af, Am=Am, Af=Af, Pm=Pm, Pf=Pf, Rm=Rm, Rf=Rf, T=T, Tm=Tm,
                Kmu=Am * Rm ** (2 / 3), Kfu=Af * Rf ** (2 / 3))


def conveyance(h, sec: Section, nm):
    p = xs_props(h, sec)
    return p["Kmu"] / nm + p["Kfu"] / sec.nf


def normal_depth(Q, S, nm, sec: Section, iters=60):
    """แก้หาความลึกปกติแบบเวกเตอร์ (bisection) สำหรับทุกจุดพร้อมกัน"""
    S = np.asarray(S, float)
    nm = np.broadcast_to(np.asarray(nm, float), S.shape)
    Kreq = Q / np.sqrt(S)
    lo = np.zeros_like(S)
    hi = np.full_like(S, max(1.0, 2 * sec.Hb))
    for _ in range(60):
        small = conveyance(hi, sec, nm) < Kreq
        if not small.any():
            break
        hi = np.where(small, hi * 2, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = conveyance(mid, sec, nm) < Kreq
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return 0.5 * (lo + hi)


def node_state(h, Q, S, nm, kappa, sec: Section) -> dict:
    """สภาพการไหลที่จุดต่าง ๆ เมื่อรู้ความลึก h และอัตราการไหล Q"""
    p = xs_props(h, sec)
    Km = p["Kmu"] / nm
    Kf = p["Kfu"] / sec.nf
    K = np.maximum(Km + Kf, 1e-12)
    Sf = (Q / K) ** 2
    Qm = Q * Km / K
    Qf = Q - Qm
    Vm = Qm / np.maximum(p["Am"], 1e-9)
    Vf = np.where(p["Af"] > 0, Qf / np.maximum(p["Af"], 1e-9), 0.0)
    V = Q / np.maximum(p["A"], 1e-9)
    Fr = Vm / np.sqrt(G * p["Am"] / np.maximum(p["Tm"], 1e-9))
    tau = RHO * G * p["Rm"] * Sf
    radius = 1.0 / np.maximum(np.abs(kappa), 1e-9)
    radius = np.maximum(radius, p["Tm"] / 2)
    superelev = Vm ** 2 * p["Tm"] / (G * radius)
    return dict(h=np.broadcast_to(h, np.shape(Vm)), V=V, Vm=Vm, Vf=Vf, Fr=Fr, tau=tau,
                T=p["T"], Tm=p["Tm"], A=p["A"], Sf=Sf, radius=radius, superelev=superelev,
                power=tau * Vm, Re=Vm * p["Rm"] / NU, over=h > sec.Hb)


def reach_n(river: River, sec: Section):
    return sec.n * meander_mult(river.sig_loc)


def q_ref(h0, river: River, sec: Section):
    """อัตราการไหลที่ทำให้ความลึกปกติเฉลี่ยของลำน้ำ = h0"""
    nG = sec.n * float(meander_mult(river.sigma))
    return conveyance(h0, sec, nG) * np.sqrt(river.mean_slope)


def _sub(river: River, max_nodes=400):
    step = max(1, river.N // max_nodes)
    return np.arange(0, river.N, step)


def summarize(h0, river: River, sec: Section, idx=None) -> dict:
    """สรุปผลการไหลทั้งลำน้ำที่ระดับน้ำอ้างอิง h0"""
    if idx is None:
        idx = _sub(river)
    Q = float(q_ref(h0, river, sec))
    nm = reach_n(river, sec)[idx]
    S = river.slope[idx]
    h = normal_depth(Q, S, nm, sec)
    st = node_state(h, Q, S, nm, river.kappa[idx], sec)
    seglen = river.L / len(idx)
    tt = float(np.sum(seglen / np.maximum(st["V"], 1e-6)))
    return dict(h0=float(h0), Q=Q, V=float(st["V"].mean()), Vmax=float(st["Vm"].max()),
                Vm=float(st["Vm"].mean()), Vf=float(st["Vf"].mean()), Fr=float(st["Fr"].mean()),
                Frmax=float(st["Fr"].max()), tau=float(st["tau"].mean()), taumax=float(st["tau"].max()),
                tt=tt, T=float(st["T"].mean()), over=float(st["over"].mean()),
                power=float(st["power"].mean()), superelev=float(st["superelev"].max()),
                Re=float(st["Re"].mean()), hmin=float(h.min()), hmax=float(h.max()))


def node_profile(h0, river: River, sec: Section) -> dict:
    """ผลรายจุดตลอดลำน้ำ (ใช้วาดรูปตัดตามยาว)"""
    Q = float(q_ref(h0, river, sec))
    nm = reach_n(river, sec)
    h = normal_depth(Q, river.slope, nm, sec, iters=40)
    st = node_state(h, Q, river.slope, nm, river.kappa, sec)
    st["Q"] = Q
    st["nm"] = nm
    return st


def rating_curves(river: River, sec: Section, hmax: float, n=36) -> dict:
    hs = np.linspace(max(0.05, hmax / 60), hmax, n)
    idx = _sub(river, 200)
    rows = [summarize(h, river, sec, idx) for h in hs]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def regime_text(r: dict, sec: Section) -> str:
    parts = ["ไหลเร็ว (Fr>1)" if r["Frmax"] > 1 else "ไหลช้า (Fr<1)"]
    if r["h0"] > sec.Hb:
        parts.append("ล้นตลิ่ง")
    elif r["h0"] > 0.85 * sec.Hb:
        parts.append("ใกล้เต็มตลิ่ง")
    else:
        parts.append("ในตลิ่ง")
    return " · ".join(parts)


def insights(rows: list[dict], sec: Section, river: River) -> list[str]:
    """สร้างข้อสังเกตอัตโนมัติจากผลเปรียบเทียบระดับน้ำ"""
    if len(rows) < 2:
        return ["ใส่ระดับน้ำอย่างน้อย 2 ค่าเพื่อเปรียบเทียบ"]
    r0, r1 = rows[0], rows[-1]
    out = []
    k = r1["h0"] / r0["h0"]
    out.append(f"ระดับน้ำเพิ่มจาก {r0['h0']:.2f} → {r1['h0']:.2f} ม. (×{k:.1f}) : อัตราการไหลเพิ่ม ×{r1['Q']/max(r0['Q'],1e-9):.1f}, "
               f"ความเร็วเฉลี่ยเพิ่ม ×{r1['V']/max(r0['V'],1e-9):.2f} — Q โตเร็วกว่าความลึก เพราะทั้งพื้นที่หน้าตัดและรัศมีชลศาสตร์เพิ่มพร้อมกัน")
    out.append(f"เวลาที่น้ำไหลผ่านลำน้ำยาว {river.L/1000:.1f} กม.: {fmt_time(r0['tt'])} ที่ระดับต่ำสุด เทียบกับ {fmt_time(r1['tt'])} ที่ระดับสูงสุด")
    inb = [r for r in rows if r["h0"] <= sec.Hb]
    ovb = [r for r in rows if r["h0"] > sec.Hb]
    if inb and ovb:
        a, b = inb[-1], ovb[0]
        out.append(f"เมื่อน้ำล้นตลิ่ง (>{sec.Hb:.1f} ม.) ความเร็วเฉลี่ยทั้งหน้าตัดเปลี่ยนจาก {a['V']:.2f} เป็น {b['V']:.2f} ม./วิ "
                   f"เพราะน้ำแผ่บนที่ราบซึ่งตื้นและขรุขระกว่า (ความกว้างผิวน้ำ {a['T']:.0f} → {b['T']:.0f} ม.) — ที่ราบทำหน้าที่ 'เก็บกักน้ำ' ชะลอการไหล")
    sup = [r for r in rows if r["Frmax"] > 1]
    if sup:
        lv = ", ".join("%.2f" % r["h0"] for r in sup)
        out.append(f"ที่ระดับ {lv} ม. มีช่วงที่ Froude > 1 (ไหลเชี่ยว) — คลื่นไม่สามารถย้อนขึ้นต้นน้ำได้ และเสี่ยงกัดเซาะสูง")
    out.append(f"แรงเฉือนท้องน้ำเฉลี่ย {r0['tau']:.1f} → {r1['tau']:.1f} N/ม² — กรวดขนาด ~{r1['tau']/0.8:.0f} มม. เริ่มเคลื่อนที่ได้ที่ระดับสูงสุด (เกณฑ์ Shields โดยประมาณ τ≈0.8·d[มม.])")
    if river.sigma > 1.2:
        out.append(f"ความคดเคี้ยว {river.sigma:.2f} ทำให้ลำน้ำยาวกว่าระยะทางตรง ×{river.sigma:.2f} ความชันท้องน้ำจึงลดลงเหลือ 1/{river.sigma:.2f} ของความชันหุบเขา "
                   f"และเพิ่มความต้านทานโค้ง (n×{float(meander_mult(river.sigma)):.2f}) → น้ำไหลช้าลงและใช้เวลานานขึ้น; "
                   f"ผิวน้ำด้านนอกโค้งยกสูงได้ถึง {r1['superelev']*100:.0f} ซม.")
    return out


def fmt_time(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f} วินาที"
    if sec < 5400:
        return f"{sec/60:.0f} นาที"
    if sec < 172800:
        return f"{sec/3600:.1f} ชม."
    return f"{sec/86400:.1f} วัน"
