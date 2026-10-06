"""ส่งออกผลการจำลอง 3 มิติเป็นภาพเคลื่อนไหว 3D บนเว็บ (ไฟล์ HTML ไฟล์เดียว, three.js)

  python -m floodsim.web3d --boat-test 6 --cases 1,5 --out results/web3d/test6.html
  python -m floodsim.web3d --cases 1:1,1:3,6:5 --out results/web3d/all.html   (การทดสอบ:กรณี จากหลายการทดสอบ)
  ตัวเลือก --gpu-duty 0.5 = ให้ GPU ทำงานครึ่งเวลา (เบาเครื่อง/ร้อนน้อยลง แต่ช้าลง 2 เท่า)

ผลแต่ละกรณีเก็บใน results/web3d/cache — รันซ้ำจะใช้ผลเดิมไม่ต้องจำลองใหม่ (--import นำไฟล์ HTML เดิมเข้า cache)

บันทึกสนามความเร็วเฉลี่ยเวลา (ย่อ 2 เท่าในแนวราบ, เก็บเป็น int8), แรงเฉือนท้องน้ำ, ท้องน้ำ, ตัวเรือ และใบพัด
เบราว์เซอร์คำนวณการเคลื่อนที่ของอนุภาคเองจากสนามนี้ (ไม่ต้องใช้ GPU ตอนเปิดดู)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from dataclasses import replace

import numpy as np

from . import reach3d
from .reach3d import Fleet, Run3D, boat_type, build_reach, compare, profiles

STRIDE = 2   # ย่อสนามความเร็วในแนว x, y (ขนาดไฟล์ ~1.5 MB ต่อกรณีที่ 2 ล้านเซลล์)


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _pool(a, fl, s):
    """ค่าเฉลี่ยของเซลล์ของเหลวในบล็อก s×s แนวราบ: a, fl (nz, ny, nx) → (nz, ny/s, nx/s)"""
    nz, ny, nx = a.shape
    ny2, nx2 = ny // s, nx // s
    a = (a * fl)[:, :ny2 * s, :nx2 * s].reshape(nz, ny2, s, nx2, s).sum((2, 4))
    n = fl[:, :ny2 * s, :nx2 * s].reshape(nz, ny2, s, nx2, s).sum((2, 4))
    return np.where(n > 0, a / np.maximum(n, 1), 0.0), n > 0


def export_case(run: Run3D, label: str, c: dict | None, res: dict) -> dict:
    """เก็บข้อมูลของกรณีหนึ่งสำหรับตัวแสดงผลบนเว็บ (เรียกหลัง run_to_end)"""
    r = run.r
    fl = (~r.solid).astype(np.float32)
    comps, fluid = [], None
    for k in ("ux", "uy", "uz"):
        a, fluid = _pool(run.field(k, mean=True), fl, STRIDE)
        comps.append(a.astype(np.float32))
    umax = max(float(np.abs(a).max()) for a in comps) or 1.0
    scale = umax / 126
    q = np.stack([np.clip(np.round(a / scale), -126, 126) for a in comps], -1).astype(np.int8)  # (nz, ny2, nx2, 3)
    q[~fluid] = -128                                       # เซลล์ของแข็ง
    sp = np.sqrt(sum(a * a for a in comps))
    nz, ny2, nx2 = fluid.shape
    dxs = r.dx * STRIDE
    # ท้องน้ำ: ความสูงของเซลล์ของเหลวล่างสุดในแต่ละคอลัมน์ (หน้าตัดที่ทางเข้า ไม่มีตัวเรือ)
    col = ~r.solid[:, :, 0]
    zb = np.where(col.any(0), col.argmax(0) * r.dx, r.h)
    zb2 = zb[:ny2 * STRIDE].reshape(ny2, STRIDE).min(1)
    tau = res["tau_bed"][:ny2 * STRIDE, :nx2 * STRIDE].reshape(ny2, STRIDE, nx2, STRIDE).max((1, 3))
    wet = tau > 0
    tau_top = float(max(np.percentile(tau[wet], 99.5) if wet.any() else 1.0, 1.0))
    tau_med = float(np.median(tau[wet])) if wet.any() else 1.0
    tq = np.clip(np.round(tau * 100), 0, 65535).astype(np.uint16)          # 0.01 N/m²
    step = max(len(res["x"]) // 300, 1)
    out = dict(
        label=label, nx=int(nx2), ny=int(ny2), nz=int(nz), dxs=float(dxs), dz=float(r.dx),
        y0=float(r.y[0] - 0.5 * r.dx), L=float(nx2 * dxs), W=float(ny2 * dxs), h=float(r.h),
        V=float(r.V), Q=float(r.Q), S=float(r.S), scale=float(scale), u=_b64(q),
        sp99=float(np.percentile(sp[fluid], 99.7)),
        tau=_b64(tq), tau_top=tau_top, tau_med=tau_med, zb=[round(float(v), 3) for v in zb2],
        hulls=[[round(float(v), 3) for v in hb] for hb in r.hulls],
        props=[dict(x=p["x"], y=p["y"], z=p["z"], D=p["D"], tilt=p["tilt"], yaw=p.get("yaw", 0.0)) for p in r.props],
        boat=r.fleet.boat.name if r.fleet.enabled else "",
        jet=bool(r.fleet.boat.is_jet) if r.fleet.enabled else False,
        prof=dict(x=[round(float(v), 2) for v in res["x"][::step]],
                  eta=[round(float(v) * 100, 3) for v in res["eta"][::step]],
                  umax=[round(float(v), 3) for v in res["umax"][::step]]),
    )
    if c:
        out["stats"] = {k: (None if (isinstance(v, float) and v != v) else round(float(v), 4))
                        for k, v in c.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return out


EMBED_MAX = 12_000_000   # หน้าที่ใหญ่กว่านี้เก็บสนามความเร็วแยกไฟล์ (ขีดจำกัด 16 MB ต่อไฟล์ของ Artifact)


def write_html(cases: list[dict], path: str, title: str, subtitle: str):
    """เขียนหน้า HTML; ถ้าข้อมูลใหญ่ เก็บสนามความเร็วของแต่ละกรณีเป็น <ชื่อ>_data/caseN.json (คืนรายชื่อไฟล์ทั้งหมด)"""
    tpl = open(os.path.join(os.path.dirname(__file__), "web3d_template.html"), encoding="utf-8").read()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    files = [path]
    if sum(len(c["u"]) + len(c["tau"]) for c in cases) > EMBED_MAX:
        stem = os.path.splitext(os.path.basename(path))[0] + "_data"
        ddir = os.path.join(os.path.dirname(os.path.abspath(path)), stem)
        os.makedirs(ddir, exist_ok=True)
        out = []
        for k, c in enumerate(cases):
            fn = os.path.join(ddir, f"case{k}.json")
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(dict(u=c["u"], tau=c["tau"]), f)
            files.append(fn)
            out.append({**{kk: v for kk, v in c.items() if kk not in ("u", "tau")}, "src": f"{stem}/case{k}.json"})
        cases = out
    data = json.dumps(dict(title=title, subtitle=subtitle, cases=cases), ensure_ascii=False, separators=(",", ":"))
    html = tpl.replace("__TITLE__", title).replace("__DATA__", data.replace("</", "<\\/"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return files


def read_html_cases(path: str) -> list[dict]:
    s = open(path, encoding="utf-8").read()
    d = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', s, re.S).group(1))
    for c in d["cases"]:
        if "src" in c:
            c.update(json.load(open(os.path.join(os.path.dirname(path), c.pop("src")), encoding="utf-8")))
    return d["cases"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="floodsim.web3d", description="ส่งออกภาพเคลื่อนไหว 3D บนเว็บ")
    ap.add_argument("--boat-test", type=int, default=6, help="การทดสอบเริ่มต้นของกรณีที่ไม่ระบุ 'การทดสอบ:'")
    ap.add_argument("--cases", default="1,5", help="กรณีที่จะแสดง เช่น 1,5 หรือ 1:3,6:5 (การทดสอบ:กรณี)")
    ap.add_argument("--no-base", action="store_true", help="ไม่รวมกรณีไม่มีเรือ")
    ap.add_argument("--import", dest="imports", nargs="*", default=[], help="นำผลจากไฟล์ HTML เดิมเข้า cache")
    ap.add_argument("--exp", type=int, default=3)
    ap.add_argument("--level", type=float, default=None)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--length", type=float, default=None, help="ความยาวช่วงจำลอง (ม.)")
    ap.add_argument("--x-first", type=float, default=None, help="ตำแหน่งหัวเรือแถวแรกจากทางเข้า (ม.)")
    ap.add_argument("--section", default=None,
                    help="หน้าตัดเอง: B,Hb,z,n = กว้างท้องคลอง, สูงตลิ่ง, ลาดตลิ่ง (นอน:ตั้ง), แมนนิ่ง (ไม่มีที่ราบน้ำท่วม)")
    ap.add_argument("--slope", type=float, default=None, help="ความลาดผิวน้ำ (ม./กม.) ลำน้ำตรง")
    ap.add_argument("--name", default=None, help="ชื่อลำน้ำที่แสดงในหน้าเว็บ")
    ap.add_argument("--gpu-duty", type=float, default=None, help="สัดส่วนเวลาที่ให้ GPU ทำงาน 0.1–1 (ค่าน้อย = เบาเครื่องแต่ช้าลง)")
    ap.add_argument("--backend", default="auto", choices=["auto", "cupy", "numpy"])
    ap.add_argument("--out", default="results/web3d/flow3d.html")
    ap.add_argument("--title", default=None, help="ชื่อหน้า (ค่าเริ่มต้น: ชื่อการทดสอบ)")
    a = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    if a.gpu_duty:
        reach3d.GPU_DUTY = a.gpu_duty

    from . import hydraulics1d as h1
    from .boat_tests import BOAT_TESTS
    from .gui import App

    app = App(a.backend)
    app.withdraw()
    app.load_experiment(a.exp - 1, quiet=True)
    if a.level:
        app.sel_level.set(f"{a.level:g}")
    if a.max_cells:
        app.cvars["max_cells"].set(a.max_cells)
    if a.length:
        app.cvars["length"].set(a.length)
    if a.x_first is not None:
        app.fvars["x_first"].set(a.x_first)
    if a.section or a.slope:
        if a.section:
            B, Hb, z, n = (float(v) for v in a.section.split(","))
            for k, v in dict(B=B, Hb=Hb, z=z, n=n, Bf=0.0).items():
                app.svars[k].set(v)
        if a.slope:   # ลำน้ำตรง: ความคดเคี้ยว 1, ความชัน = ค่าที่กำหนด
            for k, v in dict(sigma=1.0, irregularity=0.0, Sv_m_per_km=a.slope).items():
                app.rvars[k].set(v)
        app.rebuild()
        if a.level:
            app.sel_level.set(f"{a.level:g}")
    app.update()
    cfg = app.read_reach_cfg()
    h = app._sel_h()
    river, sec = app.river, app.sec
    Q, S = float(h1.q_ref(h, river, sec)), float(river.mean_slope)
    x_first = float(app.fvars["x_first"].get())
    t0 = time.time()

    # ---- cache: หนึ่งไฟล์ต่อกรณี (ผูกกับแม่น้ำ/ระดับน้ำ/จำนวนเซลล์/wall model)
    cdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "cache")
    os.makedirs(cdir, exist_ok=True)
    # รูปทรงหน้าตัด (กว้าง/ลึกตลิ่ง/ลาดตลิ่ง/ที่ราบ/n) อยู่ในชื่อด้วย — เปลี่ยนความกว้างแล้วต้องจำลองใหม่ ไม่ใช้ผลเดิม
    shape = f"B{sec.B:g}_Hb{sec.Hb:g}_z{sec.z:g}_Bf{sec.Bf:g}_n{sec.n:g}" + ("" if not a.slope else f"_S{a.slope:g}")
    if a.x_first is not None:
        shape += f"_x{a.x_first:g}"
    tag = f"e{a.exp}_{shape}_h{h:g}_n{cfg.max_cells}_L{cfg.length:g}{'' if cfg.wall_model else '_nowm'}"

    def cpath(key):
        return os.path.join(cdir, f"{key}_{tag}.json")

    def save(key, c):
        with open(cpath(key), "w", encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False)

    def load(key):
        p = cpath(key)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    def key_of_label(label):
        if label == "ไม่มีเรือ":
            return "base"
        for ti, t in enumerate(BOAT_TESTS, 1):
            for ci, v in enumerate(t["variants"], 1):
                if v["label"] == label:
                    return f"t{ti}c{ci}"
        return None

    for p in a.imports:
        for c in read_html_cases(p):
            k = key_of_label(c["label"])
            if k and not os.path.exists(cpath(k)):
                save(k, c)
                print(f"นำเข้า {c['label']} → cache ({k})")

    specs = []
    for item in a.cases.split(","):
        ti, ci = (int(v) for v in item.split(":")) if ":" in item else (a.boat_test, int(item))
        specs.append((ti, ci, BOAT_TESTS[ti - 1]["variants"][ci - 1]))

    base_c, base_prof = load("base"), None
    bnp = os.path.join(cdir, f"base_profiles_{tag}.npz")

    def go(fleet):
        run = Run3D(build_reach(sec, h, Q, S, fleet, cfg), app.be)
        run.run_to_end()
        return run, profiles(run)

    def need_base():
        nonlocal base_c, base_prof
        if base_prof is None and os.path.exists(bnp):
            base_prof = dict(np.load(bnp))
        if base_prof is None or base_c is None:
            run_b, base_prof = go(replace(Fleet(boat_type("pontoon"), x_first=x_first), enabled=False))
            np.savez_compressed(bnp, **base_prof)
            base_c = export_case(run_b, "ไม่มีเรือ", None, base_prof)
            save("base", base_c)
            del run_b
            print(f"ไม่มีเรือ เสร็จ [{time.time()-t0:.0f}s]", flush=True)
        return base_prof

    out = []
    for ti, ci, v in specs:
        key = f"t{ti}c{ci}"
        c = load(key)
        if c is None:
            b = boat_type(v["boat"])
            if "over" in v:
                b = replace(b, **v["over"])
            fleet = Fleet(b, rows=v["rows"], per_row=v["per_row"], row_spacing=v.get("row_spacing", 30.0),
                          x_first=x_first)
            run, res = go(fleet)
            cmp = compare(res, need_base(), run.r, cfg)
            c = export_case(run, v["label"], cmp, res)
            save(key, c)
            del run
            print(f"{v['label']} เสร็จ: Δระดับน้ำ {cmp['d_eta_cm']:+.2f} ซม. [{time.time()-t0:.0f}s]", flush=True)
        else:
            print(f"{v['label']}: ใช้ผลใน cache")
        out.append(c)
    if not a.no_base:
        if base_c is None:
            need_base()
        out.append(base_c)
    V = out[0]["V"]
    sub = (f"{a.name or river.name} · กว้าง {sec.B + 2 * sec.z * min(h, sec.Hb):g} ม. · ความลึก {h:g} ม. · Q {Q:,.0f} ม³/วิ · ความเร็วปกติ {V:.2f} ม./วิ · "
           f"ช่วงลำน้ำ {cfg.length:g} ม.")
    title = a.title or BOAT_TESTS[specs[0][0] - 1]["title"].split(") ", 1)[-1]
    files = write_html(out, a.out, title, sub)
    size = sum(os.path.getsize(f) for f in files)
    print(f"saved {os.path.abspath(a.out)} ({len(files)} ไฟล์, {size/1e6:.1f} MB, {time.time()-t0:.0f}s)")
    app.destroy()


if __name__ == "__main__":
    sys.exit(main())
