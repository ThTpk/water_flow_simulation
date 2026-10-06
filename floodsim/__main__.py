"""จุดเริ่มโปรแกรม

  python -m floodsim                                เปิดหน้าจอโปรแกรม
  python -m floodsim --backend numpy                บังคับใช้ CPU (ช้ามาก)
  python -m floodsim --boat-test 1 --out results    รันการทดสอบเรือแนะนำที่ 1 แบบอัตโนมัติ แล้วบันทึกรูป/CSV
      ตัวเลือก: --exp 1-5 (หน้าตัดจากการทดลองรูปทรงแม่น้ำ), --level ความลึก, --file ไฟล์แม่น้ำ,
               --max-cells จำนวนเซลล์สูงสุด, --length ความยาวช่วง (ม.), --no-wall-model,
               --gpu-duty 0.5 (ให้ GPU ทำงานครึ่งเวลา: เบาเครื่องแต่ช้าลง)
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def run_boat_batch(a):
    from . import hydraulics1d as h1
    from .boat_tests import BOAT_TESTS, run_test
    from .gui import App, _speed_norm, draw_results_fig, draw_slices_fig, draw_view3d_ax, write_results_csv
    from .io_formats import read_river_file
    from .reach3d import summary_text

    os.makedirs(a.out, exist_ok=True)
    app = App(a.backend)
    app.withdraw()
    app.load_experiment(a.exp - 1, quiet=True)
    if a.file:
        app.trace = read_river_file(a.file)
        app.rebuild()
    if a.level:
        app.sel_level.set(f"{a.level:g}")
    if a.max_cells:
        app.cvars["max_cells"].set(a.max_cells)
    if a.length:
        app.cvars["length"].set(a.length)
    if a.no_wall_model:
        app.wall_model.set(False)
    app.update()
    cfg = app.read_reach_cfg()
    test = BOAT_TESTS[a.boat_test - 1]
    h = app._sel_h()
    river, sec = app.river, app.sec
    print(f"{test['title']} | {river.name} | h={h:g} m | backend={app.be.name} ({app.be.device_name})")
    t0 = time.time()

    def save_case_figs(run, tag):
        """ภาพการไหล (ระนาบ/แนวตัด/หน้าตัด) และภาพ 3 มิติ ของกรณีหนึ่ง"""
        r = run.r
        if r.props:
            app.long_y.set(round(r.props[0]["y"], 2))
            app.cross_x.set(round(min(max(p["x"] for p in r.props) + 20, r.cfg.length - 2), 1))
            app.plan_depth.set(round(r.h - r.props[0]["z"], 2))
        run.init_particles(3000)
        run.move_particles(5.0)
        draw_slices_fig(app, run, True)
        app.figS.set_size_inches(14, 10)
        app.figS.savefig(os.path.join(a.out, f"slices3d_{tag}.png"), dpi=110)
        draw_view3d_ax(app.ax3, run, _speed_norm(r))
        app.ax3.view_init(28, -62)
        app.fig3.set_size_inches(14, 8)
        app.fig3.savefig(os.path.join(a.out, f"view3d_{tag}.png"), dpi=110)

    def on_result(c, run):
        app.results.append(c)
        print("\n".join(summary_text(c, cfg)), f"  [{time.time()-t0:.0f}s]", flush=True)
        save_case_figs(run, f"case{len(app.results)}")

    run_test(test, river, sec, app.levels, h, cfg, app.be,
             lambda hh: (float(h1.q_ref(hh, river, sec)), float(river.mean_slope)),
             x_first=float(app.fvars["x_first"].get()), on_result=on_result, boat_edits=app.boat_edits)
    draw_results_fig(app.figR, app.results)
    app.figR.set_size_inches(14, 8)
    app.figR.savefig(os.path.join(a.out, "boat_compare.png"), dpi=110)
    write_results_csv(os.path.join(a.out, "boat_results.csv"), app.results)
    with open(os.path.join(a.out, "cases.txt"), "w", encoding="utf-8") as f:
        for i, c in enumerate(app.results, 1):
            f.write(f"case{i}: {c['label']}\n")
    print(f"saved to {os.path.abspath(a.out)}  (total {time.time()-t0:.0f}s)")
    app.destroy()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="floodsim", description="แบบจำลองแม่น้ำ + การไหล 3 มิติสำหรับทดสอบเรือดันน้ำ")
    ap.add_argument("--backend", default="auto", choices=["auto", "cupy", "numpy"])
    ap.add_argument("--boat-test", type=int, default=None, help="รันการทดสอบเรือแนะนำที่ 1–6 แบบอัตโนมัติ")
    ap.add_argument("--exp", type=int, default=3, help="ใช้หน้าตัดจากการทดลองรูปทรงแม่น้ำที่ 1–5")
    ap.add_argument("--level", type=float, default=None, help="ระดับน้ำ (ความลึก ม.)")
    ap.add_argument("--file", default=None, help="ไฟล์แม่น้ำ (KML/KMZ/GPX/GeoJSON/CSV)")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--length", type=float, default=None)
    ap.add_argument("--no-wall-model", action="store_true", help="ปิด wall model (ผนังไม่ลื่นแบบเดิม)")
    ap.add_argument("--gpu-duty", type=float, default=None,
                    help="สัดส่วนเวลาที่ให้ GPU ทำงาน 0.1–1 (เช่น 0.5 = เบาเครื่อง/ร้อนน้อยลง แต่ช้าลง 2 เท่า)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    if a.gpu_duty:
        from . import reach3d
        reach3d.GPU_DUTY = a.gpu_duty
    if a.boat_test:
        run_boat_batch(a)
    else:
        from .gui import main as gui_main
        gui_main(a.backend)


if __name__ == "__main__":
    sys.exit(main())
