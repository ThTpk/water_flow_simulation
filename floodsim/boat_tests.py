"""การทดสอบเรือดันน้ำแนะนำ 5 แบบ + ตัวรันชุดทดสอบ (ใช้แม่น้ำ/หน้าตัด/ระดับน้ำที่ตั้งไว้ในโปรแกรม)"""
from __future__ import annotations

from dataclasses import replace

from .reach3d import Fleet, ReachConfig, Run3D, boat_type, build_reach, compare, profiles

# แต่ละรายการ: label, boat, rows, per_row, (ทางเลือก) row_spacing, level ('low'/'mid'/'high'), override
BOAT_TESTS = [
    dict(key="B1", title="1) ชนิดเรือ — แบบไหนดันน้ำได้ดีที่สุด",
         objective="เปรียบเทียบเรือ 4 ชนิด จำนวน 3 ลำเท่ากัน ที่ระดับน้ำเดียวกัน",
         observe=["แรงขับต่อกำลัง (N/kW): ใบพัดใหญ่หมุนช้าให้แรงขับต่อกำลังสูงกว่าใบพัดเล็กหมุนเร็ว",
                  "ระยะที่สายน้ำไปถึง: สายน้ำจากใบพัดเล็กเร็วมากแต่สลายตัวเร็ว",
                  "ระดับน้ำด้านเหนือเรือลดลงเท่าไร (ซม.)"],
         variants=[dict(label="เรือหางยาว ×3", boat="longtail", rows=1, per_row=3),
                   dict(label="เรือ outboard ×3", boat="outboard", rows=1, per_row=3),
                   dict(label="เรือลากจูง ×3", boat="tug", rows=1, per_row=3),
                   dict(label="ทุ่นผลักดันน้ำ ×3", boat="pontoon", rows=1, per_row=3)]),
    dict(key="B2", title="2) จำนวนเรือ — เพิ่มเรือแล้วได้ผลเพิ่มเท่าไร",
         objective="เพิ่มจำนวนเรือลากจูงจาก 1 → 6 ลำ ดูว่าผลเพิ่มตามจำนวนหรือไม่",
         observe=["การลดระดับน้ำเพิ่มตามแรงขับรวม (เชิงเส้นโดยประมาณ)",
                  "การวางเรือชิดกันทำให้สายน้ำรวมกันเป็นกระแสกว้าง ไปได้ไกลขึ้น"],
         variants=[dict(label="เรือลากจูง ×1", boat="tug", rows=1, per_row=1),
                   dict(label="เรือลากจูง ×2", boat="tug", rows=1, per_row=2),
                   dict(label="เรือลากจูง ×3", boat="tug", rows=1, per_row=3),
                   dict(label="เรือลากจูง 2 แถว×3", boat="tug", rows=2, per_row=3, row_spacing=40)]),
    dict(key="B3", title="3) ระดับน้ำ — เรือได้ผลต่างกันอย่างไรเมื่อน้ำต่ำ/สูง",
         objective="กลุ่มเรือหางยาว 4 ลำเดิม ที่ระดับน้ำต่ำ กลาง สูง",
         observe=["น้ำยิ่งลึก/ยิ่งไหลมาก ผลของเรือต่ออัตราการไหลยิ่งน้อยลง (สัดส่วนของโมเมนตัมเรือต่อโมเมนตัมแม่น้ำลดลง)",
                  "น้ำตื้น: สายน้ำจากใบพัดใกล้ท้องน้ำ แรงเฉือนท้องน้ำสูง เสี่ยงกัดเซาะ"],
         variants=[dict(label="น้ำต่ำ", boat="longtail", rows=1, per_row=4, level="low"),
                   dict(label="น้ำปานกลาง", boat="longtail", rows=1, per_row=4, level="mid"),
                   dict(label="น้ำสูง", boat="longtail", rows=1, per_row=4, level="high")]),
    dict(key="B4", title="4) มุมใบพัด — กดใบพัดลงมากน้อยแค่ไหนดี",
         objective="เรือหางยาว 4 ลำ เปลี่ยนมุมเพลาใบพัด 0°, 12°, 25°",
         observe=["มุมกดลงมาก: สายน้ำพุ่งลงท้องน้ำ แรงเฉือนท้องน้ำสูง (กัดเซาะ) แต่ความเร็วผิวน้ำเพิ่มน้อย",
                  "มุมแนวนอน: สายน้ำวิ่งใต้ผิวน้ำได้ไกล"],
         variants=[dict(label="มุม 0°", boat="longtail", rows=1, per_row=4, over=dict(tilt_deg=0)),
                   dict(label="มุม 12°", boat="longtail", rows=1, per_row=4, over=dict(tilt_deg=12)),
                   dict(label="มุม 25°", boat="longtail", rows=1, per_row=4, over=dict(tilt_deg=25))]),
    dict(key="B5", title="5) การจัดแถวเรือ — เรียงขวางหรือเรียงตามลำน้ำ",
         objective="เรือหางยาว 6 ลำเท่ากัน จัดเป็น 1×6, 2×3, 3×2 (ห่างแถวละ 25 ม.)",
         observe=["เรียงขวางลำน้ำ: กระจายแรงทั่วหน้าตัด ความเร็วผิวน้ำเพิ่มสม่ำเสมอ",
                  "เรียงตามกัน: เรือแถวหลังดูดน้ำที่เร็วอยู่แล้ว สายน้ำรวมกันไปไกลแต่แคบ"],
         variants=[dict(label="1 แถว × 6", boat="longtail", rows=1, per_row=6),
                   dict(label="2 แถว × 3", boat="longtail", rows=2, per_row=3, row_spacing=25),
                   dict(label="3 แถว × 2", boat="longtail", rows=3, per_row=2, row_spacing=25)]),
    dict(key="B6", title="6) เครื่องฉีดน้ำ/ม่านน้ำ เทียบกับใบพัด ที่กำลังเท่ากัน",
         objective="ทุ่นลอย 3 ชุดเรียงขวางลำน้ำ กำลังรวม 300 kW เท่ากันทุกกรณี: ใบพัดใหญ่, หัวฉีดแรงดันสูง 100 บาร์ "
                   "(ตรง/ฉีดกระจาย ±30°), หัวฉีด 10 บาร์, ม่านน้ำความเร็วต่ำ 3 ม./วิ",
         observe=["สิ่งที่ดันน้ำคือโมเมนตัม (มวล×ความเร็ว) ไม่ใช่แรงดัน: แรงขับต่อกำลัง = 2η/v — ยิ่งฉีดเร็วยิ่งได้แรงน้อย",
                  "ฉีดกระจาย: แรงด้านข้างหักล้างกัน แรงดันไปข้างหน้าลดลง",
                  "สายน้ำแรงดันสูงสลายตัวในระยะสั้น ไม่เกิด 'กำแพงน้ำ' ที่เคลื่อนที่ไปไกล"],
         variants=[dict(label="ใบพัดใหญ่ 100 kW ×3", boat="pontoon", rows=1, per_row=3, over=dict(power_kW=100)),
                   dict(label="ฉีด 100 บาร์ ตรง ×3", boat="jet_hp", rows=1, per_row=3),
                   dict(label="ฉีด 100 บาร์ กระจาย ±30° ×3", boat="jet_hp", rows=1, per_row=3,
                        over=dict(spread_deg=30)),
                   dict(label="ฉีด 10 บาร์ ×3", boat="jet_mp", rows=1, per_row=3),
                   dict(label="ม่านน้ำ 3 ม./วิ ×3", boat="curtain", rows=1, per_row=3)]),
]


def pick_level(levels, which, Hb):
    lv = sorted(levels)
    inb = [h for h in lv if h <= Hb] or lv
    if which == "low":
        return lv[0]
    if which == "high":
        return lv[-1]
    return inb[len(inb) // 2] if which == "mid" else which


def run_test(test, river, sec, levels, h_sel, cfg: ReachConfig, be, flow_fn, x_first=40.0,
             progress=None, cancel=None, on_result=None, boat_edits=None):
    """รันทุกกรณีของการทดสอบ (พร้อมกรณีไม่มีเรือที่ระดับน้ำเดียวกัน) คืนรายการผลเปรียบเทียบ"""
    base_cache = {}
    results = []
    n = len(test["variants"])
    for k, v in enumerate(test["variants"]):
        if cancel and cancel():
            break
        h = pick_level(levels, v["level"], sec.Hb) if "level" in v else h_sel
        Q, S = flow_fn(h)
        b = boat_type(v["boat"])
        if boat_edits and v["boat"] in boat_edits:
            b = replace(b, **boat_edits[v["boat"]])
        if "over" in v:
            b = replace(b, **v["over"])
        fleet = Fleet(b, rows=v["rows"], per_row=v["per_row"], row_spacing=v.get("row_spacing", 30.0),
                      x_first=x_first)

        def prog(f, k=k, part=0):
            if progress:
                progress((k + 0.5 * part + 0.5 * f) / n, v["label"])

        if h not in base_cache:
            rb = build_reach(sec, h, Q, S, replace(fleet, enabled=False), cfg)
            run_b = Run3D(rb, be)
            run_b.run_to_end(progress=lambda f: prog(f, part=0), cancel=cancel)
            base_cache[h] = profiles(run_b)
        rr = build_reach(sec, h, Q, S, fleet, cfg)
        run = Run3D(rr, be)
        run.run_to_end(progress=lambda f: prog(f, part=1), cancel=cancel)
        res = profiles(run)
        c = compare(res, base_cache[h], rr, cfg)
        c["label"] = v["label"]
        c["profiles"] = res
        c["base_profiles"] = base_cache[h]
        c["notes"] = rr.notes
        results.append(c)
        if on_result:
            on_result(c, run)
    return results
