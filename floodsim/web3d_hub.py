"""รวมหน้า 3D หลายหน้า (คนละลำน้ำ) เป็นหน้าหลักหน้าเดียว — เลือกลำน้ำได้ที่แผงควบคุม พร้อมเมนูลิงก์ไปหน้าอื่น

  python -m floodsim.web3d_hub --out results/web3d/all_boats.html ^
      --scene river=results/web3d/all_boats.html "แม่น้ำจำลอง" "9 กรณี" ^
      --scene prawet=results/web3d/prawet_canal.html "คลองประเวศ" "500 ม."

สนามความเร็วของทุกกรณีเก็บแยกไฟล์ <ชื่อ>_data/<ลำน้ำ>/caseN.json (ไม่ต้องจำลองใหม่ อ่านจากหน้าเดิม)
ลิงก์เมนูอ่านจาก --links (JSON: [{"label", "note", "href"}, ...])
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .web3d import read_html_cases


def page_meta(path: str) -> dict:
    import re
    s = open(path, encoding="utf-8").read()
    d = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', s, re.S).group(1))
    if "scenes" in d:   # หน้าหลักเดิม: ใช้ลำน้ำแรก
        raise SystemExit(f"{path} เป็นหน้าหลักอยู่แล้ว — ใช้หน้าของแต่ละลำน้ำแทน")
    return dict(title=d["title"], subtitle=d["subtitle"])


def build(scenes: list[tuple[str, str, str, str]], out: str, title: str, links: list[dict]) -> list[str]:
    """scenes = [(key, html, label, note)] → เขียนหน้าหลัก คืนรายชื่อไฟล์ทั้งหมด (สำหรับเผยแพร่)"""
    tpl = open(os.path.join(os.path.dirname(__file__), "web3d_template.html"), encoding="utf-8").read()
    outdir = os.path.dirname(os.path.abspath(out))
    stem = os.path.splitext(os.path.basename(out))[0] + "_data"
    files, data = [out], []
    for key, html, label, note in scenes:
        meta, cases = page_meta(html), read_html_cases(html)
        ddir = os.path.join(outdir, stem, key)
        os.makedirs(ddir, exist_ok=True)
        lite = []
        for k, c in enumerate(cases):
            fn = os.path.join(ddir, f"case{k}.json")
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(dict(u=c["u"], tau=c["tau"]), f)
            files.append(fn)
            lite.append({**{kk: v for kk, v in c.items() if kk not in ("u", "tau")}, "src": f"{stem}/{key}/case{k}.json"})
        data.append(dict(key=key, label=label, note=note, **meta, cases=lite))
    blob = json.dumps(dict(scenes=data, links=links), ensure_ascii=False, separators=(",", ":"))
    with open(out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("__TITLE__", title).replace("__DATA__", blob.replace("</", "<\\/")))
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(prog="floodsim.web3d_hub", description="รวมหน้า 3D หลายลำน้ำเป็นหน้าหลัก")
    ap.add_argument("--scene", nargs=3, action="append", required=True, metavar=("KEY=HTML", "LABEL", "NOTE"),
                    help="ลำน้ำหนึ่ง: key=ไฟล์หน้า 3D, ชื่อปุ่ม, คำอธิบายสั้น (ใส่ได้หลายครั้ง; อันแรกเป็นค่าเริ่มต้น)")
    ap.add_argument("--links", default=None, help="ไฟล์ JSON รายการลิงก์เมนูหน้าอื่น")
    ap.add_argument("--title", default="เทียบเรือดันน้ำทุกแบบ")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    scenes = []
    for kv, label, note in a.scene:
        key, html = kv.split("=", 1)
        scenes.append((key, html, label, note))
    # อ่านหน้าเดิมทั้งหมดก่อนเขียน — หน้าออกอาจเป็นไฟล์เดียวกับหน้าใดหน้าหนึ่ง
    links = json.load(open(a.links, encoding="utf-8")) if a.links else []
    files = build(scenes, a.out, a.title, links)
    size = sum(os.path.getsize(f) for f in files)
    print(f"saved {os.path.abspath(a.out)} ({len(files)} ไฟล์, {size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
